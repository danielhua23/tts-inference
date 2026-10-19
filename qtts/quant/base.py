"""Quantization dispatch: pick a method, attach it to the chosen linears.

The shape of this module is the one thing worth explaining. A quant method is
not a layer subclass; it is an object assigned to ``LinearBase.quant_method``
after the checkpoint is loaded. That ordering is deliberate: the loader writes
real weights into the fp16/bf16 parameter, and only then does
:meth:`QuantMethodBase.process_weights` replace them with quantized buffers.
Quantizing at construction time would mean quantizing uninitialized memory,
which does not error -- it produces scales computed from garbage, and the model
still runs.

Selection happens per submodel and per layer name, because the three submodels
have nothing in common numerically:

* the talker is a 1.7B decode loop at batch 1-64, so it is bandwidth-bound and
  weight-only quantization is the variant that should pay;
* the code predictor is 5 layers re-prefilled 15 times per frame, where the
  bottleneck is launch count rather than bytes, so quantizing it can easily be
  slower;
* Code2Wav is convolutional and loads in fp32, and its linears are a handful of
  attention projections inside the pre-transformer.

``QuantConfig.exclude_substrings`` keeps the output heads in full precision.
That is not conservatism: the codec head's logits are sampled from, so an error
there changes which code is emitted rather than moving a score slightly, and
the change is audible while the loss on any aggregate metric is invisible.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import QuantConfig
from ..layers.linear import LinearBase, iter_linears
from ..utils import get_logger, human_bytes

logger = get_logger("quant")


class QuantMethodBase:
    """Base for the assign-after-load quant methods."""

    name = "base"

    def process_weights(self, layer: LinearBase) -> None:
        raise NotImplementedError

    def applies_to(self, layer: LinearBase) -> bool:
        """Whether ``process_weights`` actually converted this layer.

        A method may decline a layer -- :class:`~qtts.quant.int8.
        Int8WeightOnlyMethod` declines the shapes where its kernel measures
        slower than cuBLAS. Declining has to leave ``quant_method`` unset so the
        layer keeps its original GEMM, which is why the decision is reported
        here rather than by raising.
        """
        return True

    def apply(
        self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        raise NotImplementedError


def _param_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters(recurse=True))


def _buffer_bytes(module: nn.Module) -> int:
    return sum(b.numel() * b.element_size() for b in module.buffers(recurse=True))


def quantize_module(module: nn.Module, method: QuantMethodBase, config: QuantConfig, label: str) -> dict[str, int]:
    """Attach ``method`` to every eligible :class:`LinearBase` under ``module``.

    Returns the byte counts before and after so the caller can report what the
    method actually bought, which is the only honest way to describe it: a
    method that quantizes three layers out of a hundred is a no-op with a
    plausible name.
    """
    before = _param_bytes(module) + _buffer_bytes(module)
    converted = 0
    excluded = 0
    declined = 0
    for name, layer in iter_linears(module):
        if config.excludes(name):
            excluded += 1
            continue
        method.process_weights(layer)
        if not method.applies_to(layer):
            declined += 1
            continue
        layer.quant_method = method
        converted += 1
    after = _param_bytes(module) + _buffer_bytes(module)
    logger.info(
        "%s: %s on %d linears (%d excluded by name, %d declined as not faster), %s -> %s",
        label,
        method.name,
        converted,
        excluded,
        declined,
        human_bytes(before),
        human_bytes(after),
    )
    return {
        "converted": converted,
        "excluded": excluded,
        "declined": declined,
        "bytes_before": before,
        "bytes_after": after,
    }


def build_method(config: QuantConfig, calibration_batch: int = 16) -> QuantMethodBase:
    from .fp8 import Fp8Method, Fp8WeightOnlyMethod
    from .int8 import Int8Method, Int8WeightOnlyMethod

    method = config.method
    if method in ("fp8", "fp8_weight_only"):
        supported, reason = fp8_supported()
        if not supported:
            raise RuntimeError(f"fp8 quantization unavailable: {reason}")
        return Fp8WeightOnlyMethod() if method.endswith("_weight_only") else Fp8Method()
    if method == "int8_weight_only":
        return Int8WeightOnlyMethod(calibrate=config.calibrate, calibration_batch=calibration_batch)
    if method == "int8":
        return Int8Method()
    raise ValueError(f"unknown quantization method {method!r}")


def fp8_supported() -> tuple[bool, str]:
    """Whether this GPU has the fp8 scaled GEMM, and why not if it does not.

    Checked up front rather than at the first matmul: torch's fp8 path fails
    deep inside a kernel launch, and by then the model is loaded and the run
    has started.
    """
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    if not hasattr(torch, "float8_e4m3fn"):
        return False, f"torch {torch.__version__} has no float8_e4m3fn dtype"
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        return False, (
            f"compute capability {major}.{minor} has no fp8 tensor cores "
            "(needs 8.9 / Ada or 9.0 / Hopper)"
        )
    if not hasattr(torch, "_scaled_mm"):
        return False, "torch._scaled_mm is missing"
    return True, ""


def apply_quantization(
    config: QuantConfig,
    *,
    talker: nn.Module | None = None,
    code_predictor: nn.Module | None = None,
    code2wav: nn.Module | None = None,
    calibration_batch: int = 16,
) -> dict[str, dict[str, int]]:
    """Quantize the submodels named in ``config.targets``. Call after loading.

    ``code_predictor`` follows the ``talker`` target rather than having its own,
    matching how the reference treats them as one stage. Quantizing only one
    half is possible with ``exclude_substrings`` and is not worth a flag.
    """
    if not config.enabled:
        return {}
    unknown = set(config.targets) - {"talker", "code_predictor", "code2wav"}
    if unknown:
        raise ValueError(f"unknown quant targets {sorted(unknown)}; pick from talker, code_predictor, code2wav")
    method = build_method(config, calibration_batch=calibration_batch)
    report: dict[str, dict[str, int]] = {}
    if config.targets_submodel("talker"):
        if talker is not None:
            report["talker"] = quantize_module(talker, method, config, "talker")
        if code_predictor is not None:
            report["code_predictor"] = quantize_module(code_predictor, method, config, "code_predictor")
    if config.targets_submodel("code2wav") and code2wav is not None:
        report["code2wav"] = quantize_module(code2wav, method, config, "code2wav")
    return report
