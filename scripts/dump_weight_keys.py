"""Print the checkpoint's tensor inventory, collapsing layer indices.

Every module in `qtts/models` was written against this output rather than
against the reference code's defaults, because the two disagree in places (the
config class default for `codec_eos_token_id` is not this checkpoint's value).
Run it after changing checkpoints to see what the loader has to cover.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--shapes", action="store_true", help="also read shapes from the safetensors headers")
    args = parser.parse_args()

    root = Path(args.model_path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        from safetensors import safe_open

        weight_map = {}
        for shard in sorted(root.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as f:
                weight_map.update({k: shard.name for k in f.keys()})

    shapes: dict[str, list[int]] = {}
    if args.shapes:
        from safetensors import safe_open

        for shard in sorted({v for v in weight_map.values()}):
            with safe_open(str(root / shard), framework="pt") as f:
                for key in f.keys():
                    shapes[key] = list(f.get_slice(key).get_shape())

    keys = sorted(weight_map)
    print(f"total tensors: {len(keys)}")

    def collapse(key: str) -> str:
        return re.sub(r"\.\d+\.", ".N.", key)

    counts = collections.Counter(collapse(k) for k in keys)
    example = {}
    for k in keys:
        example.setdefault(collapse(k), k)

    by_prefix: dict[str, list[str]] = collections.defaultdict(list)
    for pattern in counts:
        by_prefix[pattern.split(".")[0]].append(pattern)

    for prefix in sorted(by_prefix):
        owned = sum(1 for k in keys if k.split(".")[0] == prefix)
        print(f"\n===== {prefix} ({owned} tensors)")
        for pattern in sorted(by_prefix[prefix]):
            shape = f"  {shapes[example[pattern]]}" if args.shapes else ""
            print(f"  {counts[pattern]:>4}x  {pattern}{shape}")


if __name__ == "__main__":
    main()
