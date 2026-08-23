#!/usr/bin/env python3
"""Turn the prompt files into the `--prompt` arguments `leanlm submission` wants.

A prompt spanning twenty lines has to reach `metadata.json` as one JSON string
with its newlines intact. Retyping it into a shell command is how a trailing
line goes missing, and the loss is invisible: the JSON stays valid and the model
is simply asked something slightly different from what was tested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", default=None,
                        help="prompt files, in order (default: prompts/tp_*.txt)")
    parser.add_argument("--json", action="store_true",
                        help="emit the test_prompts array instead of CLI arguments")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files else \
        sorted(Path("prompts").glob("tp_*.txt"))
    if len(paths) != 2:
        print(f"expected exactly 2 prompt files, found {len(paths)}: "
              f"{[str(p) for p in paths]}")
        print("the template requires exactly two, and the organisers add two hidden")
        return 1

    prompts = []
    for index, path in enumerate(paths, start=1):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"{path} is empty")
            return 1
        prompts.append({"prompt_id": f"tp_{index:03d}", "prompt": text})

    if args.json:
        print(json.dumps(prompts, indent=2, ensure_ascii=False))
        return 0

    for entry in prompts:
        # Single quotes: the prompts contain double quotes and newlines, and a
        # shell is the wrong place to be careful about either.
        escaped = entry["prompt"].replace("'", "'\"'\"'")
        print(f"  --prompt '{escaped}' \\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
