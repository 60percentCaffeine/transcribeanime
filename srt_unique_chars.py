#!/usr/bin/env python3
"""Count unique characters in an SRT file, ignoring timestamps and index lines."""

import re
import sys
from pathlib import Path


def extract_text(srt_path: str) -> str:
    text = Path(srt_path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\n+", text.strip())
    lines = []
    for block in blocks:
        block_lines = block.strip().splitlines()
        if len(block_lines) < 3:
            continue
        # Skip index (line 0) and timestamp (line 1)
        content = " ".join(block_lines[2:]).strip()
        content = re.sub(r"<[^>]+>", "", content)
        lines.append(content)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.srt>")
        sys.exit(1)

    text = extract_text(sys.argv[1])
    unique = sorted(set(text))
    print(f"Unique characters: {len(unique)}")
    print("".join(unique))


if __name__ == "__main__":
    main()
