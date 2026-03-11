#!/usr/bin/env python3
"""Compare two SRT files using normalized Levenshtein distance (text only, no timestamps)."""

import re
import sys


def extract_text(srt_path):
    """Extract only the text lines from an SRT file, ignoring indices and timestamps."""
    with open(srt_path, encoding="utf-8") as f:
        content = f.read()
    # Remove index lines (standalone numbers) and timestamp lines
    lines = []
    for line in content.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->", line):
            continue
        lines.append(line)
    return " ".join(lines)


def normalized_levenshtein(s1, s2):
    """Compute normalized Levenshtein distance (0 = identical, 1 = completely different)."""
    if s1 == s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    if not len1 or not len2:
        return 1.0
    # Standard DP with two rows
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len2] / max(len1, len2)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file1.srt> <file2.srt>")
        sys.exit(1)

    text1 = extract_text(sys.argv[1])
    text2 = extract_text(sys.argv[2])

    distance = normalized_levenshtein(text1, text2)
    similarity = 1.0 - distance

    print(f"File 1: {sys.argv[1]} ({len(text1)} chars)")
    print(f"File 2: {sys.argv[2]} ({len(text2)} chars)")
    print(f"Normalized Levenshtein distance: {distance:.4f}")
    print(f"Similarity: {similarity:.4f} ({similarity * 100:.1f}%)")


if __name__ == "__main__":
    main()
