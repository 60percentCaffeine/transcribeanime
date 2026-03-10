#!/usr/bin/env python3
"""
Fix ML-transcribed Mandarin subtitles using reference subtitles and an LLM.

For each transcribed subtitle line, finds the 3 closest reference lines by
timestamp, then asks an LLM to correct homophone/ASR errors using those
references as context.

Usage:
  ./fixtranscribeanime input.srt --reference reference.srt -o output.srt
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """\
你是一名專業的繁體中文字幕校對編輯。

你會收到：
1. 一行由語音識別（ASR）自動轉錄的中文字幕（「ASR轉錄」，可能有同音字錯誤）
2. 前後文（ASR轉錄的上下文，用>>>標記當前行）
3. 附近時間段的參考字幕（從其他語言翻譯而來，意思正確但措辭不同）

你的任務：
- 只修正當前ASR轉錄行中的同音字/近音字錯誤
- 參考翻譯字幕來理解正確含義，但不要照抄
- 全部使用繁體中文
- 修正後的文字長度應與原文相近

嚴格禁止：
- 絕對不要把前後文或參考字幕的內容合併到當前行
- 絕對不要讓輸出比原文長很多
- 絕對不要添加原文中沒有說的話
- 如果原文已正確，原封不動返回
- 只輸出一行文本，不要有任何解釋"""


@dataclass
class Sub:
    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_srt(path: str) -> list[Sub]:
    text = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\n+", text.strip())
    subs = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1],
        )
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1
        end = (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2
        content = re.sub(r"<[^>]+>", "", " ".join(lines[2:]).strip())
        subs.append(Sub(idx, start, end, content))
    return subs


def ms_to_srt(ms: int) -> str:
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def find_closest_refs(sub: Sub, refs: list[Sub], n: int = 3) -> list[Sub]:
    """Find n closest reference subs by start timestamp."""
    scored = []
    for r in refs:
        dist = abs(sub.start_ms - r.start_ms)
        scored.append((dist, r))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:n]]


def call_llm(
    transcribed: str,
    ref_lines: list[Sub],
    context_lines: list[str],
    api_key: str,
    model: str,
) -> str:
    ref_text = "\n".join(
        f"  [{ms_to_srt(r.start_ms)}] {r.text}" for r in ref_lines
    )
    ctx_text = "\n".join(f"  {line}" for line in context_lines)
    user_msg = (
        f"ASR轉錄：{transcribed}\n\n"
        f"前後文（ASR轉錄的上下文）：\n{ctx_text}\n\n"
        f"附近參考字幕：\n{ref_text}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    # Strip any thinking tags if present (some models output these)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    # Take only the first non-empty line
    for line in content.splitlines():
        line = line.strip()
        if line:
            return line
    return transcribed


def main():
    parser = argparse.ArgumentParser(
        description="Fix ML-transcribed Mandarin subtitles using reference subs + LLM"
    )
    parser.add_argument("input", help="Input SRT file (noisy ML transcription)")
    parser.add_argument("--reference", "-r", required=True, help="Reference SRT file")
    parser.add_argument("--output", "-o", required=True, help="Output corrected SRT")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls (default: 0.3s)")
    parser.add_argument("--ref-count", type=int, default=3, help="Number of reference lines per segment (default: 3)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set. See .env.example", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing input: {args.input}")
    input_subs = parse_srt(args.input)
    print(f"  {len(input_subs)} segments")

    print(f"Parsing reference: {args.reference}")
    ref_subs = parse_srt(args.reference)
    print(f"  {len(ref_subs)} segments")

    print(f"\nCorrecting with {args.model}...")
    corrected_subs = []
    for i, sub in enumerate(input_subs):
        closest = find_closest_refs(sub, ref_subs, n=args.ref_count)
        # Build surrounding context (2 lines before and after)
        context = []
        for j in range(max(0, i - 2), min(len(input_subs), i + 3)):
            marker = ">>>" if j == i else "   "
            context.append(f"{marker} {input_subs[j].text}")
        try:
            fixed = call_llm(sub.text, closest, context, api_key, args.model)
        except Exception as e:
            print(f"  [{i+1}/{len(input_subs)}] Error: {e} — keeping original")
            fixed = sub.text

        corrected_subs.append(Sub(sub.index, sub.start_ms, sub.end_ms, fixed))

        if fixed != sub.text:
            print(f"  [{i+1}/{len(input_subs)}] {sub.text} → {fixed}")
        else:
            print(f"  [{i+1}/{len(input_subs)}] {sub.text} (unchanged)")

        if args.delay > 0 and i < len(input_subs) - 1:
            time.sleep(args.delay)

    # Write output
    out_lines = []
    for sub in corrected_subs:
        out_lines.append(
            f"{sub.index}\n{ms_to_srt(sub.start_ms)} --> {ms_to_srt(sub.end_ms)}\n{sub.text}\n"
        )
    Path(args.output).write_text("\n".join(out_lines), encoding="utf-8")

    changed = sum(1 for a, b in zip(input_subs, corrected_subs) if a.text != b.text)
    print(f"\nDone! {changed}/{len(input_subs)} segments corrected.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
