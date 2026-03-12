#!/usr/bin/env python3
"""
Fix ML-transcribed Mandarin subtitles using reference subtitles and an LLM.

Two-step approach:
1. LLM identifies potential homophone errors
2. pypinyin verifies each fix is a true homophone (same/similar pronunciation)
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import requests
from dotenv import load_dotenv
from pypinyin import lazy_pinyin, Style

load_dotenv()

DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """\
你是一名專業的繁體中文字幕校對編輯，專門修正語音識別（ASR）產生的同音字/近音字錯誤。

你會收到一行ASR轉錄字幕、前後文、和附近的參考字幕（從日文翻譯而來）。

你的任務是找出ASR轉錄中的同音字/近音字錯誤，並以JSON格式回覆。

什麼是同音字錯誤：ASR聽到正確的發音，但寫成了發音相同或相似的不同漢字。
常見類型：
1. 普通同音字：「過重」→「國中」、「文化機」→「文化祭」、「離隊」→「裡蹲」
2. 成語/固定搭配被拆散：「最根就地」→「追根究底」（每個字都是近音字錯誤）
3. 日常詞彙被拆散：「避除」→「壁櫥」（bìchú，完全同音）

什麼不是同音字錯誤（絕對不要改）：
- 同義詞替換（辛苦→糟糕、留言→評論、回應→理睬、擺→放）
- 措辭不同但意思相同（人氣→熱門、樂團→樂隊）
- 語序調整、添加或刪除字詞

規則：
- 只找發音相同或相似但漢字寫錯的情況
- 修正的字和原字必須讀音相同或很接近
- 特別注意：是否有常見成語、四字格或詞語被ASR寫成了同音但不成詞的漢字組合
- 參考字幕只用來理解這句話應該表達什麼意思
- 如果原文沒有同音字錯誤，回覆空列表 []
- 寧可漏報也不要誤報
- 每個修正的「wrong」和「correct」字數必須相同

回覆格式（JSON陣列）：
[{"wrong": "錯誤的字", "correct": "正確的字"}]

正確例子：
- 過重→國中 ✓  - 文化機→文化祭 ✓  - 離隊→裡蹲 ✓
- 最根就地→追根究底 ✓（成語修正）  - 避除→壁櫥 ✓（同音詞）
- 登上→等上 ✓  - 反碎→粉碎 ✓

錯誤例子（不要改）：
- 辛苦→糟糕 ✗  - 留言→評論 ✗  - 回應→理睬 ✗
- 樂團→樂隊 ✗  - 爛歌→屎歌 ✗  - 試彈→翻彈 ✗"""


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


def clean_reference_subs(refs: list[Sub]) -> list[Sub]:
    """Filter out noisy reference lines (danmaku, on-screen text, etc.)."""
    cleaned = []
    for r in refs:
        text = r.text.strip()
        if not text:
            continue
        if len(text) > 60:
            continue
        if text.count("！") > 2 or text.count("!") > 2:
            continue
        if re.search(r'(?:233|2：17|站台|站上行|品川方向|京急本線|平潟灣)', text):
            continue
        cleaned.append(r)
    return cleaned


def find_refs_by_overlap(sub: Sub, refs: list[Sub], n: int = 5, window_ms: int = 5000) -> list[Sub]:
    scored = []
    for r in refs:
        overlap = max(0, min(sub.end_ms, r.end_ms) - max(sub.start_ms, r.start_ms))
        if overlap > 0:
            score = overlap
        else:
            dist = min(abs(sub.start_ms - r.end_ms), abs(sub.end_ms - r.start_ms))
            if dist <= window_ms:
                score = -dist
            else:
                continue
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:n]]


def is_homophone(wrong: str, correct: str) -> bool:
    """Check if two strings are homophones using pypinyin.

    Only checks CHANGED characters — all changed chars must have
    similar pronunciation for this to be considered a homophone fix.
    """
    if len(wrong) != len(correct):
        return False
    if wrong == correct:
        return False

    # Get pinyin without tones
    wrong_py = lazy_pinyin(wrong, style=Style.NORMAL)
    correct_py = lazy_pinyin(correct, style=Style.NORMAL)

    if len(wrong_py) != len(correct_py):
        return False

    # Only check characters that actually changed
    changed_count = 0
    similar_count = 0

    for wc, cc, wp, cp in zip(wrong, correct, wrong_py, correct_py):
        if wc == cc:
            continue  # unchanged char, skip
        changed_count += 1
        if wp == cp or similar_pinyin(wp, cp):
            similar_count += 1

    if changed_count == 0:
        return False

    # ALL changed characters must be phonetically similar
    return similar_count == changed_count


def similar_pinyin(a: str, b: str) -> bool:
    """Check if two pinyin syllables are similar enough to be confused by ASR."""
    if a == b:
        return True

    # Common ASR confusion groups for initials
    initial_groups = [
        {'zh', 'z', 'j'},
        {'ch', 'c', 'q'},
        {'sh', 's', 'x'},
        {'n', 'l'},
        {'r', 'l'},
        {'f', 'h'},
        {'b', 'p'},
        {'d', 't'},
        {'g', 'k'},
        {'j', 'q', 'x'},  # palatal group
        {'zh', 'ch', 'sh'},  # retroflex group
        {'z', 'c', 's'},  # alveolar group
    ]

    # Common ASR confusion groups for finals
    final_groups = [
        {'an', 'ang'},
        {'en', 'eng'},
        {'in', 'ing'},
        {'ian', 'iang'},
        {'uan', 'uang'},
        {'an', 'en'},  # commonly confused
        {'ui', 'un'},  # commonly confused
        {'iu', 'ou'},
        {'ei', 'i'},
        {'ong', 'eng'},
        {'iong', 'ong'},
        {'uo', 'o'},
        {'ai', 'ei'},
        {'ou', 'u'},
        {'ai', 'a'},
        {'ei', 'e'},
        {'ie', 'i'},
        {'ue', 'e'},
        {'ao', 'iao'},
    ]

    def get_initial(py):
        for init in ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                      'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']:
            if py.startswith(init):
                return init, py[len(init):]
        return '', py

    init_a, final_a = get_initial(a)
    init_b, final_b = get_initial(b)

    initials_match = init_a == init_b
    finals_match = final_a == final_b

    if initials_match and finals_match:
        return True

    initials_similar = any(init_a in g and init_b in g for g in initial_groups) if init_a and init_b else init_a == init_b
    finals_similar = finals_match or any(final_a in g and final_b in g for g in final_groups)

    if initials_match and finals_similar:
        return True
    if initials_similar and finals_match:
        return True

    return False


def align_and_find_homophone_subs(transcribed: str, ref_text: str) -> list[tuple[str, str]]:
    """Align transcription with reference text and find homophone substitutions.

    Uses edit distance DP to find character-level alignment, then identifies
    positions where characters differ but have similar pronunciation.
    Returns list of (wrong, correct) single-character substitutions.
    """
    # Remove spaces and punctuation from both for alignment
    def clean(s):
        return re.sub(r'[\s，。、！？「」…⋯\-（）()：]', '', s)

    t = clean(transcribed)
    r = clean(ref_text)

    if not t or not r:
        return []

    n, m = len(t), len(r)
    # DP for edit distance with backtracking
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if t[i-1] == r[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack to find alignment
    subs = []
    matches = 0
    total_ops = 0
    i, j = n, m
    while i > 0 and j > 0:
        if t[i-1] == r[j-1]:
            matches += 1
            total_ops += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j-1] + 1:
            total_ops += 1
            # Substitution - check if homophone
            tc, rc = t[i-1], r[j-1]
            tp = lazy_pinyin(tc, style=Style.NORMAL)[0]
            rp = lazy_pinyin(rc, style=Style.NORMAL)[0]
            if tp == rp or similar_pinyin(tp, rp):
                subs.append((tc, rc))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + 1:
            total_ops += 1
            i -= 1  # deletion
        else:
            total_ops += 1
            j -= 1  # insertion

    # Only trust alignment if enough characters match (good alignment quality)
    if total_ops == 0 or matches / total_ops < 0.4:
        return []

    return subs


def apply_ref_homophone_subs(transcribed: str, ref_subs_list: list[Sub]) -> str:
    """Apply homophone substitutions found by aligning with reference."""
    ref_text = " ".join(r.text for r in ref_subs_list)
    subs = align_and_find_homophone_subs(transcribed, ref_text)

    result = transcribed
    for wrong_char, correct_char in subs:
        if wrong_char in result and wrong_char != correct_char:
            result = result.replace(wrong_char, correct_char, 1)

    return result


def fix_common_homophones(text: str, ref_subs: list[Sub]) -> str:
    """Fix very common homophone confusions using reference context.

    These are characters that sound identical and ASR commonly confuses:
    他/她/它, 的/地/得, 在/再, etc.
    Only applies when the reference text clearly indicates which variant is correct.
    """
    ref_text = " ".join(r.text for r in ref_subs)

    # 他→它: both 'tā', check if reference uses 它
    if "他" in text and "它" in ref_text and "他" not in ref_text:
        text = text.replace("他", "它")
    elif "它" in text and "他" in ref_text and "它" not in ref_text:
        text = text.replace("它", "他")

    return text


def call_openrouter(messages: list[dict], api_key: str, model: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def get_fixes(
    transcribed: str,
    ref_lines: list[Sub],
    context_lines: list[str],
    api_key: str,
    model: str,
) -> list[dict]:
    ref_text = "\n".join(
        f"  [{ms_to_srt(r.start_ms)}] {r.text}" for r in ref_lines
    )
    ctx_text = "\n".join(f"  {line}" for line in context_lines)
    user_msg = (
        f"ASR轉錄：{transcribed}\n\n"
        f"前後文：\n{ctx_text}\n\n"
        f"附近參考字幕（翻譯自日文）：\n{ref_text}\n\n"
        f"請找出同音字/近音字錯誤，以JSON陣列格式回覆。如果沒有錯誤回覆 []"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    content = call_openrouter(messages, api_key, model)
    content = re.sub(r"^```json\s*|^```\s*|```$", "", content, flags=re.MULTILINE).strip()

    try:
        fixes = json.loads(content)
    except json.JSONDecodeError:
        return []

    if not isinstance(fixes, list):
        return []
    return fixes


def apply_fixes(transcribed: str, fixes: list[dict]) -> str:
    result = transcribed
    for fix in fixes:
        if not isinstance(fix, dict):
            continue
        wrong = fix.get("wrong", "")
        correct = fix.get("correct", "")
        if not wrong or not correct:
            continue
        if wrong not in result:
            continue
        # Reject if lengths differ
        if len(correct) != len(wrong):
            continue
        # Verify it's actually a homophone using pypinyin
        if not is_homophone(wrong, correct):
            print(f"    [rejected] {wrong}→{correct} (pinyin mismatch)")
            continue
        result = result.replace(wrong, correct, 1)

    return result


def deduplicate_segments(subs: list[Sub], window: int = 3) -> list[Sub]:
    """Remove duplicate and near-duplicate ASR segments.

    Checks each segment against nearby segments (within window) for:
    1. Exact duplicates (same text) — remove the later one
    2. High similarity (SequenceMatcher ratio >= 0.9) — remove the shorter one
    """
    if not subs:
        return subs

    to_remove = set()

    for i in range(len(subs)):
        if i in to_remove:
            continue
        for j in range(i + 1, min(i + window + 1, len(subs))):
            if j in to_remove:
                continue

            text_i = subs[i].text.strip()
            text_j = subs[j].text.strip()

            if not text_i or not text_j:
                continue

            # Exact duplicate
            if text_i == text_j:
                print(f"  [dedup] removing seg {subs[j].index} '{text_j}' (exact duplicate of seg {subs[i].index})")
                to_remove.add(j)
                continue

            # High similarity — remove the shorter one
            ratio = SequenceMatcher(None, text_i, text_j).ratio()
            if ratio >= 0.9:
                if len(text_i) <= len(text_j):
                    print(f"  [dedup] removing seg {subs[i].index} '{text_i}' (similar to seg {subs[j].index}, ratio={ratio:.2f})")
                    to_remove.add(i)
                    break
                else:
                    print(f"  [dedup] removing seg {subs[j].index} '{text_j}' (similar to seg {subs[i].index}, ratio={ratio:.2f})")
                    to_remove.add(j)

    result = [s for idx, s in enumerate(subs) if idx not in to_remove]
    if to_remove:
        print(f"  [dedup] removed {len(to_remove)} duplicate/near-duplicate segments")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fix ML-transcribed Mandarin subtitles using reference subs + LLM"
    )
    parser.add_argument("input", help="Input SRT file (noisy ML transcription)")
    parser.add_argument("--reference", "-r", required=True, help="Reference SRT file")
    parser.add_argument("--output", "-o", required=True, help="Output corrected SRT")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between API calls (default: 0.0s)")
    parser.add_argument("--workers", "-w", type=int, default=20, help="Number of parallel API workers (default: 20)")
    parser.add_argument("--ref-count", type=int, default=5, help="Number of reference lines per segment (default: 5)")
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
    print(f"  {len(ref_subs)} segments (raw)")

    ref_subs = clean_reference_subs(ref_subs)
    print(f"  {len(ref_subs)} segments (after cleaning)")

    print(f"\nCorrecting with {args.model} ({args.workers} workers)...")

    def process_sub(i: int, sub: Sub) -> tuple[int, Sub, str]:
        """Process a single subtitle. Returns (index, corrected_sub, original_text)."""
        closest = find_refs_by_overlap(sub, ref_subs, n=args.ref_count)
        context = []
        for j in range(max(0, i - 3), min(len(input_subs), i + 4)):
            marker = ">>>" if j == i else "   "
            context.append(f"{marker} {input_subs[j].text}")
        try:
            # Step 1: LLM-based homophone detection (primary model)
            fixes = get_fixes(sub.text, closest, context, api_key, args.model)
            if fixes:
                fixed = apply_fixes(sub.text, fixes)
            else:
                fixed = sub.text

            # Step 2: Reference-alignment homophone detection (catches what LLM misses)
            fixed = apply_ref_homophone_subs(fixed, closest)

            # Step 3: Common ASR homophone pairs (他/她/它, 的/地/得, etc.)
            fixed = fix_common_homophones(fixed, closest)
        except Exception as e:
            print(f"  [{i+1}/{len(input_subs)}] Error: {e} — keeping original")
            fixed = sub.text

        if args.delay > 0:
            time.sleep(args.delay)

        return i, Sub(sub.index, sub.start_ms, sub.end_ms, fixed), sub.text

    results = [None] * len(input_subs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sub, i, sub): i for i, sub in enumerate(input_subs)}
        for future in as_completed(futures):
            i, corrected, original = future.result()
            results[i] = corrected
            if corrected.text != original:
                print(f"  [{i+1}/{len(input_subs)}] {original} → {corrected.text}")
            else:
                print(f"  [{i+1}/{len(input_subs)}] {original} (unchanged)")

    corrected_subs = results

    # Deduplicate segments (remove exact/near-duplicate ASR segments)
    print(f"\nDeduplicating segments...")
    corrected_subs = deduplicate_segments(corrected_subs)

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
