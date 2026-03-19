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
from pathlib import Path

import requests
from dotenv import load_dotenv
from pypinyin import lazy_pinyin, Style

_g2pw = None
_use_g2pw = True


def init_pinyin(use_g2pw: bool = True):
    """Initialize pinyin backend. Call before using get_pinyin()."""
    global _g2pw, _use_g2pw
    _use_g2pw = use_g2pw
    if use_g2pw:
        from pypinyin_g2pw import G2PWPinyin
        _g2pw = G2PWPinyin()


def get_pinyin(text: str) -> list[str]:
    """Get pinyin with guaranteed 1:1 character-to-pinyin mapping.

    When using g2pw: context-aware (handles polyphones like 樂=yuè in 樂團 vs lè in 快樂).
    pypinyin-g2pw collapses consecutive non-Chinese characters into single tokens
    (e.g. 'ABC' -> ['ABC']). This function expands them back to one element per character.

    When using plain pypinyin: no context awareness but no g2pw model dependency.
    """
    if _use_g2pw:
        raw = _g2pw.lazy_pinyin(text, style=Style.NORMAL)
    else:
        raw = lazy_pinyin(text, style=Style.NORMAL)
    if len(raw) == len(text):
        return raw
    result = []
    pos = 0
    for token in raw:
        if pos >= len(text):
            break
        # Check if this token is a literal substring (collapsed non-Chinese chars)
        tlen = len(token)
        if tlen > 1 and text[pos:pos + tlen] == token:
            # Expand: each character gets itself as its "pinyin"
            for ch in token:
                result.append(ch)
            pos += tlen
        else:
            result.append(token)
            pos += 1
    # Append any remaining characters that g2pw didn't cover
    while pos < len(text):
        result.append(text[pos])
        pos += 1
    return result


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


def is_homophone(wrong: str, correct: str, wrong_pinyin: list[str] | None = None, correct_pinyin: list[str] | None = None) -> bool:
    """Check if two strings are homophones using pypinyin.

    Only checks CHANGED characters — all changed chars must have
    similar pronunciation for this to be considered a homophone fix.

    If wrong_pinyin/correct_pinyin are provided (from context-aware g2pw),
    use those instead of per-character lookup.
    """
    if len(wrong) != len(correct):
        return False
    if wrong == correct:
        return False

    # Get pinyin without tones
    w_py = wrong_pinyin if wrong_pinyin and len(wrong_pinyin) == len(wrong) else get_pinyin(wrong)
    c_py = correct_pinyin if correct_pinyin and len(correct_pinyin) == len(correct) else get_pinyin(correct)

    if len(w_py) != len(c_py):
        return False

    # Only check characters that actually changed
    changed_count = 0
    similar_count = 0

    for wc, cc, wp, cp in zip(wrong, correct, w_py, c_py):
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

    # Get context-aware pinyin for full strings (handles polyphones)
    t_py = get_pinyin(t)
    r_py = get_pinyin(r)

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
            # Substitution - check if homophone using context-aware pinyin
            tc, rc = t[i-1], r[j-1]
            tp = t_py[i-1]
            rp = r_py[j-1]
            if tp == rp:  # Require exact pinyin match for ref alignment (stricter = fewer false positives)
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
    # Common function words that should never be substituted by ref alignment
    # These are too frequent and alignment noise causes false positives
    PROTECTED_CHARS = set('是的了在有我他她它不這那就都也要會可以')

    ref_text = " ".join(r.text for r in ref_subs_list)
    subs = align_and_find_homophone_subs(transcribed, ref_text)

    result = transcribed
    for wrong_char, correct_char in subs:
        if wrong_char in PROTECTED_CHARS:
            continue
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


def extract_entity_mappings(ref_subs: list[Sub], input_subs: list[Sub], api_key: str, model: str) -> dict[str, str]:
    """Extract named entity variant mappings using LLM.

    Sends both reference and transcription text to LLM, asks it to identify
    entity names in the reference and find their ASR-mangled variants in the
    transcription. Returns a dict mapping wrong_form -> correct_form.
    """
    # Collect Chinese reference lines
    chinese_lines = []
    for r in ref_subs:
        text = re.sub(r'<[^>]+>', '', r.text).strip()
        text = re.sub(r'\{\\[^}]+\}', '', text).strip()
        if not re.search(r'[\u4e00-\u9fff]', text):
            continue
        kana_count = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        if kana_count > len(text) * 0.5:
            continue
        if len(text) > 60:
            continue
        chinese_lines.append(text)

    if not chinese_lines:
        return {}

    seen = set()
    unique_ref = []
    for line in chinese_lines:
        if line not in seen:
            seen.add(line)
            unique_ref.append(line)

    ref_text = "\n".join(unique_ref[:150])

    # Collect transcription lines
    trans_lines = []
    seen_t = set()
    for s in input_subs:
        text = s.text.strip()
        if text and text not in seen_t and re.search(r'[\u4e00-\u9fff]', text):
            seen_t.add(text)
            trans_lines.append(text)

    trans_text = "\n".join(trans_lines[:200])

    prompt = (
        "以下有兩組字幕：「參考字幕」是正確的翻譯，「ASR轉錄」是語音識別的結果。\n"
        "ASR經常把專有名詞（角色名、地名、種族名、技能名等）寫錯，用了發音相近但字不同的漢字。\n\n"
        "任務：\n"
        "1. 先從參考字幕中找出所有專有名詞\n"
        "2. 再從ASR轉錄中找出這些名詞的錯誤寫法（發音相近但漢字不同）\n"
        "3. 回覆JSON物件，key=ASR錯誤寫法，value=參考字幕正確寫法\n\n"
        "規則：\n"
        "- 只替換專有名詞，不要替換普通詞語（如「頭目」「主人」「嚮導」等不是專有名詞）\n"
        "- ASR錯誤和正確寫法的發音必須相近（如 立魔路≈利姆路，因為每個字的聲母相同）\n"
        "- 包含所有變體，即使只出現一次\n"
        "- 如果沒有錯誤回覆 {}\n\n"
        f"參考字幕：\n{ref_text}\n\n"
        f"ASR轉錄：\n{trans_text}"
    )

    messages = [{"role": "user", "content": prompt}]
    for attempt in range(3):
        try:
            content = call_openrouter(messages, api_key, model, max_tokens=1000)
            break
        except Exception as e:
            if attempt < 2:
                print(f"  Entity mapping attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(2)
            else:
                print(f"  Entity mapping failed after 3 attempts: {e}")
                return {}
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```json\s*|^```\s*|```$", "", content, flags=re.MULTILINE).strip()

    try:
        mappings = json.loads(content)
        if isinstance(mappings, dict):
            # Filter: both keys and values must be strings of length >= 2
            result = {}
            for k, v in mappings.items():
                if isinstance(k, str) and isinstance(v, str) and len(k) >= 2 and len(v) >= 2 and k != v:
                    result[k] = v
            return result
    except json.JSONDecodeError:
        pass
    return {}


def apply_entity_mappings(text: str, mappings: dict[str, str]) -> str:
    """Apply named entity replacement mappings to text."""
    for wrong, correct in mappings.items():
        if wrong in text:
            print(f"    [entity] {wrong}→{correct}")
            text = text.replace(wrong, correct)
    return text


def _get_initial(py: str) -> tuple[str, str]:
    """Extract initial and final from a pinyin syllable."""
    for init in ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                  'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']:
        if py.startswith(init):
            return init, py[len(init):]
    return '', py


def fix_named_entities_auto(text: str, entities: list[str], ref_subs: list[Sub]) -> str:
    """Fix ASR-mangled named entities by automated matching against entity list.

    Uses lenient matching: all changed characters must share the same initial
    consonant or have similar pinyin, and at least one character must match exactly.
    """
    ref_text = " ".join(r.text for r in ref_subs)
    entity_set = set(entities)

    sorted_entities = sorted(entities, key=len, reverse=True)

    for entity in sorted_entities:
        n = len(entity)
        if n < 3 or entity in text:
            continue
        if entity not in ref_text:
            continue

        entity_py = get_pinyin(entity)
        best_pos = -1
        best_exact = 0

        for pos in range(len(text) - n + 1):
            window = text[pos:pos + n]
            if window in entity_set:
                continue

            window_py = get_pinyin(window)

            exact = 0
            compatible = True
            for wc, ec, wp, ep in zip(window, entity, window_py, entity_py):
                if wc == ec:
                    exact += 1
                elif wp == ep or similar_pinyin(wp, ep):
                    pass
                else:
                    wi, _ = _get_initial(wp)
                    ei, _ = _get_initial(ep)
                    if wi and wi == ei:
                        pass
                    else:
                        compatible = False
                        break

            if compatible and exact >= 1 and exact > best_exact:
                best_exact = exact
                best_pos = pos

        if best_pos >= 0:
            old = text[best_pos:best_pos + n]
            print(f"    [entity-auto] {old}→{entity}")
            text = text[:best_pos] + entity + text[best_pos + n:]

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
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
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
    # Get context-aware pinyin for full sentence (handles polyphones like 樂=yuè in 樂團)
    sentence_py = get_pinyin(transcribed)

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
        # Extract context-aware pinyin for the wrong substring
        idx = result.find(wrong)
        wrong_py = sentence_py[idx:idx + len(wrong)]
        # Get context-aware pinyin for the correct string
        correct_py = get_pinyin(correct)
        # Verify it's actually a homophone using context-aware pinyin
        if not is_homophone(wrong, correct, wrong_pinyin=wrong_py, correct_pinyin=correct_py):
            print(f"    [rejected] {wrong}→{correct} (pinyin mismatch)")
            continue
        result = result.replace(wrong, correct, 1)
        # Update sentence pinyin after replacement
        sentence_py = get_pinyin(result)

    return result


def normalize_numbers(text: str) -> str:
    """Convert Chinese numerals to Arabic in number+counter patterns.

    E.g. 三年→3年, 六小時→6小時, 二十個→20個.
    Only converts when followed by a counter/unit word to avoid false positives.
    """
    cn_digits = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                 '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}

    def cn_to_arabic(s: str) -> int | None:
        """Convert a short Chinese numeral string to an integer (up to 99)."""
        if not s:
            return None
        if len(s) == 1:
            return cn_digits.get(s)
        # Handle 十X, X十, X十X patterns
        if s == '十':
            return 10
        if len(s) == 2 and s[0] == '十' and s[1] in cn_digits:
            return 10 + cn_digits[s[1]]
        if len(s) == 2 and s[0] in cn_digits and s[1] == '十':
            return cn_digits[s[0]] * 10
        if len(s) == 3 and s[0] in cn_digits and s[1] == '十' and s[2] in cn_digits:
            return cn_digits[s[0]] * 10 + cn_digits[s[2]]
        return None

    counters = r'(?:小時|分鐘|公里|公斤|[年月日天時分秒個位樓號班組排期屆次集歲杯瓶件張把隻條首頁篇章節回場局步塊圈人])'
    # Exclude 一 — in subtitles it almost always means "a/an" not "1"
    num_chars = r'[零二三四五六七八九十]'
    pattern = rf'({num_chars}{{1,3}})({counters})'

    # Pre-check: skip adverbial 十分 (means "very much", not "10 minutes")
    text = re.sub(r'十分(?=[受到的地得])', '___SHIFEN___', text)

    def replace_match(m):
        cn_str, counter = m.group(1), m.group(2)
        val = cn_to_arabic(cn_str)
        if val is not None:
            return f"{val}{counter}"
        return m.group(0)

    text = re.sub(pattern, replace_match, text)
    text = text.replace('___SHIFEN___', '十分')
    return text



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
    parser.add_argument("--no-g2pw", action="store_true", help="Use plain pypinyin instead of pypinyin-g2pw for pinyin conversion")
    parser.add_argument("--no-entities", action="store_true", help="Disable named entity extraction and replacement")
    args = parser.parse_args()

    init_pinyin(use_g2pw=not args.no_g2pw)

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

    entity_list = []
    entity_mappings = {}
    if not args.no_entities:
        print(f"\nExtracting named entities from reference...")
        # Extract entity list from reference
        chinese_ref_lines = []
        for r in ref_subs:
            text = re.sub(r'<[^>]+>', '', r.text).strip()
            text = re.sub(r'\{\\[^}]+\}', '', text).strip()
            if re.search(r'[\u4e00-\u9fff]', text) and len(text) <= 60:
                kana = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
                if kana <= len(text) * 0.5:
                    chinese_ref_lines.append(text)
        seen_ref = set()
        unique_ref_lines = [l for l in chinese_ref_lines if l not in seen_ref and not seen_ref.add(l)]
        entity_prompt = (
            "從以下動畫字幕中提取所有專有名詞（角色名、地名、組織名、種族名、技能名、怪物名等）。\n"
            "注意：\n"
            "- 只提取專有名詞，不要提取普通詞語\n"
            "- 每個名詞只列一次\n"
            "- 包含所有出現的人名，即使只出現一次（如 培斯塔、迦盧姆、葛洛姆）\n"
            "- 包含所有怪物/生物的名稱（如 盔甲龍）\n"
            "以JSON陣列格式回覆，例如：[\"利姆路\", \"德瓦崗\", \"培斯塔\"]\n\n"
            f"字幕文本：\n" + "\n".join(unique_ref_lines[:200])
        )
        for attempt in range(3):
            try:
                econtent = call_openrouter([{"role": "user", "content": entity_prompt}],
                                           api_key, args.model, max_tokens=800)
                econtent = re.sub(r"<think>.*?</think>", "", econtent, flags=re.DOTALL).strip()
                econtent = re.sub(r"^```json\s*|^```\s*|```$", "", econtent, flags=re.MULTILINE).strip()
                elist = json.loads(econtent)
                if isinstance(elist, list):
                    entity_list = [e for e in elist if isinstance(e, str) and len(e) >= 2]
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"  Entity list extraction failed: {e}")
        print(f"  Entity list ({len(entity_list)}): {', '.join(entity_list[:30])}")

        # Extract entity mappings (wrong ASR forms -> correct reference forms)
        from wordfreq import word_frequency
        COMMON_WORD_THRESHOLD = 1e-6  # words more frequent than this are common vocabulary
        print(f"\nExtracting entity mappings...")
        entity_mappings = extract_entity_mappings(ref_subs, input_subs, api_key, args.model)
        # Filter out common words using wordfreq, and require value appears in reference
        ref_full_text = " ".join(r.text for r in ref_subs)
        filtered_mappings = {}
        for k, v in entity_mappings.items():
            k_freq = word_frequency(k, 'zh')
            v_freq = word_frequency(v, 'zh')
            if k_freq > COMMON_WORD_THRESHOLD or v_freq > COMMON_WORD_THRESHOLD:
                print(f"  Filtered out common word mapping: {k}→{v} (freq: {k_freq:.2e}/{v_freq:.2e})")
                continue
            if v not in ref_full_text:
                print(f"  Filtered out (value not in reference): {k}→{v}")
                continue
            if len(k) > 6 or len(v) > 6:
                print(f"  Filtered out (too long): {k}→{v}")
                continue
            filtered_mappings[k] = v
        entity_mappings = filtered_mappings
        print(f"  Entity mappings ({len(entity_mappings)}): {entity_mappings}")
    else:
        print(f"\nNamed entity replacement disabled.")

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

            # Step 4: Named entity correction (LLM mappings)
            if entity_mappings:
                fixed = apply_entity_mappings(fixed, entity_mappings)

            # Step 5: Named entity correction (auto-matching to reference forms)
            if entity_list:
                fixed = fix_named_entities_auto(fixed, entity_list, closest)
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

    # Normalize number format (Chinese numerals → Arabic in number+counter patterns)
    print(f"\nNormalizing numbers...")
    for sub in corrected_subs:
        normalized = normalize_numbers(sub.text)
        if normalized != sub.text:
            print(f"  [{sub.index}] {sub.text} → {normalized}")
            sub.text = normalized

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
