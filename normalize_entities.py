#!/usr/bin/env python3
"""Document-internal named-entity normalization for ASR output.

Named entities (character names, places, organizations, foreign words)
are mentioned many times per episode. Acoustic noise causes ASR to scatter
their spellings — in Code Geass S01E03 the same Britannia comes out as
不列颠尼亚×3 and 布列颠尼亚×1, and Clovis as 克洛维斯×5 vs 克鲁维斯×1.
The canonical form is the within-document majority.

This step clusters every 3-to-max-char Han span across all cues by
pinyin, then rewrites minority variants to the majority-form canonical
— without needing a reference or context file. jieba is used to gate
the candidate clusters two ways: only spans whose boundaries align to
jieba token breaks in their cue are admitted (rejects 什么时 slicing
out of 什么|时候), and the canonical must look entity-like rather than
a common dictionary phrase (知道|了, 我|一定, 好|恐怖).

A second cousin-merging pass (3-char spans and up) absorbs clusters whose
pinyin differs by exactly one character-edit-≤1 syllable slot, catching
lu↔luo pairs like 克鲁维斯↔克洛维斯 and shu↔su pairs like 枢木朱雀↔
苏慕朱雀 that exact-pinyin clustering would miss. Tied cluster canonicals
are broken by preferring variants whose peak single-char jieba frequency
stays under a common-char cap (rejects 里瓦尔 where 里 is really a
grammar char), then by highest median char freq.

At 2 characters — too risky for cousin-merging — we instead consult
jieba's dictionary: a dict-known form with POS nr/nrt/ns/nt/nz or a
rare 'n' (freq ≤ 20) wins canonical status even when it's the minority
in the cluster, so 新宿 (freq 8) absorbs 心素×3 and 心宿×3. Variants
that are themselves jieba-known proper nouns (妮娜 nrt 3) are treated
as distinct entities and left untouched.

The step runs on the realigned SRT, before fixtranscribeanime — so the
downstream LLM sees consistent entity spellings and doesn't waste
homophone-fix budget on them.

--- LLM mode (--llm) ------------------------------------------------------

With `use_llm=True` (CLI: `--llm` on this script or `--llm-entities` on
transcribe.py), an OpenRouter LLM call is added on top of the heuristic.
The LLM receives the full deduped transcription and returns a JSON list
of `{canonical, variants[]}` entity clusters, which is then union-find
resolved across `llm_runs` consensus calls to pick one canonical per
component. Each variant is still filtered by pinyin edit distance, POS
tag, and jieba frequency — the LLM does not bypass safety gates.

The LLM mode catches non-homophone ASR errors the pinyin heuristic can't
touch (朱雀↔朱母, pinyin distance 2 on one slot), context-dependent
polyphone collapses (新宿↔星宿 where pypinyin reads 星宿 as a different
compound), and semantic links (小丽→夏利 diminutive → full name).

The prompt is fully generic — it teaches the pinyin-clustering mechanism
via common confusion pairs (菲/费, 利/丽/里, sh↔s, u↔uo, polyphones like
宿 sù/xiù) without naming any specific show. Because a generic prompt
sometimes picks the wrong canonical direction, the union with the
heuristic ruleset uses heuristic-wins-on-direction-conflict, and a
path-compression step collapses chained rules (LLM A→B + heuristic B→C
⇒ A→C). A final pinyin-near expansion scans for variants the LLM
missed using context-free per-char pinyin to defeat polyphones.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HAN_RE = re.compile(r"[㐀-䶿一-鿿\U00020000-\U0002a6df]+")

# Characters that are pure grammar/function — pronouns, copulas, particles,
# modals, common adverbs. Never appear inside a proper name.
# Crucially EXCLUDES 不, 下, 上, 里, 外, 中, 来, 去 — those start real
# named entities (不列顛, 下町, 上条, ...).
_FUNCTION_CHARS = frozenset(
    "的了是在有我你他她它們们這这那個个些么呢吗嗎吧啊呀嘛哦哎"
    "就才也沒没再還还把給给讓让對对和與与從从到向要會会得很最"
    "更又卻却啦喔哟哇嘻哈呵哼唉哪吗著过過"
)


from functools import lru_cache


@lru_cache(maxsize=8192)
def _get_pinyin_for_run(text: str) -> tuple[str, ...]:
    """Pinyin for a pure-Han run, one token per character.

    Uses plain pypinyin (not g2pw) — entities are mostly transliterations
    that don't hit the polyphones g2pw disambiguates, and g2pw's ~100ms
    per call would dominate this step's runtime. lru_cache handles the
    high rate of repeated short runs across an episode.
    """
    from pypinyin import lazy_pinyin, Style
    raw = lazy_pinyin(text, style=Style.NORMAL)
    if len(raw) == len(text):
        return tuple(raw)
    out = []
    pos = 0
    for tok in raw:
        if len(tok) > 1 and text[pos : pos + len(tok)] == tok:
            for ch in tok:
                out.append(ch)
            pos += len(tok)
        else:
            out.append(tok)
            pos += 1
    while pos < len(text):
        out.append(text[pos])
        pos += 1
    return tuple(out)


@lru_cache(maxsize=8192)
def _context_free_pinyin(text: str) -> tuple[str, ...]:
    """Context-free per-character pinyin — each char resolved independently.

    Needed because pypinyin/g2pw resolve polyphones based on surrounding
    context, which makes 星宿 (constellation compound → xīng xiù) look
    phonetically different from 新宿 (xīn sù), masking the fact that the
    ASR error is really a single-char 新↔星 confusion (both xīn). Per-char
    reading flattens the polyphone — 宿 always = sù — so entity-matching
    sees one-slot diff instead of two.
    """
    from pypinyin import lazy_pinyin, Style
    return tuple(lazy_pinyin(ch, style=Style.NORMAL)[0] for ch in text)


def _jieba_freq(word: str) -> int:
    import jieba
    # Trigger lazy init
    if not jieba.dt.initialized:
        jieba.initialize()
    return jieba.dt.FREQ.get(word, 0)


def _jieba_main_tag(word: str) -> str:
    """POS tag of the longest jieba segment of `word`."""
    import jieba.posseg as psg
    tags = list(psg.cut(word))
    if not tags:
        return ""
    return max(tags, key=lambda t: len(t.word)).flag


def _is_jieba_known_canonical(word: str) -> bool:
    """True when jieba recognises `word` as an entity-suitable canonical.

    Accept proper-noun POS (nr/nrt/ns/nt/nz); also accept generic-noun 'n'.
    In both cases the freq must stay under a length-dependent cap — 2-char
    forms get a tighter cap because jieba's POS tagger misfires on short
    reduplications and single-char-heavy combos (tagging 东东 as 'ns',
    矫健 as 'nr'), and 2-char entries are much more likely to be common
    Chinese vocabulary (发誓 v 315, 大叔 n 283, 社团 n 420).
    """
    freq = _jieba_freq(word)
    if freq < _JIEBA_KNOWN_THRESHOLD:
        return False
    tag = _jieba_main_tag(word)
    max_freq = 20 if len(word) == 2 else _JIEBA_COMMON_NOUN_MAX_FREQ
    if tag in _PROPER_NOUN_TAGS and freq <= max_freq:
        return True
    if tag == "n" and freq <= max_freq:
        return True
    return False


def _is_distinct_entity(word: str) -> bool:
    """A variant tagged as a proper noun AND present in jieba's dict is
    probably a separate named entity (妮娜=Nina vs 娜娜=Nunnally) and
    must NOT be merged into a pinyin-adjacent cluster sibling."""
    if _jieba_freq(word) < _JIEBA_KNOWN_THRESHOLD:
        return False
    return _jieba_main_tag(word) in _PROPER_NOUN_TAGS


_ENTITY_MULTICHAR_MAX_FREQ = 500
_ENTITY_SINGLECHAR_MAX_FREQ = 20_000
# A variant whose jieba dict frequency hits this threshold is treated as an
# authoritative canonical — it beats within-document majority voting. Handles
# place names like 新宿 (freq 8) that are minority-spelled in the document
# because ASR scattered them to 心素/心宿.
_JIEBA_KNOWN_THRESHOLD = 3
# jieba-known canonicals tagged with a generic noun ('n') must also be rare
# (freq ≤ this) — stops high-freq common words like 发誓/大叔/社团/晚餐 from
# hijacking a pinyin cluster.
_JIEBA_COMMON_NOUN_MAX_FREQ = 50
_PROPER_NOUN_TAGS = frozenset({"nr", "nrt", "ns", "nt", "nz"})
# POS tags for clearly-non-entity words — verbs/adjectives/adverbs/
# time/locative/direction/particle/measure/pronoun. A canonical tagged
# as any of these (or containing a segment tagged any of these) is
# almost certainly a common Chinese word that happens to share pinyin
# with a real entity, not the entity itself.
_NON_ENTITY_POS_TAGS = frozenset({
    "v", "vn", "vd", "vg",
    "a", "ad", "an", "ag",
    "d", "dg",
    "t", "f", "s",
    "p", "c", "u", "e", "y",
    "m", "q", "r",
})


def _has_non_entity_pos(word: str) -> bool:
    """Any jieba segment of `word` tagged as a clearly-non-entity POS?
    Checks all segments, not just the longest, so '小/a + 袋子/n' fails
    on the 'a' prefix even though the longer token is a noun."""
    import jieba.posseg as psg
    return any(p.flag in _NON_ENTITY_POS_TAGS for p in psg.cut(word))


def _looks_entity_like(word: str) -> bool:
    """True when `word`, tokenized by jieba on its own, looks like a named
    entity rather than a common Chinese phrase.

    An entity is either
      (a) tokenized into chunks where at least one multi-char chunk is a
          rare/unknown jieba dict entry (克洛维斯 → 克洛维|斯 : 克洛维
          freq 17; 不列颠尼亚 → 不列颠|尼亚 : 不列颠 freq 258), or
      (b) tokenized into all single-char chunks but none of them is a
          grammar word and every char has moderate or lower corpus
          frequency (鲁鲁修: 鲁=1594, 修=6007).

    A span composed entirely of common dict words (我|一定, 知道|了,
    好|恐怖: 恐怖 freq 1463) is a phrase, not an entity.
    """
    import jieba
    if not jieba.dt.initialized:
        jieba.initialize()
    tokens = list(jieba.cut(word, HMM=False))
    multichar = [t for t in tokens if len(t) >= 2]
    if multichar:
        # Real known entity — at least one jieba multi-char token is rare.
        # (不列颠 freq 258, 克洛维 freq 17 pass here even though they contain
        # highly frequent single-chars like 不 which would fail below.)
        return any(
            jieba.dt.FREQ.get(t, 0) < _ENTITY_MULTICHAR_MAX_FREQ for t in multichar
        )
    # All single-char tokens: jieba didn't find a known compound, so we
    # judge by the chars themselves — no function chars, and no
    # super-frequent char that's really just a grammar/cliché word in
    # disguise (里, 那, ...).
    if len(word) < 3:
        return False
    for ch in word:
        if ch in _FUNCTION_CHARS:
            return False
        if jieba.dt.FREQ.get(ch, 0) > _ENTITY_SINGLECHAR_MAX_FREQ:
            return False
    return True


def _han_runs(text: str) -> list[tuple[int, str]]:
    """Return list of (start_offset, han_run)."""
    return [(m.start(), m.group(0)) for m in _HAN_RE.finditer(text)]


def _syllable_char_edit(a: str, b: str) -> int:
    """Levenshtein on two pinyin syllable strings (≤ 6 chars each)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _one_syllable_off(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """True when two pinyin tuples differ in exactly one position AND the
    differing syllable pair is 1 char away (lu↔luo ✓, si↔sheng ✗).

    Same-length tuples only (callers bucket by character length).
    """
    if len(a) != len(b):
        return False
    diff_idx = -1
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            if diff_idx != -1:
                return False
            diff_idx = i
    if diff_idx == -1:
        return False
    return _syllable_char_edit(a[diff_idx], b[diff_idx]) <= 1


def _jieba_boundaries(text: str) -> set[int]:
    """Return set of character offsets inside `text` where a jieba token starts.

    The set always includes 0 and len(text). A span text[a:b] is "boundary-
    aligned" when both a and b are in the set — i.e. it does not cut a
    jieba-known compound (什么|时候) in half.
    """
    import jieba
    if not jieba.dt.initialized:
        jieba.initialize()
    out = {0}
    pos = 0
    for tok in jieba.cut(text, HMM=False):
        pos += len(tok)
        out.add(pos)
    return out


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_LLM_MODEL = "qwen/qwen3-235b-a22b-2507"

_LLM_PROMPT = """你是一名专业的中文字幕校对编辑。以下是一整集动画（或影视剧）的 ASR 语音识别转录字幕。

ASR 因为发音相近，会把同一个命名实体（角色名、姓氏、地名、组织名、外语音译词）写成几种不同的汉字组合。你的工作是把这些写法聚合到同一个实体下。

**核心流程：对每一个命名实体，先把它的拼音序列写出来，然后在整段转录里逐句扫描发音相近的汉字组合——每个字位置上同音或差一个声/韵母——把它们全部收进 variants。**

ASR 产生的同一实体多种写法，常见来源：
1. **同音字替换**：每个字的拼音完全相同，只是汉字不同。例如「菲 fēi / 费 fèi」「利 lì / 丽 lì / 里 lǐ」「洪 hóng / 红 hóng」「朱 zhū / 珠 zhū」。
2. **近音字替换**：某一字的声母或韵母差一个音。例如声母差异 l/n、f/h、z/zh、c/ch、s/sh、j/q、sh/s、n/l；韵母差异 u/uo、in/ing、an/ang、iu/iong、e/ei。
3. **多音字分裂**：同一个汉字本身有两种读音，ASR 在不同上下文里按不同读音听写，结果写成不同字。例如「宿」可读 sù 或 xiù；「行」可读 xíng 或 háng；「乐」可读 lè 或 yuè。这类错误要尤其注意，因为拼音序列会整体偏移一个字。
4. **连续轻辅音脱/加**：例如 xiao/xiǎo ↔ xiào ↔ jiào，shi ↔ si，tuan ↔ duan。

任务：
1. 通读整段转录，找出所有命名实体（角色名、姓氏、名字、地名、组织名、音译外语名、势力名、种族名、专有技能名等专有名词）。
2. 对每个实体，**主动** 在整个转录里搜索所有发音相似的汉字写法，全部列出。即使某个写法只出现一次也必须列出。**特别注意检查那些一出现就显得突兀、和上下文意思不搭的词**——它们通常就是同一实体的同音/近音误写。
3. 选定 canonical 的规则（按优先级从高到低）：
   (a) 最像一个真实的人名/地名/外语音译词（整体读着像一个专有名词，而不是日常词语的拼接）。
   (b) 避开碰巧撞上该拼音的高频普通词汇。如果某个写法作为普通中文词汇非常常见（例如「成天」「杀死」「社团」「落网」「家里」），那它几乎肯定不是 canonical —— 另一个同音/近音写法才是真实体。
   (c) 在同样合理的候选里，用转录中出现次数最多的那一个。
4. 回复 JSON 数组：[{"canonical": "...", "variants": ["...", ...]}, ...]

硬性要求（违反则整条作废）：
- **variants 必须与 canonical 字数完全相同**。带称谓/修饰的长形式（例如在名字后附加「殿下」「先生」「小姐」「兄」「姐」「哥哥」「同学」「一等」「大人」「将军」「队长」「家」「社」「殿」，或名字前缀「小」「老」「大」）绝不是裸名字的变体；两者字数不同，不要合并。
- 短子串也不是变体：如果实体是四字复合名，不要把三字子串当作变体。
- 只收录专有名词。普通词汇、动词、形容词、副词、时间词、方位词、称谓（如「殿下」「同学」「家里」「大叔」「成天」「落网」「晚餐」「遗物」「发誓」「杀死」「离席」「下次」「黄花」）一律不是命名实体。
- **不同角色可能拼音相近但是是独立的实体，绝不合并**。两个同音的名字分别是两个不同的角色（例如「妮娜 Nina」和「娜娜 Nunnally」），一个是甲角色一个是乙角色，各自在不同语境引入，千万不要当成同一个人的两种写法。判断：如果两个写法在转录里都以明确引入某个人的方式出现（如「我是 X」「X 是谁」「我叫 X」），它们多半是两个不同的实体。
- 变体和 canonical 每个位置的汉字读音必须相同或极近（单个音节 edit distance ≤ 2 on pinyin）。相差超过一个音节就不是变体。
- 只用转录中真实出现的字符串，不要臆造，也不要对字符做「合理化」修正。
- 如果转录中没有任何命名实体，回复 []。
- 直接输出 JSON 数组。不要代码块包装，不要 <think>，不要任何解释文字。

转录："""


def _extract_entities_llm(
    cues: list[dict],
    *,
    model: str,
    api_key: str,
    runs: int = 1,
    verbose: bool = True,
) -> dict[str, str]:
    """Ask the LLM to list entity variants across the whole transcription,
    optionally `runs` times. Unions the resulting (variant → canonical)
    maps; conflicts are resolved by majority vote (run-count), with ties
    broken by the first canonical seen.

    Every variant is still pinyin-gated against its canonical to keep the
    LLM from merging phonetically-distant pairs (鲁鲁兄 vs 鲁鲁修).
    """
    import requests

    seen_text: set[str] = set()
    deduped: list[str] = []
    for cue in cues:
        t = cue["text"].strip()
        if not t or t in seen_text:
            continue
        seen_text.add(t)
        deduped.append(t)
    body = "\n".join(deduped)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def one_call(run_idx: int) -> list:
        messages = [{"role": "user", "content": _LLM_PROMPT + "\n" + body}]
        # Small temperature on consensus runs so repeats aren't identical.
        temp = 0.0 if runs == 1 else (0.3 if run_idx > 0 else 0.0)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 8000,
            "temperature": temp,
        }
        for attempt in range(3):
            try:
                t0 = time.time()
                resp = requests.post(
                    _OPENROUTER_URL, headers=headers, json=payload, timeout=120
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                if verbose:
                    print(
                        f"[entnorm-llm] run {run_idx + 1}/{runs} {model} "
                        f"responded in {time.time() - t0:.1f}s (temp={temp})",
                        flush=True,
                    )
                break
            except Exception as exc:
                if attempt < 2:
                    if verbose:
                        print(
                            f"[entnorm-llm] attempt {attempt + 1} failed: {exc} — retrying",
                            flush=True,
                        )
                    time.sleep(2)
                else:
                    if verbose:
                        print(
                            f"[entnorm-llm] failed after 3 attempts: {exc}",
                            flush=True,
                        )
                    return []
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE
        ).strip()
        # Qwen occasionally sneaks a literal newline/tab into a JSON string;
        # strict json.loads rejects it. Strip control chars inside string
        # literals before parsing.
        content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            if verbose:
                print(
                    f"[entnorm-llm] JSON parse failed: {exc}\n{content[:400]}",
                    flush=True,
                )
            return []
        if not isinstance(parsed, list):
            if verbose:
                print(
                    f"[entnorm-llm] expected list, got {type(parsed).__name__}",
                    flush=True,
                )
            return []
        return parsed

    # Gather entity lists from all consensus runs.
    all_entities: list = []
    for i in range(runs):
        all_entities.extend(one_call(i))

    # Pre-compute the text body once for variant-existence filtering.
    joined_text = "\n".join(c["text"] for c in cues)

    # POS/freq helpers moved to module-level (_NON_ENTITY_POS_TAGS,
    # _has_non_entity_pos) so the expansion pass in normalize_cues can
    # reuse them.
    _LLM_REJECT_TAGS = _NON_ENTITY_POS_TAGS
    _has_reject_tag = _has_non_entity_pos

    # Union with majority canonical: each variant gets Counter of candidate
    # canonicals across runs. Resolve by most-seen canonical; ties fall to
    # first-seen order.
    variant_votes: dict[str, Counter] = defaultdict(Counter)
    first_seen_order: dict[str, int] = {}
    seen_ctr = 0
    replacements: dict[str, str] = {}
    kept_entities: list[tuple[str, list[str]]] = []
    for ent in all_entities:
        if not isinstance(ent, dict):
            continue
        canonical = ent.get("canonical")
        variants = ent.get("variants", [])
        if not isinstance(canonical, str) or not isinstance(variants, list):
            continue
        if not canonical or len(canonical) < 2:
            continue
        canonical_py = _get_pinyin_for_run(canonical)
        for v in variants:
            if not isinstance(v, str) or not v or v == canonical:
                continue
            if len(v) != len(canonical):
                continue
            if v not in joined_text:
                continue  # guard against hallucinated variants
            # Pinyin sanity filter: at most ONE character position may
            # disagree, and the disagreeing pair must be within 2 char
            # edits on the pinyin syllable. This lets 朱母↔朱雀 (mǔ↔què
            # edit 2), 星宿↔新宿 (xīng↔xīn edit 1), 十一军人↔十一区人
            # (jūn↔qū edit 2) through while rejecting the LLM's common
            # mistake of treating "name+honorific" as a name variant:
            # 鲁鲁兄↔鲁鲁修 differs at one slot but xiōng↔xiū is edit 3.
            var_py = _get_pinyin_for_run(v)
            diff_positions = [
                i for i, (a, b) in enumerate(zip(canonical_py, var_py))
                if a != b
            ]
            if len(diff_positions) > 1:
                continue
            if diff_positions and _syllable_char_edit(
                canonical_py[diff_positions[0]], var_py[diff_positions[0]]
            ) > 2:
                continue
            # Variant POS/freq filter: block LLM proposing common Chinese
            # vocabulary (下次 f, 小袋子 a/n composite, 杀死 v 1588,
            # 一般 n 30k, 离席 n 70) as variants of a name even when the
            # pinyin roughly matches.
            if _has_reject_tag(v):
                continue
            v_freq = _jieba_freq(v)
            if _jieba_main_tag(v) == "n" and v_freq > 50:
                continue
            # Variant must not itself be a jieba-known proper noun: that
            # would mean it's likely a distinct entity (妮娜=Nina vs
            # 娜娜=Nunnally) and the LLM mis-grouped them.
            if _is_distinct_entity(v):
                continue
            variant_votes[v][canonical] += 1
            if canonical not in first_seen_order:
                first_seen_order[canonical] = seen_ctr
                seen_ctr += 1

    # POS/freq filter: reject LLM proposals whose canonical is jieba-known
    # but not entity-shaped (verb 发誓, adverb 成天, common noun 社团,
    # adjective 细腻). Keeps real proper nouns like 朱雀 (nr freq 176)
    # even though they're too common for the heuristic path's stricter
    # 2-char cap — the LLM already did semantic filtering, we just
    # block the clearly-wrong categories here.
    def _canonical_acceptable(word: str) -> bool:
        if any(ch in _FUNCTION_CHARS for ch in word):
            return False
        tag = _jieba_main_tag(word)
        if tag in _LLM_REJECT_TAGS:
            return False
        # Generic 'n' with freq > 200 is almost never a proper noun
        # (社团/大叔/遗物/衣物 all n > 200).
        if tag == "n" and _jieba_freq(word) > 200:
            return False
        return True

    # Cycles are common when runs disagree on canonical direction (利瓦尔↔
    # 丽瓦尔, 休坦菲尔特↔休坦费尔特). Union-find everyone that ever voted
    # into the same component, then pick one canonical per component by
    # total incoming-vote mass.
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    canonical_mass: Counter = Counter()
    for variant, votes in variant_votes.items():
        for canon, n in votes.items():
            _union(variant, canon)
            canonical_mass[canon] += n

    # Group all forms by component root.
    all_forms = set(variant_votes) | {
        c for votes in variant_votes.values() for c in votes
    }
    component: dict[str, set[str]] = defaultdict(set)
    for f in all_forms:
        component[_find(f)].add(f)

    per_canonical: dict[str, list[str]] = defaultdict(list)
    for members in component.values():
        # Canonical = form with the most votes across the component.
        # Ties broken by earliest-seen order (stable), then lex.
        # Prefer forms that pass the POS/freq acceptability check — a
        # non-acceptable winner (verb/adverb/common noun) usually means
        # the LLM got the direction wrong (城田→成天 instead of 成天→城田).
        acceptable = [m for m in members if _canonical_acceptable(m)]
        pool = acceptable or list(members)
        canon = max(
            pool,
            key=lambda f: (
                canonical_mass[f],
                -first_seen_order.get(f, 1 << 30),
                f,
            ),
        )
        if not _canonical_acceptable(canon):
            continue  # no member passes — drop the whole component
        for m in members:
            if m == canon:
                continue
            # Re-run the pinyin gate for this pair — clustering can merge
            # forms via transitive votes that weren't directly filtered.
            canon_py = _get_pinyin_for_run(canon)
            m_py = _get_pinyin_for_run(m)
            if len(canon_py) != len(m_py):
                continue
            diffs = [
                i for i, (a, b) in enumerate(zip(canon_py, m_py)) if a != b
            ]
            if len(diffs) > 1:
                continue
            if diffs and _syllable_char_edit(
                canon_py[diffs[0]], m_py[diffs[0]]
            ) > 2:
                continue
            replacements[m] = canon
            per_canonical[canon].append(m)
    # Pinyin-near expansion happens after we union with heuristic rules
    # in normalize_cues — this LLM step only returns the consensus map.
    for canon, vs in per_canonical.items():
        kept_entities.append((canon, vs))

    if verbose:
        print(
            f"[entnorm-llm] {len(all_entities)} entity entries across {runs} run(s), "
            f"{len(kept_entities)} canonicals with variants, "
            f"{len(replacements)} variant rules",
            flush=True,
        )
        for canonical, vs in kept_entities:
            print(f"[entnorm-llm]   {canonical}  <-  {', '.join(vs)}", flush=True)

    return replacements


def _apply_replacements(
    cues: list[dict], replacements: dict[str, str]
) -> tuple[list[dict], Counter]:
    """Apply `replacements` (wrong→correct) to every cue, longest-first so
    3-char rules consume their own 2-char subsegments before shorter ones.
    Returns (new_cues, repl_counts)."""
    order = sorted(replacements.keys(), key=lambda s: (-len(s), s))
    new_cues: list[dict] = []
    repl_counts: Counter = Counter()
    for cue in cues:
        text = cue["text"]
        for wrong in order:
            if wrong not in text:
                continue
            correct = replacements[wrong]
            n = text.count(wrong)
            text = text.replace(wrong, correct)
            repl_counts[(wrong, correct)] += n
        new_cues.append({**cue, "text": text})
    return new_cues, repl_counts


def normalize_cues(
    cues: list[dict],
    *,
    min_len: int = 2,
    max_len: int = 6,
    min_occurrences: int = 2,
    plurality: float = 0.3,
    common_word_freq: int = 2000,
    use_llm: bool = False,
    llm_model: str = _DEFAULT_LLM_MODEL,
    llm_api_key: str | None = None,
    llm_runs: int = 1,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, str]]:
    """Return (new_cues, replacement_map) with minority-variant entity forms
    rewritten to their within-document majority form.

    Parameters
    ----------
    min_len, max_len : inclusive character-length window for candidate spans.
    min_occurrences  : a pinyin cluster needs ≥ this many total occurrences
                       before it's considered an entity candidate.
    plurality        : top form must account for ≥ this fraction of the
                       cluster's occurrences to be accepted as canonical.
    common_word_freq : clusters whose canonical form is in jieba's dict with
                       freq ≥ this threshold are skipped (likely not entities).
    """
    llm_replacements: dict[str, str] = {}
    if use_llm:
        if not llm_api_key:
            raise ValueError(
                "use_llm=True requires llm_api_key (or set OPENROUTER_API_KEY)."
            )
        llm_replacements = _extract_entities_llm(
            cues, model=llm_model, api_key=llm_api_key, runs=llm_runs,
            verbose=verbose,
        )

    # 1) Collect pinyin-tupled spans for every (length, pinyin) across cues.
    # Only admit spans whose character boundaries align with jieba tokens in
    # the cue — this rejects cross-boundary slices like "什么时" out of
    # "什么|时候".
    # clusters: {(L, pinyin_tuple): Counter({surface_form: count})}
    clusters: dict[tuple[int, tuple[str, ...]], Counter] = defaultdict(Counter)
    for cue in cues:
        text = cue["text"]
        boundaries = _jieba_boundaries(text)
        for run_start, run in _han_runs(text):
            run_py = _get_pinyin_for_run(run)
            R = len(run)
            for L in range(min_len, max_len + 1):
                if L > R:
                    break
                for i in range(R - L + 1):
                    abs_start = run_start + i
                    abs_end = abs_start + L
                    if abs_start not in boundaries or abs_end not in boundaries:
                        continue
                    form = run[i : i + L]
                    py = run_py[i : i + L]
                    clusters[(L, py)][form] += 1

    # 2) Merge near-homophone cousin clusters into their strongest exact
    # cluster. Example: 克洛维斯 ('ke luo wei si', ×5) absorbs 克鲁维斯
    # ('ke lu wei si', ×1) — the pinyin streams differ by one char insert.
    # Process clusters large-first so the bigger one always acts as the host.
    # Bucket by length so we only compare same-length clusters; sort each
    # bucket by strength so the biggest cluster always acts as the host.
    by_length: dict[int, list[tuple[int, tuple[str, ...]]]] = defaultdict(list)
    for key in clusters:
        by_length[key[0]].append(key)
    for bucket in by_length.values():
        bucket.sort(key=lambda k: -sum(clusters[k].values()))

    merged: dict[tuple[int, tuple[str, ...]], Counter] = {}
    consumed: set[tuple[int, tuple[str, ...]]] = set()
    for length, bucket in by_length.items():
        # Don't cousin-merge at 2 characters — one-syllable-off amounts to
        # a single-char edit on a very short string, and it sucks unrelated
        # 2-char words into entity clusters (新宿 accidentally absorbing
        # 心思 via su↔si).
        cousin_merge = length >= 3
        for key in bucket:
            if key in consumed:
                continue
            _, py = key
            forms = Counter(clusters[key])
            host_total = sum(forms.values())
            if not cousin_merge:
                merged[key] = forms
                continue
            for other in bucket:
                if other == key or other in consumed:
                    continue
                if not _one_syllable_off(py, other[1]):
                    continue
                other_total = sum(clusters[other].values())
                # A strictly-larger cousin would itself be the host; same-
                # size ties (1+1 → 2) are absorbed here because the bucket
                # iteration order determines host arbitrarily but the later
                # canonical-selection step tie-breaks on jieba char freqs.
                if other_total > host_total:
                    continue
                for form, cnt in clusters[other].items():
                    forms[form] += cnt
                consumed.add(other)
            merged[key] = forms

    # 3) Build variant -> canonical replacement map.
    replacements: dict[str, str] = {}
    cluster_report: list[tuple[str, int, dict[str, int]]] = []
    for (L, py), forms in merged.items():
        total = sum(forms.values())
        if total < min_occurrences:
            continue
        if len(forms) < 2:
            continue
        # Canonical selection happens in two layers:
        #
        # (a) If any form is in the jieba dictionary with freq ≥
        #     _JIEBA_KNOWN_THRESHOLD, treat it as authoritative — a real
        #     place/word that beats document-majority voting. 新宿 (freq 8)
        #     wins over 心素/心宿 even though 心宿 has 3x the occurrences.
        #     小狮子 (freq 3) wins over 孝世子/校狮子/叫世子 despite the
        #     cluster being a 4-way 1:1:1:1 tie the plurality check would
        #     otherwise reject.
        #
        # (b) Otherwise fall back to majority vote. Ties are broken by
        #     rejecting peak-outlier-char candidates (里瓦尔 where 里 freq
        #     77k is really a grammar char), then by highest median char
        #     freq (枢木朱雀 over 苏慕朱雀: 木 is more corpus-common than 慕).
        import jieba
        known = [
            f for f in forms
            if _is_jieba_known_canonical(f)
            and not any(ch in _FUNCTION_CHARS for ch in f)
        ]
        canonical: str | None = None
        if known:
            canonical = max(known, key=lambda f: (jieba.dt.FREQ.get(f, 0), f))
        else:
            top_count = max(forms.values())
            candidates = [f for f, c in forms.items() if c == top_count]
            if len(candidates) > 1:
                entity_candidates = [f for f in candidates if _looks_entity_like(f)]
                if entity_candidates:
                    candidates = entity_candidates
            if len(candidates) == 1:
                canonical = candidates[0]
            else:
                CAP = 30_000
                peak_ok = [f for f in candidates
                           if max(jieba.dt.FREQ.get(ch, 0) for ch in f) <= CAP]
                pool = peak_ok or candidates

                def _tiebreak(f: str) -> tuple[int, str]:
                    freqs = sorted(jieba.dt.FREQ.get(ch, 0) for ch in f)
                    return (freqs[len(freqs) // 2], f)

                canonical = max(pool, key=_tiebreak)
        canonical_freq = forms[canonical]
        # If jieba knows this word, it must be entity-shaped — otherwise
        # it's a common Chinese phrase that slipped through majority
        # voting (社团 n 420, 发誓 v 315, 抛出 v 3, 撤离 v 400, ...).
        canon_freq_jieba = _jieba_freq(canonical)
        if canon_freq_jieba >= _JIEBA_KNOWN_THRESHOLD and not _is_jieba_known_canonical(canonical):
            continue
        # Plurality gate only applies to anime-specific names (not
        # jieba-known) — for an in-dict canonical the dictionary is the
        # authority, so minority occurrence counts are fine.
        is_known = canon_freq_jieba >= _JIEBA_KNOWN_THRESHOLD
        if not is_known and canonical_freq / total < plurality:
            continue
        # Entities are all-content — no grammar chars inside. A canonical
        # containing 的/了/是/我/你/... is a phrase boundary artefact
        # (的事, 子的魔法, 外衣的), not a proper noun.
        if any(ch in _FUNCTION_CHARS for ch in canonical):
            continue
        # A canonical that's a high-frequency dictionary word is almost
        # never a named entity — skip the whole cluster.
        if _jieba_freq(canonical) >= common_word_freq:
            continue
        # Canonical must look entity-like: jieba must not see it as a
        # clean composition of common dict words (我|一定, 知道|了,
        # 好|恐怖 are phrases, not entities).
        if not _looks_entity_like(canonical):
            continue

        cluster_accepted: dict[str, int] = {canonical: canonical_freq}
        for variant, variant_freq in forms.items():
            if variant == canonical:
                continue
            # Block merges into/out of common words — both sides of the
            # swap must look entity-like.
            if _jieba_freq(variant) >= common_word_freq:
                continue
            # For anime-specific canonicals, never rewrite a variant to a
            # strictly LESS frequent form (ties are OK). For jieba-known
            # canonicals this check is intentionally skipped — the dict
            # entry is authoritative, so 心宿×3 absorbs into 新宿×1.
            if not is_known and variant_freq > canonical_freq:
                continue
            # Variants must not be grammar-heavy phrases in disguise
            # (你把/那就/她是/…) — those share pinyin with entity names by
            # accident.
            if any(ch in _FUNCTION_CHARS for ch in variant):
                continue
            # If the variant is itself a jieba-known proper noun, it's
            # probably a distinct character/place (妮娜=Nina is not an
            # ASR mistake for 娜娜=Nunnally even though they share pinyin).
            # Only skip this check when the canonical is the *same*
            # jieba-known entity (handled by variant == canonical above).
            if _is_distinct_entity(variant):
                continue
            # A common Chinese word shouldn't be merged into a small
            # entity cluster just because it shares pinyin — 杀死 (v 1588)
            # is not an ASR slip for the place 沙士 (n 3).
            variant_jieba_freq = _jieba_freq(variant)
            variant_cap = 50 if len(variant) == 2 else 500
            if variant_jieba_freq > variant_cap:
                continue
            # If the variant was already claimed by another cluster's
            # canonical, keep whichever claim has more evidence.
            prior_correct = replacements.get(variant)
            if prior_correct is not None and prior_correct != canonical:
                prior_key = (len(prior_correct), _get_pinyin_for_run(prior_correct))
                prior_freq = merged.get(prior_key, Counter()).get(prior_correct, 0)
                if prior_freq >= canonical_freq:
                    continue
            replacements[variant] = canonical
            cluster_accepted[variant] = variant_freq
        if len(cluster_accepted) >= 2:
            cluster_report.append((canonical, canonical_freq, cluster_accepted))

    # 3) Union LLM-proposed rules on top. LLM catches context-driven fixes
    # that the pinyin heuristic can't see (星宿↔新宿 edit 1 but 星/新 are
    # ambiguous out of context; 朱母↔朱雀 where mǔ/què are pinyin-distant).
    # If LLM disagrees with the heuristic on direction (LLM says A→B while
    # heuristic says B→A), the heuristic wins — document-majority voting
    # is more reliable than the LLM guessing on a generic prompt without
    # show-specific hints.
    for wrong, correct in llm_replacements.items():
        if replacements.get(correct) == wrong:
            continue  # heuristic already reversed this pair
        replacements[wrong] = correct

    # Path-compress: if A→B and B→C, collapse to A→C. Handles the case
    # where LLM sends several variants to a form the heuristic already
    # rewrites (LLM: 心素→心宿; heuristic: 心宿→新宿 → effective: 心素→新宿).
    for _ in range(5):  # short bound — fixpoint within 2-3 rounds typically
        changed = False
        for src, dst in list(replacements.items()):
            if dst in replacements and replacements[dst] != src:
                replacements[src] = replacements[dst]
                changed = True
        if not changed:
            break

    # Expansion: after LLM + heuristic + path-compression, scan the
    # transcription one more time for pinyin-near variants of every
    # final canonical. Catches 星宿→新宿 (context-free pinyin differs at
    # one char: xīng↔xīn) whether the canonical came from LLM or
    # heuristic. Uses context-free per-char pinyin to neutralise
    # polyphones (宿=sù always, not xiù-inside-星宿).
    canonical_targets = set(replacements.values())
    if canonical_targets:
        canonical_pinyin = {
            c: _context_free_pinyin(c) for c in canonical_targets
        }
        scanned: set[tuple[int, tuple[str, ...], str]] = set()
        for cue in cues:
            text = cue["text"]
            boundaries = _jieba_boundaries(text)
            for run_start, run in _han_runs(text):
                run_py = _context_free_pinyin(run)
                R = len(run)
                for canon, py_c in canonical_pinyin.items():
                    L = len(canon)
                    if L > R:
                        continue
                    for i in range(R - L + 1):
                        a, b = run_start + i, run_start + i + L
                        if a not in boundaries or b not in boundaries:
                            continue
                        span = run[i : i + L]
                        if span == canon or span in replacements:
                            continue
                        py_s = run_py[i : i + L]
                        key = (L, py_s, canon)
                        if key in scanned:
                            continue
                        scanned.add(key)
                        if len(py_c) != len(py_s):
                            continue
                        diffs = [
                            k for k, (x, y) in enumerate(zip(py_c, py_s))
                            if x != y
                        ]
                        if len(diffs) != 1:
                            continue
                        if _syllable_char_edit(
                            py_c[diffs[0]], py_s[diffs[0]]
                        ) > 2:
                            continue
                        if any(ch in _FUNCTION_CHARS for ch in span):
                            continue
                        if _is_distinct_entity(span):
                            continue
                        if _has_non_entity_pos(span):
                            continue
                        if _jieba_main_tag(span) == "n" and _jieba_freq(span) > 1000:
                            continue
                        replacements[span] = canon

    # 4) Apply replacements (longest-first).
    new_cues, repl_counts = _apply_replacements(cues, replacements)

    if verbose:
        total_edits = sum(repl_counts.values())
        print(
            f"[entnorm] {len(replacements)} variant→canonical rules, "
            f"{total_edits} text edits across {len(cues)} cues",
            flush=True,
        )
        cluster_report.sort(key=lambda t: -t[1])
        for canonical, canonical_freq, members in cluster_report[:40]:
            others = ", ".join(
                f"{v}×{c}" for v, c in sorted(
                    ((v, c) for v, c in members.items() if v != canonical),
                    key=lambda t: -t[1],
                )
            )
            applied = sum(
                repl_counts[(v, canonical)] for v in members if v != canonical
            )
            print(
                f"[entnorm]   {canonical}×{canonical_freq}  <-  {others}"
                f"  (edits={applied})",
                flush=True,
            )

    return new_cues, replacements


# --- Minimal SRT I/O (mirrors transcribe.py's subset) -----------------------

_SRT_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})"
)


def _ts(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        if not block.strip():
            continue
        lines = block.split("\n")
        for i, line in enumerate(lines):
            tm = _SRT_TIME.search(line)
            if tm:
                start = _ts(*tm.group(1, 2, 3, 4))
                end = _ts(*tm.group(5, 6, 7, 8))
                body = re.sub(r"\s+", " ", "\n".join(lines[i + 1 :])).strip()
                cues.append({"start": start, "end": end, "text": body})
                break
    return cues


def _fmt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[dict], path: Path) -> None:
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{_fmt_ts(c['start'])} --> {_fmt_ts(c['end'])}", c["text"], ""]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument("--max-len", type=int, default=6)
    p.add_argument("--min-occurrences", type=int, default=2)
    p.add_argument("--plurality", type=float, default=0.3)
    p.add_argument("--common-word-freq", type=int, default=2000)
    p.add_argument(
        "--llm",
        action="store_true",
        help="Use an LLM (OpenRouter) to detect all named entities instead "
        "of the pinyin/jieba heuristic. Requires OPENROUTER_API_KEY.",
    )
    p.add_argument("--llm-model", default=_DEFAULT_LLM_MODEL,
                   help=f"OpenRouter model ID. Default: {_DEFAULT_LLM_MODEL}")
    p.add_argument("--llm-runs", type=int, default=3,
                   help="Consensus runs: call the LLM N times (temperature "
                        "rises after the first run) and union the results. "
                        "LLM output is non-deterministic so a single run "
                        "often misses half the entities. Default: 3.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    api_key = None
    if args.llm:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("--llm requires OPENROUTER_API_KEY (set it in .env or the environment).",
                  file=sys.stderr)
            return 2

    cues = parse_srt(args.input)
    new_cues, _ = normalize_cues(
        cues,
        min_len=args.min_len,
        max_len=args.max_len,
        min_occurrences=args.min_occurrences,
        plurality=args.plurality,
        common_word_freq=args.common_word_freq,
        use_llm=args.llm,
        llm_model=args.llm_model,
        llm_api_key=api_key,
        llm_runs=args.llm_runs,
        verbose=not args.quiet,
    )
    write_srt(new_cues, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
