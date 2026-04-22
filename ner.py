#!/usr/bin/env python3
"""Named-entity recognition on a Chinese SRT via an LLM (OpenRouter).

Distinct from normalize_entities.py: that script collapses ASR-variant
spellings of the SAME entity into one canonical form. This script simply
LISTS all named entities it finds in the transcription, grouped by type
(character, place, organization, …), with their mention counts and
representative context cues.

Usage:
    poetry run python3 ner.py input.srt
    poetry run python3 ner.py input.srt --json out.json

Requires OPENROUTER_API_KEY in the environment or .env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"


_PROMPT = """你是一名专业的中文字幕分析师。以下是一整集影视作品的中文字幕。

任务：找出字幕中出现的所有**命名实体**（Named Entities），给出一份完整清单。

**实体类别（只用下列之一）：**
- `character` —— 剧中角色的名字或姓氏（不含称谓后缀）。包括出现过的别名/昵称。
- `place` —— 地名（城市、街区、建筑、国家、地区、战场等专指名词）。
- `organization` —— 组织、军队、派系、学校、公司、社团、政府机构等专有组织名。
- `title` —— 剧中使用的头衔/封号（例：殿下、陛下、将军 —— 但只在它们被当作独立称呼时列出；不要拆日常对话中的一般敬语）。
- `technology` —— 剧中专有的科技产物或机械名（机甲、飞船、武器型号、技能名称等）。
- `other` —— 以上都不属于的专有名词。

**要求：**
1. 只列**命名实体**，不要日常词汇、普通动词、普通名词、称谓附加语等。
2. 每个实体只列一次（实体的 canonical 写法）；count 字段给出它在字幕中出现的总次数（包括复合形式，如「X殿下」也算 X 的一次出现）。
3. `mention_example` 是字幕中原句的一条简短引用，用来证明这个实体出现过。只取一句（≤ 50 字），能看出实体使用语境就够。
4. 按 count 从大到小排序。
5. 如果两个写法其实是同一个实体的不同拼法（前处理应已合并，但如果还漏了），就合并成一条，写进 `aliases` 数组。
6. 如果字幕中没有命名实体，回复 `{"entities": []}`。
7. **直接输出 JSON，不要代码块，不要 <think>，不要解释文字。**

回复 JSON 格式：
{
  "entities": [
    {
      "name": "规范写法",
      "type": "character|place|organization|title|technology|other",
      "count": 整数,
      "aliases": ["…其他拼法，如有"],
      "mention_example": "字幕中原句片段"
    }
  ]
}

字幕："""


_SRT_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        if not block.strip():
            continue
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if _SRT_TIME.search(line):
                body = re.sub(r"\s+", " ", "\n".join(lines[i + 1 :])).strip()
                cues.append({"text": body})
                break
    return cues


def _call_llm(
    transcript: str, *, model: str, api_key: str, verbose: bool = True
) -> list[dict]:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _PROMPT + "\n" + transcript}],
        "max_tokens": 8000,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    content: str | None = None
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if verbose:
                print(f"[ner] {model} responded in {time.time() - t0:.1f}s",
                      file=sys.stderr, flush=True)
            break
        except Exception as exc:
            if attempt < 2:
                if verbose:
                    print(f"[ner] attempt {attempt + 1} failed: {exc} — retrying",
                          file=sys.stderr, flush=True)
                time.sleep(2)
            else:
                print(f"[ner] failed after 3 attempts: {exc}", file=sys.stderr)
                return []

    assert content is not None
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"[ner] JSON parse failed: {exc}\n{content[:500]}", file=sys.stderr)
        return []

    entities = data.get("entities", data) if isinstance(data, dict) else data
    if not isinstance(entities, list):
        print(f"[ner] expected list, got {type(entities).__name__}", file=sys.stderr)
        return []
    return entities


def _recount(entities: list[dict], joined_text: str) -> list[dict]:
    """Verify/refresh each entity's count by actually scanning the text.

    The LLM's self-reported count is often a rough estimate; recounting
    from the text makes the final numbers reliable and lets us flag
    entities whose name doesn't actually appear (hallucinations).
    """
    out = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name")
        if not isinstance(name, str) or not name:
            continue
        names = [name] + [
            a for a in ent.get("aliases", [])
            if isinstance(a, str) and a and a != name
        ]
        total = sum(joined_text.count(n) for n in names)
        if total == 0:
            # LLM hallucinated a name not in the text — drop it
            continue
        ent = dict(ent)
        ent["count"] = total
        ent["aliases"] = [a for a in names[1:] if joined_text.count(a) > 0]
        out.append(ent)
    # Sort by count desc, then by type, then by name (stable for ties).
    out.sort(key=lambda e: (-e.get("count", 0), e.get("type", ""), e.get("name", "")))
    return out


def _print_report(entities: list[dict], stream=sys.stdout) -> None:
    if not entities:
        print("(no named entities found)", file=stream)
        return

    by_type: dict[str, list[dict]] = {}
    for ent in entities:
        by_type.setdefault(ent.get("type", "other"), []).append(ent)

    type_order = ["character", "place", "organization", "title", "technology", "other"]
    remaining = [t for t in by_type if t not in type_order]
    order = [t for t in type_order if t in by_type] + sorted(remaining)

    total_entities = len(entities)
    total_mentions = sum(e.get("count", 0) for e in entities)
    print(
        f"Found {total_entities} named entities "
        f"({total_mentions} total mentions)\n",
        file=stream,
    )

    for typ in order:
        group = by_type[typ]
        print(f"## {typ} ({len(group)})", file=stream)
        name_w = max(len(e.get("name", "")) for e in group)
        for ent in group:
            name = ent.get("name", "")
            count = ent.get("count", 0)
            aliases = ent.get("aliases", [])
            alias_str = f"  [{', '.join(aliases)}]" if aliases else ""
            example = ent.get("mention_example", "")
            print(
                f"  {name.ljust(name_w)}  ×{count:<3d}{alias_str}",
                file=stream,
            )
            if example:
                example = example if len(example) <= 60 else example[:57] + "..."
                print(f"    “{example}”", file=stream)
        print("", file=stream)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("input", type=Path, help="Input SRT file")
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help=f"OpenRouter model ID (default: {_DEFAULT_MODEL})")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path to write the full JSON entity list")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set (put it in .env or the environment).",
              file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    cues = parse_srt(args.input)
    if not cues:
        print(f"No cues parsed from {args.input}", file=sys.stderr)
        return 1

    # Dedup cue text to keep the prompt small; LLM doesn't need repeats.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in cues:
        t = c["text"].strip()
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    transcript = "\n".join(deduped)
    joined = "\n".join(c["text"] for c in cues)  # for recount

    entities = _call_llm(transcript, model=args.model, api_key=api_key)
    entities = _recount(entities, joined)

    _print_report(entities)

    if args.json:
        args.json.write_text(
            json.dumps({"entities": entities}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ner] wrote {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
