#!/usr/bin/env python3
"""
Mandarin Subtitle Correction Pipeline
Corrects ML-transcribed Mandarin SRT using:
  1. Whisper re-transcription (audio-grounded)
  2. Japanese→Chinese translated SRT as meaning anchor
  3. Claude LLM correction pass
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class SubtitleSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    source: str = ""


@dataclass
class AlignedSegment:
    index: int
    start_ms: int
    end_ms: int
    noisy_zh: str  # ML transcription (wrong hanzi)
    ref_zh: str  # Translated subtitles (right meaning, different wording)
    whisper_zh: str = ""  # Whisper re-transcription
    corrected_zh: str = ""  # Final LLM-corrected output
    confidence: float = 1.0
    flagged: bool = False
    flag_reason: str = ""


# ─── SRT Parsing / Writing ────────────────────────────────────────────────────


def parse_srt(path: str) -> list[SubtitleSegment]:
    """Parse an SRT file into SubtitleSegment objects."""
    text = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\n+", text.strip())
    segments = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1],
        )
        if not time_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
        start_ms = (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1
        end_ms = (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2
        content = " ".join(lines[2:]).strip()
        # Strip HTML tags
        content = re.sub(r"<[^>]+>", "", content)
        segments.append(SubtitleSegment(idx, start_ms, end_ms, content))
    return segments


def ms_to_srt_time(ms: int) -> str:
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    segments: list[AlignedSegment], path: str, flagged_path: Optional[str] = None
):
    """Write corrected SRT. Optionally write a separate file with flagged segments."""
    lines = []
    flagged_lines = []

    for i, seg in enumerate(segments, 1):
        text = seg.corrected_zh or seg.noisy_zh
        block = (
            f"{i}\n"
            f"{ms_to_srt_time(seg.start_ms)} --> {ms_to_srt_time(seg.end_ms)}\n"
            f"{text}\n"
        )
        lines.append(block)
        if seg.flagged:
            flag_block = (
                f"{i}\n"
                f"{ms_to_srt_time(seg.start_ms)} --> {ms_to_srt_time(seg.end_ms)}\n"
                f"[FLAG: {seg.flag_reason}]\n"
                f"CORRECTED: {text}\n"
                f"ORIGINAL:  {seg.noisy_zh}\n"
                f"REFERENCE: {seg.ref_zh}\n"
            )
            flagged_lines.append(flag_block)

    Path(path).write_text("\n".join(lines), encoding="utf-8")

    if flagged_path and flagged_lines:
        Path(flagged_path).write_text("\n".join(flagged_lines), encoding="utf-8")
        print(f"  ⚑  {len(flagged_lines)} flagged segments written to: {flagged_path}")


# ─── Step 1: Align SRT Files ──────────────────────────────────────────────────


def overlap_ms(a: SubtitleSegment, b: SubtitleSegment) -> int:
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def align_segments(
    noisy: list[SubtitleSegment], ref: list[SubtitleSegment]
) -> list[AlignedSegment]:
    """
    Align noisy Mandarin transcription with reference (translated) subtitles
    by maximising timestamp overlap. One-to-many merges handled.
    """
    aligned = []
    ref_idx = 0

    for seg in noisy:
        # Find ref segments that overlap with this segment
        matching_refs = []
        for r in ref:
            if r.end_ms < seg.start_ms:
                continue
            if r.start_ms > seg.end_ms:
                break
            ov = overlap_ms(seg, r)
            if ov > 0:
                matching_refs.append(r)

        ref_text = " ".join(r.text for r in matching_refs) if matching_refs else ""

        aligned.append(
            AlignedSegment(
                index=seg.index,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                noisy_zh=seg.text,
                ref_zh=ref_text,
            )
        )

    print(
        f"  ✓  Aligned {len(aligned)} segments ({sum(1 for a in aligned if a.ref_zh)} with reference matches)"
    )
    return aligned


# ─── Step 2: Whisper Re-transcription ────────────────────────────────────────


def extract_audio_segment(video_path: str, start_ms: int, end_ms: int, out_path: str):
    """Extract a short audio clip using ffmpeg."""
    start_s = start_ms / 1000
    duration_s = (end_ms - start_ms) / 1000
    cmd = (
        f'ffmpeg -y -ss {start_s:.3f} -i "{video_path}" '
        f'-t {duration_s:.3f} -vn -ar 16000 -ac 1 -f wav "{out_path}" '
        f"-loglevel error"
    )
    os.system(cmd)


def whisper_transcribe(
    aligned: list[AlignedSegment], video_path: str, batch_size: int = 20
) -> list[AlignedSegment]:
    """Re-transcribe each segment using Whisper with reference as initial_prompt."""
    try:
        import whisper
    except ImportError:
        print("  ⚠  whisper not installed. Skipping re-transcription step.")
        print("     Install with: pip install openai-whisper")
        return aligned

    import tempfile

    print("  Loading Whisper model (large-v3)...")
    model = whisper.load_model("large-v3")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, seg in enumerate(aligned):
            audio_path = os.path.join(tmpdir, f"seg_{i}.wav")
            extract_audio_segment(video_path, seg.start_ms, seg.end_ms, audio_path)

            # Use reference text as a hint prompt (guides vocabulary selection)
            prompt = seg.ref_zh[:224] if seg.ref_zh else seg.noisy_zh[:224]

            try:
                result = model.transcribe(
                    audio_path,
                    language="zh",
                    initial_prompt=prompt,
                    fp16=False,
                )
                seg.whisper_zh = result["text"].strip()
            except Exception as e:
                seg.whisper_zh = seg.noisy_zh  # fallback

            if (i + 1) % 10 == 0:
                print(f"    Whisper: {i + 1}/{len(aligned)} segments")

    print(f"  ✓  Whisper re-transcription complete")
    return aligned


# ─── Step 3: LLM Correction Pass ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional Mandarin Chinese subtitle editor and linguist.

Your task is to produce a corrected Mandarin subtitle line given three sources of evidence:
1. A noisy ML transcription (likely contains wrong hanzi, especially homophones)
2. A Whisper re-transcription (more accurate but still may have errors)
3. A reference Chinese subtitle translated FROM Japanese (correct meaning but different phrasing)

Rules:
- Output ONLY the corrected Mandarin Chinese text, nothing else
- Fix wrong hanzi (especially ASR homophone errors) using the reference meaning as guide
- Prefer natural spoken Mandarin phrasing
- Keep the corrected text close to what was actually spoken (don't over-rewrite)
- If the three sources strongly disagree and you are uncertain, output the best guess and include a JSON confidence score
- Maximum one line of Mandarin text

Response format (JSON):
{"corrected": "corrected text here", "confidence": 0.0-1.0, "notes": "optional brief note if uncertain"}"""


def batch_correct_llm(
    aligned: list[AlignedSegment],
    batch_size: int = 5,
    api_key: Optional[str] = None,
    rate_limit_delay: float = 0.5,
) -> list[AlignedSegment]:
    """Call Claude API to correct each segment."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    print(f"  Correcting {len(aligned)} segments via Claude...")

    for i, seg in enumerate(aligned):
        user_msg = f"""Please correct this Mandarin subtitle segment.

NOISY ML TRANSCRIPTION: {seg.noisy_zh}
WHISPER RE-TRANSCRIPTION: {seg.whisper_zh or "(not available)"}
REFERENCE (translated from Japanese): {seg.ref_zh or "(not available)"}

Respond with JSON only."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            raw = re.sub(
                r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE
            ).strip()

            parsed = json.loads(raw)
            seg.corrected_zh = parsed.get("corrected", seg.noisy_zh)
            seg.confidence = float(parsed.get("confidence", 1.0))
            notes = parsed.get("notes", "")

            # Flag low-confidence corrections
            if seg.confidence < 0.75:
                seg.flagged = True
                seg.flag_reason = notes or f"Low confidence ({seg.confidence:.2f})"

        except json.JSONDecodeError:
            # LLM returned plain text instead of JSON — use it directly
            seg.corrected_zh = raw if raw else seg.noisy_zh
            seg.confidence = 0.7
        except Exception as e:
            print(f"    ⚠  Segment {i + 1} error: {e}")
            seg.corrected_zh = seg.noisy_zh
            seg.confidence = 0.0
            seg.flagged = True
            seg.flag_reason = f"API error: {e}"

        if (i + 1) % 10 == 0:
            print(f"    LLM: {i + 1}/{len(aligned)} segments")

        time.sleep(rate_limit_delay)

    flagged_count = sum(1 for s in aligned if s.flagged)
    print(f"  ✓  LLM correction complete. {flagged_count} segments flagged for review.")
    return aligned


# ─── Step 4: Quality Checks ───────────────────────────────────────────────────


def quality_check(aligned: list[AlignedSegment]) -> list[AlignedSegment]:
    """Apply heuristic quality checks and flag suspicious segments."""
    for seg in aligned:
        text = seg.corrected_zh or seg.noisy_zh

        # Flag empty output
        if not text.strip():
            seg.flagged = True
            seg.flag_reason = "Empty corrected text"
            continue

        # Flag if corrected text is dramatically longer/shorter than noisy input
        len_ratio = len(text) / max(len(seg.noisy_zh), 1)
        if len_ratio > 2.5 or len_ratio < 0.3:
            seg.flagged = True
            seg.flag_reason = f"Length ratio suspicious ({len_ratio:.1f}x)"

        # Flag if corrected text contains non-CJK/punctuation characters unexpectedly
        non_cjk = re.findall(r"[a-zA-Z]{3,}", text)
        if non_cjk:
            seg.flagged = True
            seg.flag_reason = f"Unexpected latin text: {non_cjk}"

    return aligned


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correct ML-transcribed Mandarin subtitles using Whisper + Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python correct_subtitles.py \\
    --noisy transcript_zh.srt \\
    --reference translated_from_jp.srt \\
    --video video.mp4 \\
    --output corrected.srt

  # Skip Whisper (SRT files only)
  python correct_subtitles.py \\
    --noisy transcript_zh.srt \\
    --reference translated_from_jp.srt \\
    --no-whisper \\
    --output corrected.srt

  # Use specific Anthropic API key
  python correct_subtitles.py ... --api-key sk-ant-...
        """,
    )
    parser.add_argument(
        "--noisy", required=True, help="Noisy ML Mandarin transcription (.srt)"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference subtitles translated from Japanese (.srt)",
    )
    parser.add_argument(
        "--video", default=None, help="Video/audio file for Whisper re-transcription"
    )
    parser.add_argument("--output", required=True, help="Output corrected SRT path")
    parser.add_argument(
        "--flagged",
        default=None,
        help="Output path for flagged segments (default: <output>.flagged.txt)",
    )
    parser.add_argument(
        "--no-whisper", action="store_true", help="Skip Whisper re-transcription step"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM correction (alignment + Whisper only)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=0.3,
        help="Delay between LLM API calls in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--intermediate",
        default=None,
        help="Save intermediate aligned JSON to this path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n┌─────────────────────────────────────────┐")
    print("│   Mandarin Subtitle Correction Pipeline  │")
    print("└─────────────────────────────────────────┘\n")

    # ── Step 1: Parse
    print("① Parsing SRT files...")
    noisy_segs = parse_srt(args.noisy)
    ref_segs = parse_srt(args.reference)
    print(f"  Noisy transcription:  {len(noisy_segs)} segments")
    print(f"  Reference subtitles:  {len(ref_segs)} segments")

    # ── Step 2: Align
    print("\n② Aligning segments by timestamp...")
    aligned = align_segments(noisy_segs, ref_segs)

    # ── Step 3: Whisper
    if not args.no_whisper:
        if args.video:
            print("\n③ Re-transcribing with Whisper...")
            aligned = whisper_transcribe(aligned, args.video)
        else:
            print("\n③ Skipping Whisper (no --video provided)")
            for seg in aligned:
                seg.whisper_zh = seg.noisy_zh
    else:
        print("\n③ Skipping Whisper (--no-whisper)")
        for seg in aligned:
            seg.whisper_zh = seg.noisy_zh

    # Save intermediate if requested
    if args.intermediate:
        with open(args.intermediate, "w", encoding="utf-8") as f:
            json.dump([vars(s) for s in aligned], f, ensure_ascii=False, indent=2)
        print(f"  Intermediate data saved to: {args.intermediate}")

    # ── Step 4: LLM Correction
    if not args.no_llm:
        print("\n④ Running LLM correction pass (Claude)...")
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not os.environ.get("ANTHROPIC_API_KEY"):
            print("  ⚠  No API key found. Set ANTHROPIC_API_KEY or use --api-key.")
            print("     Skipping LLM step.")
        else:
            aligned = batch_correct_llm(
                aligned, api_key=api_key, rate_limit_delay=args.batch_delay
            )
    else:
        print("\n④ Skipping LLM correction (--no-llm)")
        for seg in aligned:
            seg.corrected_zh = seg.whisper_zh or seg.noisy_zh

    # ── Step 5: Quality checks
    print("\n⑤ Running quality checks...")
    aligned = quality_check(aligned)

    # ── Step 6: Write output
    print(f"\n⑥ Writing output...")
    flagged_path = (
        args.flagged or str(Path(args.output).with_suffix("")) + ".flagged.txt"
    )
    write_srt(aligned, args.output, flagged_path)

    # ── Summary
    total = len(aligned)
    flagged = sum(1 for s in aligned if s.flagged)
    avg_conf = sum(s.confidence for s in aligned) / total if total else 0
    no_ref = sum(1 for s in aligned if not s.ref_zh)

    print(f"\n┌─ Summary ──────────────────────────────────")
    print(f"│  Segments processed:  {total}")
    print(f"│  Avg confidence:      {avg_conf:.2%}")
    print(f"│  Flagged for review:  {flagged} ({flagged / total:.0%})")
    print(f"│  No reference match:  {no_ref}")
    print(f"│  Output:              {args.output}")
    print(f"└────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
