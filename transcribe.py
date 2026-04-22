#!/usr/bin/env python3
"""Transcribe Mandarin anime media to SRT using the whisperjav qwen +
whisperx wav2vec2 realignment pipeline.

This is the same pipeline that produced
`whisperjav/results_belle_wx/qwen_wxalign_geass.v2.srt` — whisperjav in qwen
mode with `silero-v3.1` segmentation, followed by a per-cue wav2vec2 forced
realignment (ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt) with
t2s/digits/strip_whitespace normalization, 1 s pad on each side, and a 1.5 s
minimum-cue-duration floor.

Usage:
    poetry run python3 transcribe.py geass1.mkv geass2.mkv
    -> writes geass1.srt and geass2.srt next to the inputs.

Assumes `whisperjav` is on PATH and `whisperx` is importable — i.e. the
poetry virtualenv has both installed.
"""
from __future__ import annotations

import argparse
import gc
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import torch
from whisperx.alignment import align, load_align_model
from whisperx.audio import SAMPLE_RATE, load_audio

from normalize_entities import normalize_cues as normalize_entity_cues

# --- Pipeline settings (reproduces qwen_wxalign_geass.v2.srt) ---------------
ALIGN_LANGUAGE = "zh"
ALIGN_MODEL = "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
NORMALIZE_RULES = ("t2s", "digits", "strip_whitespace")
INTERPOLATE_METHOD = "nearest"
ALIGN_PAD_START_MS = 1000
ALIGN_PAD_END_MS = 1000
NEIGHBOR_GAP_MS = 50
MIN_CUE_DURATION_MS = 1500

WHISPERJAV_FLAGS = (
    "--mode", "qwen",
    "--language", "chinese",
    "--qwen-language", "zh",
    "--qwen-segmenter", "silero-v3.1",
    "--no-signature",
    "--verbosity", "summary",
)

# Preference order when whisperjav produces multiple SRTs under out_dir.
SRT_PREFERENCE = (".merged.srt", ".pass1.srt", ".whisperjav.srt", ".srt")


# --- whisperjav invocation --------------------------------------------------
def run_whisperjav(input_path: Path, out_dir: Path) -> None:
    cmd = ["whisperjav", str(input_path), "--output-dir", str(out_dir),
           *WHISPERJAV_FLAGS]
    print(f"[whisperjav] {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[whisperjav] done in {time.time()-t0:.1f}s")


def pick_srt(out_dir: Path) -> Path:
    """Find whisperjav's final SRT, preferring merged > pass1 > plain."""
    srts = list(out_dir.rglob("*.srt"))
    if not srts:
        raise RuntimeError(f"No SRT produced by whisperjav in {out_dir}")
    for suffix in SRT_PREFERENCE:
        matches = [p for p in srts
                   if p.name.endswith(suffix) and "raw_subs" not in p.parts]
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]
    srts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return srts[0]


# --- SRT I/O ----------------------------------------------------------------
_SRT_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})"
)


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        if not block.strip():
            continue
        lines = block.split("\n")
        t_idx = None
        for i, line in enumerate(lines):
            if _SRT_TIME.search(line):
                t_idx = i
                break
        if t_idx is None:
            continue
        tm = _SRT_TIME.search(lines[t_idx])
        start = _ts_to_seconds(tm.group(1), tm.group(2), tm.group(3), tm.group(4))
        end = _ts_to_seconds(tm.group(5), tm.group(6), tm.group(7), tm.group(8))
        body = "\n".join(lines[t_idx + 1:]).strip()
        body = re.sub(r"\s+", " ", body).strip()
        cues.append({"start": start, "end": end, "text": body})
    return cues


def fmt_srt_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[dict], path: Path) -> None:
    lines = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_ts(cue['start'])} --> {fmt_srt_ts(cue['end'])}")
        lines.append(cue["text"])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# --- Text normalization for the aligner -------------------------------------
_DIGIT_TO_HANZI = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}

_OPENCC_T2S = None


def _get_t2s():
    global _OPENCC_T2S
    if _OPENCC_T2S is None:
        from opencc import OpenCC
        _OPENCC_T2S = OpenCC("t2s")
    return _OPENCC_T2S


def _digits_to_hanzi(text: str) -> str:
    """Read ASCII digits as Chinese-numeral chars (2010 → 二零一零).

    Digit-by-digit (not "两千零十") matches how speakers read years/IDs and
    turns unalignable digit chars into hanzi the CTC model knows.
    """
    return "".join(_DIGIT_TO_HANZI.get(ch, ch) for ch in text)


def normalize_for_align(text: str, rules) -> str:
    out = text
    for rule in rules:
        if rule == "t2s":
            out = _get_t2s().convert(out)
        elif rule == "digits":
            out = _digits_to_hanzi(out)
        elif rule == "strip_whitespace":
            out = re.sub(r"\s+", "", out)
        elif rule in ("fullwidth_fold", "nfkc"):
            out = unicodedata.normalize("NFKC", out)
        else:
            raise ValueError(f"Unknown normalize rule: {rule!r}")
    return out


# --- wav2vec2 realignment ---------------------------------------------------
def realign_cues(cues: list[dict], audio_path: Path, device: str) -> list[dict]:
    """Run whisperx align() per cue, replacing start/end with word min/max.

    Cues that fail alignment keep their original whisperjav timestamps — same
    fallback policy as whisperx itself.
    """
    print(f"[align] loading wav2vec2 '{ALIGN_MODEL}' on {device} ...")
    t0 = time.time()
    align_model, align_metadata = load_align_model(
        ALIGN_LANGUAGE, device, model_name=ALIGN_MODEL
    )
    print(f"[align] model loaded in {time.time()-t0:.1f}s "
          f"(dict={len(align_metadata['dictionary'])})")

    print(f"[align] loading audio {audio_path} ...")
    t0 = time.time()
    audio = load_audio(str(audio_path))
    audio_duration_s = audio.shape[-1] / SAMPLE_RATE
    print(f"[align] audio loaded: {audio_duration_s:.1f}s mono @ "
          f"{SAMPLE_RATE} Hz ({time.time()-t0:.1f}s)")

    out_cues: list[dict] = []
    n_aligned = n_fallback = n_empty = n_clipped = 0

    t0 = time.time()
    for idx, cue in enumerate(cues):
        text = cue["text"].strip()
        start = cue["start"]
        end = cue["end"]

        if not text:
            n_empty += 1
            out_cues.append({"start": start, "end": end, "text": text})
            continue

        seg_end = min(end, audio_duration_s)
        if seg_end <= start:
            n_clipped += 1
            out_cues.append({"start": start, "end": end, "text": text})
            continue

        # Expand the audio window by pad_*_ms, but clamp so we never reach
        # into a neighbour cue's audio (preserving NEIGHBOR_GAP_MS).
        prev_end = cues[idx - 1]["end"] if idx > 0 else 0.0
        next_start = (cues[idx + 1]["start"] if idx + 1 < len(cues)
                      else audio_duration_s)
        gap_s = NEIGHBOR_GAP_MS / 1000.0
        max_left = max(0.0, (start - prev_end) - gap_s)
        max_right = max(0.0, (next_start - end) - gap_s)
        left_pad = min(ALIGN_PAD_START_MS / 1000.0, max_left)
        right_pad = min(ALIGN_PAD_END_MS / 1000.0, max_right)
        padded_start = max(0.0, start - left_pad)
        padded_end = min(audio_duration_s, seg_end + right_pad)

        align_text = normalize_for_align(text, NORMALIZE_RULES)
        segment = {"start": padded_start, "end": padded_end, "text": align_text}

        try:
            result = align(
                [segment], align_model, align_metadata, audio, device,
                interpolate_method=INTERPOLATE_METHOD,
                return_char_alignments=False,
                print_progress=False,
            )
        except Exception as exc:
            print(f"[align] cue {idx+1} raised {type(exc).__name__}: {exc} "
                  f"— keeping original times")
            n_fallback += 1
            out_cues.append({"start": start, "end": end, "text": text})
            continue

        word_starts: list[float] = []
        word_ends: list[float] = []
        for sub in result["segments"]:
            for w in sub.get("words", []):
                if "start" in w:
                    word_starts.append(w["start"])
                if "end" in w:
                    word_ends.append(w["end"])

        if word_starts and word_ends:
            new_start = float(min(word_starts))
            new_end = float(max(word_ends))
            n_aligned += 1

            # If alignment squashed the cue below the floor, extend the END
            # back toward the original whisperjav end (never past the next
            # cue's start − NEIGHBOR_GAP_MS). Start is not moved.
            if MIN_CUE_DURATION_MS:
                floor_s = MIN_CUE_DURATION_MS / 1000.0
                if (new_end - new_start) < floor_s:
                    wanted_end = new_start + floor_s
                    ceiling = min(end, next_start - gap_s)
                    wanted_end = min(wanted_end, ceiling)
                    if wanted_end > new_end:
                        new_end = wanted_end

            if new_end < new_start:
                new_start, new_end = start, end
            out_cues.append({"start": new_start, "end": new_end, "text": text})
        else:
            n_fallback += 1
            out_cues.append({"start": start, "end": end, "text": text})

        if (idx + 1) % 50 == 0 or (idx + 1) == len(cues):
            print(f"[align] {idx+1}/{len(cues)} cues "
                  f"(aligned={n_aligned} fallback={n_fallback})")

    print(f"[align] finished {len(cues)} cues in {time.time()-t0:.1f}s")
    print(f"[stats] in={len(cues)} aligned={n_aligned} fallback={n_fallback} "
          f"empty_text={n_empty} clipped={n_clipped}")

    del align_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return out_cues


# --- end-to-end per-file orchestration --------------------------------------
def transcribe_one(
    input_path: Path,
    output_path: Path,
    device: str,
    *,
    llm_entities: bool = False,
    llm_api_key: str | None = None,
    llm_runs: int = 3,
) -> None:
    work_root = Path(tempfile.mkdtemp(prefix="transcribeanime_"))
    wjav_out = work_root / "wjav"
    wjav_out.mkdir()
    try:
        run_whisperjav(input_path, wjav_out)
        raw_srt = pick_srt(wjav_out)
        print(f"[pick] whisperjav SRT: {raw_srt} ({raw_srt.stat().st_size} bytes)")

        cues = parse_srt(raw_srt)
        print(f"[parse] {len(cues)} cues from {raw_srt.name}")
        if not cues:
            raise SystemExit(f"No cues parsed from whisperjav SRT for {input_path}")

        realigned = realign_cues(cues, input_path, device)
        realigned.sort(key=lambda c: (c["start"], c["end"]))
        realigned, _ = normalize_entity_cues(
            realigned,
            use_llm=llm_entities,
            llm_api_key=llm_api_key,
            llm_runs=llm_runs,
        )
        write_srt(realigned, output_path)
        print(f"[out] {output_path} ({output_path.stat().st_size} bytes, "
              f"{len(realigned)} cues)")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def run_fixtranscribeanime(srt_path: Path, reference: Path, chinese: str) -> None:
    """Correct srt_path in-place against reference using fixtranscribeanime."""
    cmd = [
        "fixtranscribeanime",
        str(srt_path),
        "--reference", str(reference),
        "--output", str(srt_path),
        "--chinese", chinese,
    ]
    print(f"[fix] {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[fix] done in {time.time()-t0:.1f}s")


def resolve_reference(
    src: Path,
    *,
    explicit: Path | None,
    reference_dir: Path | None,
) -> Path | None:
    """Resolve the reference SRT for `src`. Returns the path if found, else None.

    Precedence: --reference > <reference-dir>/<stem>.reference.srt >
                <src.parent>/<stem>.reference.srt.
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidate_dir = reference_dir if reference_dir is not None else src.parent
    candidate = candidate_dir / f"{src.stem}.reference.srt"
    return candidate if candidate.exists() else None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Transcribe Mandarin media to SRT "
                    "(whisperjav qwen + wav2vec2 realign + fixtranscribeanime).",
    )
    p.add_argument("inputs", nargs="+", type=Path,
                   help="Input media files. Each produces <stem>.srt.")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="Directory for SRTs. Default: next to each input.")
    p.add_argument("-r", "--reference", type=Path, default=None,
                   help="Reference SRT filename. Only valid with a single input.")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="Directory to search for <stem>.reference.srt. "
                        "Default: each input's own directory.")
    p.add_argument("--nofix", action="store_true",
                   help="Skip fixtranscribeanime post-processing "
                        "(and skip the startup reference check).")
    p.add_argument("--chinese", choices=["s", "t"], default="s",
                   help="Chinese variant passed to fixtranscribeanime "
                        "(default: s).")
    p.add_argument("--device", default=None,
                   help="Device for the aligner. Default: cuda if available.")
    p.add_argument("--llm-entities", action="store_true",
                   help="Also use an LLM (OpenRouter) to detect named "
                        "entities in the realigned SRT. Unioned with the "
                        "pinyin/jieba heuristic. Requires OPENROUTER_API_KEY.")
    p.add_argument("--llm-runs", type=int, default=3,
                   help="LLM consensus runs (default 3). Each run has a "
                        "small temperature jitter; results are unioned.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    missing = [f for f in args.inputs if not f.exists()]
    if missing:
        print("Missing inputs:", ", ".join(str(m) for m in missing),
              file=sys.stderr)
        return 2

    if args.reference is not None and len(args.inputs) > 1:
        print("--reference only accepts one input file; use --reference-dir "
              "or default lookup for multiple inputs.", file=sys.stderr)
        return 2

    # Resolve references up front so we can bail before any transcription work.
    references: dict[Path, Path] = {}
    if not args.nofix:
        unresolved = []
        for src in args.inputs:
            ref = resolve_reference(
                src,
                explicit=args.reference,
                reference_dir=args.reference_dir,
            )
            if ref is None:
                unresolved.append(src)
            else:
                references[src] = ref
        if unresolved:
            print("Reference SRT not found for:", file=sys.stderr)
            for src in unresolved:
                expected = (
                    args.reference if args.reference is not None
                    else (args.reference_dir or src.parent)
                        / f"{src.stem}.reference.srt"
                )
                print(f"  {src} -> expected {expected}", file=sys.stderr)
            print("Provide --reference / --reference-dir, or pass --nofix "
                  "to skip correction.", file=sys.stderr)
            return 2

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    llm_api_key: str | None = None
    if args.llm_entities:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        import os as _os
        llm_api_key = _os.environ.get("OPENROUTER_API_KEY")
        if not llm_api_key:
            print("--llm-entities requires OPENROUTER_API_KEY (set in .env or the environment).",
                  file=sys.stderr)
            return 2

    for src in args.inputs:
        out_dir = args.output_dir if args.output_dir is not None else src.parent
        dst = (out_dir / f"{src.stem}.srt").resolve()
        transcribe_one(
            src.resolve(), dst, device,
            llm_entities=args.llm_entities,
            llm_api_key=llm_api_key,
            llm_runs=args.llm_runs,
        )
        if not args.nofix:
            run_fixtranscribeanime(dst, references[src].resolve(), args.chinese)

    return 0


if __name__ == "__main__":
    sys.exit(main())
