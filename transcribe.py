#!/usr/bin/env python3
"""Generate SRT subtitles for Mandarin Chinese videos using Whisper on Apple Silicon (MLX)."""

import argparse
import sys
import time
from pathlib import Path

import mlx_whisper

MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-turbo",
}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_to_srt(video_path: str, model_name: str = "medium", output_path: str | None = None) -> str:
    """Transcribe a video file and generate an SRT subtitle file.

    Args:
        video_path: Path to the input video/audio file.
        model_name: Whisper model size (tiny, base, small, medium, large, turbo).
        output_path: Path for the output SRT file. Defaults to video_path with .srt extension.

    Returns:
        Path to the generated SRT file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = video_path.with_suffix(".srt")
    else:
        output_path = Path(output_path)

    repo = MODEL_MAP[model_name]
    print(f"Using MLX backend (Apple Silicon GPU)")
    print(f"Loading model '{repo}'...")

    start_time = time.monotonic()
    result = mlx_whisper.transcribe(
        str(video_path),
        path_or_hf_repo=repo,
        language="zh",
        verbose=False,
    )
    elapsed = time.monotonic() - start_time

    segments = result["segments"]
    print(f"Found {len(segments)} segments in {elapsed:.1f}s.")

    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved SRT to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate SRT subtitles for Mandarin Chinese videos using Whisper.")
    parser.add_argument("video", help="Path to the video or audio file")
    parser.add_argument("-m", "--model", default="medium", choices=list(MODEL_MAP.keys()),
                        help="Whisper model size (default: medium)")
    parser.add_argument("-o", "--output", default=None, help="Output SRT file path (default: same as input with .srt extension)")
    args = parser.parse_args()

    transcribe_to_srt(args.video, model_name=args.model, output_path=args.output)


if __name__ == "__main__":
    main()
