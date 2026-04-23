"""Snap cue start times to the acoustic speech onset.

CTC forced alignment (wav2vec2) hands out frames to the nearest token even
when those frames are silent — so cue starts can land up to a second or
more before the actual voice. This pass re-examines the audio window
around each cue start and moves the start forward to the first frame
whose short-term energy exceeds a relative threshold.

Exposes `snap_onsets(cues, audio_path, ...)`. Non-destructive: only
advances starts later (never earlier), and never past the cue's own end.
"""
from __future__ import annotations

from pathlib import Path


def _audio(audio_path: Path):
    from whisperx.audio import load_audio, SAMPLE_RATE
    import numpy as np
    arr = load_audio(str(audio_path)).astype(np.float32)
    return arr, SAMPLE_RATE


def _hop_rms(samples, sr, hop_ms: int):
    import numpy as np
    hop = int(sr * hop_ms / 1000)
    n_hops = len(samples) // hop
    if n_hops == 0:
        return np.array([0.0], dtype=np.float32), hop
    trimmed = samples[: n_hops * hop].reshape(n_hops, hop)
    rms = np.sqrt((trimmed ** 2).mean(axis=1) + 1e-10)
    return rms, hop


def _global_rms(samples, sr, hop_ms: int):
    """Compute hop-level RMS for the full audio; used as a global reference."""
    return _hop_rms(samples, sr, hop_ms)


def snap_cumulative_char_dur(
    cues: list[dict],
    *,
    threshold_s: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """Advance each cue's start past the first ``threshold_s`` seconds of
    wav2vec-aligned character content.

    For each cue, walk its `chars` list summing each char's aligned
    duration. Once the running sum reaches ``threshold_s``, set the cue
    start to that char's end time. Cues without retained char timings
    (``chars`` is None / empty) pass through unchanged.

    Rationale: wav2vec CTC tends to park early cue frames on the leading
    char during pre-speech silence; trimming the first second of aligned
    content pushes the displayed start past that pre-onset lead.
    """
    deltas: list[float] = []
    for i, cue in enumerate(cues):
        chars = cue.get("chars") or []
        if not chars:
            continue
        start = cue["start"]
        end = cue["end"]
        acc = 0.0
        new_start = None
        for ch in chars:
            s = ch.get("start")
            e = ch.get("end")
            if s is None or e is None:
                continue
            acc += max(0.0, e - s)
            if acc >= threshold_s:
                new_start = e
                break
        if new_start is None or new_start <= start:
            continue
        if i + 1 < len(cues):
            new_start = min(new_start, cues[i + 1]["start"])
        new_start = min(new_start, end - 0.1)
        if new_start <= start:
            continue
        deltas.append(new_start - start)
        cue["start"] = float(new_start)

    if verbose and deltas:
        d = sorted(deltas)
        n = len(d)
        print(f"[snap-cumdur] moved {n}/{len(cues)} cues  "
              f"median=+{d[n//2]:.3f}s mean=+{sum(deltas)/n:.3f}s "
              f"max=+{d[-1]:.3f}s")
    return cues


def snap_to_peak(
    cues: list[dict],
    audio_path: Path,
    *,
    window_s: float = 4.0,
    hop_ms: int = 20,
    smooth_hops: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """Snap each cue start to the time of peak RMS within the first
    ``window_s`` of the cue. Moves start to the stressed-vowel-like burst
    where the speaker is loudest — per-cue variable shift.
    """
    import numpy as np
    samples, sr = _audio(audio_path)
    duration_s = len(samples) / sr
    rms_full, hop = _hop_rms(samples, sr, hop_ms)
    if smooth_hops > 1:
        k = np.ones(smooth_hops, dtype=np.float32) / smooth_hops
        rms_full = np.convolve(rms_full, k, mode="same")

    deltas: list[float] = []
    for i, cue in enumerate(cues):
        start = cue["start"]
        end = cue["end"]
        ceiling = min(end - 0.1, start + window_s, duration_s)
        if ceiling <= start:
            continue
        lo = int(start * sr / hop)
        hi = int(ceiling * sr / hop)
        if hi <= lo:
            continue
        block = rms_full[lo:hi]
        if not len(block):
            continue
        peak = int(np.argmax(block))
        new_start = start + peak * hop / sr
        if i + 1 < len(cues):
            new_start = min(new_start, cues[i + 1]["start"])
        new_start = min(new_start, end - 0.1)
        if new_start <= start:
            continue
        deltas.append(new_start - start)
        cue["start"] = float(new_start)

    if verbose and deltas:
        d = sorted(deltas)
        n = len(d)
        print(f"[snap-peak] moved {n}/{len(cues)} cues  "
              f"median=+{d[n//2]:.3f}s mean=+{sum(deltas)/n:.3f}s "
              f"max=+{d[-1]:.3f}s")
    return cues


def snap_proportional(
    cues: list[dict],
    *,
    fraction: float = 0.4,
    min_shift_s: float = 0.0,
    max_shift_s: float = 3.5,
    verbose: bool = True,
) -> list[dict]:
    """Shift each cue start forward by `fraction * cue_duration`, clamped
    to [min_shift_s, max_shift_s]. Longer cues shift more, shorter cues
    shift less — a per-cue variable delay that can settle a systematic
    lead offset without being a constant.
    """
    deltas: list[float] = []
    for i, cue in enumerate(cues):
        start = cue["start"]
        end = cue["end"]
        dur = end - start
        shift = max(min_shift_s, min(max_shift_s, fraction * dur))
        new_start = start + shift
        # Keep cue non-degenerate and ordered.
        new_start = min(new_start, end - 0.1)
        if i + 1 < len(cues):
            new_start = min(new_start, cues[i + 1]["start"])
        if new_start <= start:
            continue
        deltas.append(new_start - start)
        cue["start"] = float(new_start)

    if verbose and deltas:
        d = sorted(deltas)
        n = len(d)
        print(f"[snap-prop] moved {n}/{len(cues)} cues  "
              f"median=+{d[n//2]:.3f}s mean=+{sum(deltas)/n:.3f}s "
              f"min=+{d[0]:.3f}s max=+{d[-1]:.3f}s")
    return cues


_SILERO_CACHE: dict = {}


def _silero_speech_regions(samples, sr, *, threshold: float = 0.5,
                           min_silence_ms: int = 200,
                           min_speech_ms: int = 100):
    """Run silero-vad on the full waveform, return list of (start_s, end_s)."""
    import torch
    if "model" not in _SILERO_CACHE:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            trust_repo=True, verbose=False,
        )
        _SILERO_CACHE["model"] = model
        _SILERO_CACHE["utils"] = utils
    model = _SILERO_CACHE["model"]
    get_speech_timestamps = _SILERO_CACHE["utils"][0]
    wav = torch.from_numpy(samples).float()
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
    )
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def snap_onsets_vad(
    cues: list[dict],
    audio_path: Path,
    *,
    threshold: float = 0.5,
    min_silence_ms: int = 200,
    min_speech_ms: int = 100,
    max_advance_s: float = 4.0,
    lookback_s: float = 0.15,
    verbose: bool = True,
) -> list[dict]:
    """Silero-VAD variant: snap each cue start to the nearest speech-region
    that starts within [cue.start - lookback_s, cue.start + max_advance_s].

    `lookback_s` lets a cue that started slightly inside a speech region
    anchor to that region's true start (can move the cue EARLIER by a few
    frames); otherwise we only advance.
    """
    samples, sr = _audio(audio_path)
    regions = _silero_speech_regions(
        samples, sr,
        threshold=threshold,
        min_silence_ms=min_silence_ms,
        min_speech_ms=min_speech_ms,
    )
    if verbose:
        total_voice = sum(e - s for s, e in regions)
        print(f"[snap-vad] {len(regions)} speech regions, "
              f"{total_voice:.1f}s total voice")

    deltas: list[float] = []
    ri = 0
    for i, cue in enumerate(cues):
        start = cue["start"]
        end = cue["end"]
        # Advance ri to first region whose end > start - lookback_s.
        while ri < len(regions) and regions[ri][1] <= start - lookback_s:
            ri += 1
        if ri >= len(regions):
            continue
        reg_start, reg_end = regions[ri]
        # If region is entirely before our acceptable window, skip.
        if reg_end <= start - lookback_s:
            continue
        new_start = reg_start
        # Reject moves beyond max_advance_s forward.
        if new_start > start + max_advance_s:
            continue
        # Reject moves earlier than -lookback_s.
        if new_start < start - lookback_s:
            new_start = start - lookback_s
        # Cap so we don't cross into next cue or past end.
        if i + 1 < len(cues):
            new_start = min(new_start, cues[i + 1]["start"])
        new_start = min(new_start, end - 0.1)
        if abs(new_start - start) < 0.005:
            continue
        deltas.append(new_start - start)
        cue["start"] = float(new_start)

    if verbose and deltas:
        deltas_sorted = sorted(deltas)
        n = len(deltas_sorted)
        median = deltas_sorted[n // 2]
        mean = sum(deltas) / n
        print(f"[snap-vad] moved {n}/{len(cues)} cues  "
              f"median={median:+.3f}s  mean={mean:+.3f}s  "
              f"min={deltas_sorted[0]:+.3f}s max={deltas_sorted[-1]:+.3f}s")
    return cues


def snap_onsets(
    cues: list[dict],
    audio_path: Path,
    *,
    max_advance_s: float = 3.0,
    min_voice_frac: float = 1.0,
    voice_percentile: float = 90.0,
    sustain_hops: int = 6,
    hop_ms: int = 20,
    smooth_hops: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """Advance each cue's start to the first clearly-voiced frame.

    Uses a GLOBAL amplitude reference so quiet + loud cues are compared
    on the same scale. A frame counts as "voiced" when its smoothed RMS
    exceeds ``min_voice_frac`` times the global ``voice_percentile`` of
    hop RMS values, and the next ``sustain_hops`` frames also average
    above that threshold (protects against clicks).
    """
    import numpy as np

    samples, sr = _audio(audio_path)
    duration_s = len(samples) / sr

    full_rms, hop = _global_rms(samples, sr, hop_ms)
    if smooth_hops > 1 and len(full_rms) >= smooth_hops:
        kernel = np.ones(smooth_hops, dtype=np.float32) / smooth_hops
        full_rms_s = np.convolve(full_rms, kernel, mode="same")
    else:
        full_rms_s = full_rms
    ref = float(np.percentile(full_rms_s, voice_percentile))
    threshold = ref * min_voice_frac
    if verbose:
        print(f"[snap] global P{voice_percentile:.0f} RMS={ref:.5f}  "
              f"threshold={threshold:.5f}  hop={hop_ms}ms")

    deltas: list[float] = []
    for i, cue in enumerate(cues):
        start = cue["start"]
        end = cue["end"]
        ceiling = min(end - 0.1, start + max_advance_s, duration_s)
        if ceiling <= start:
            continue

        lo = int(start * sr / hop)
        hi = int(ceiling * sr / hop)
        if hi - lo < sustain_hops + 1:
            continue

        block = full_rms_s[lo:hi]
        found = None
        # Iterate with a sliding window: require sustained energy.
        for j in range(len(block) - sustain_hops):
            if block[j] >= threshold:
                # Average the next `sustain_hops` frames to confirm.
                if block[j : j + sustain_hops].mean() >= threshold:
                    found = j
                    break
        if found is None:
            continue
        new_start = start + found * hop / sr
        if new_start <= start:
            continue
        if i + 1 < len(cues):
            new_start = min(new_start, cues[i + 1]["start"])
        new_start = min(new_start, end - 0.1)
        if new_start <= start:
            continue
        deltas.append(new_start - start)
        cue["start"] = float(new_start)

    if verbose and deltas:
        deltas.sort()
        n = len(deltas)
        median = deltas[n // 2]
        mean = sum(deltas) / n
        print(f"[snap] advanced {n}/{len(cues)} cues  "
              f"median=+{median:.3f}s  mean=+{mean:.3f}s  "
              f"max=+{deltas[-1]:.3f}s")
    return cues
