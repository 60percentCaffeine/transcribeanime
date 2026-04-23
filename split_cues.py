"""Split overly long cues into 2–3 shorter ones using per-char word timings.

Input cues have {"start", "end", "text", "chars"} where `chars` is the
per-character timing list produced by realign_cues (aligned from wav2vec2).
Entries in `chars` may have start/end = None for characters the aligner
had no timing for (whitespace, rarely punctuation).

Output cues have {"start", "end", "text"} — no `chars`.

Splitting strategy:
  1. Find sentence-ending punctuation (。！？…) as primary split points.
  2. If a resulting chunk is still too long, fall back to clause
     punctuation (，、；:).
  3. Greedily pack chunks toward `target_chars`, obeying `min_chars` and
     `min_duration_s` floors (merging short fragments with a neighbour).
  4. Time each chunk by the earliest/latest word-timing inside it, with
     fallback interpolation if a chunk has no usable timings.

If alignment failed for a cue (chars is None) or the text is short, the
cue is returned unchanged.
"""
from __future__ import annotations

from typing import Iterable


_SENTENCE_END = frozenset("。！？!?…")
_CLAUSE_END = frozenset("，、；,;:：")


def _split_points(text: str, punct: Iterable[str]) -> list[int]:
    """Indices marking chunk-end positions (inclusive of the punctuation).

    Returns a list of cut positions `j` such that text[:j] ends with a
    punctuation char. The final value is always len(text) so chunks cover
    the full string when paired.
    """
    pset = set(punct)
    cuts: list[int] = []
    i = 0
    while i < len(text):
        if text[i] in pset:
            # Absorb a run of consecutive punctuation/whitespace so
            # "。」 " isn't split mid-cluster.
            j = i + 1
            while j < len(text) and (text[j] in pset or text[j].isspace()):
                j += 1
            cuts.append(j)
            i = j
        else:
            i += 1
    if not cuts or cuts[-1] != len(text):
        cuts.append(len(text))
    return cuts


def _chunk_timing(
    chars: list[dict], lo: int, hi: int,
    cue_start: float, cue_end: float,
) -> tuple[float, float]:
    """Pick (start, end) for text[lo:hi] from per-char timings.

    Falls back to interpolation (proportional to char index within the
    cue) if no chars in the slice have timings.
    """
    starts = [c["start"] for c in chars[lo:hi]
              if c.get("start") is not None]
    ends = [c["end"] for c in chars[lo:hi] if c.get("end") is not None]
    if starts and ends:
        return float(min(starts)), float(max(ends))
    # Fall back to proportional time across the cue. Should be rare.
    n = max(1, len(chars))
    span = cue_end - cue_start
    return cue_start + span * lo / n, cue_start + span * hi / n


def _build_chunks(
    text: str, chars: list[dict],
    cue_start: float, cue_end: float,
    *, target_chars: int, min_chars: int, max_chars: int,
    min_duration_s: float,
) -> list[tuple[int, int, float, float]]:
    """Pack text into chunks honouring length + duration floors.

    Returns [(lo, hi, start, end), ...]. Prefers sentence-end cuts;
    falls back to clause-end cuts for any chunk still over max_chars;
    merges chunks shorter than min_chars (or min_duration_s) with a
    neighbour.
    """
    # Step 1: cut at sentence boundaries.
    sentence_cuts = _split_points(text, _SENTENCE_END)

    # Step 2: for any piece still over max_chars, sub-cut at clause punct.
    all_cuts: list[int] = []
    prev = 0
    for cut in sentence_cuts:
        piece = text[prev:cut]
        if len(piece) > max_chars:
            sub_cuts = _split_points(piece, _CLAUSE_END)
            for sc in sub_cuts[:-1]:
                all_cuts.append(prev + sc)
        all_cuts.append(cut)
        prev = cut

    # Step 3: greedy pack — accumulate slices until >= target_chars or a
    # sentence-end is close; then emit.
    atoms: list[tuple[int, int]] = []
    prev = 0
    for cut in all_cuts:
        if cut > prev:
            atoms.append((prev, cut))
            prev = cut

    packed: list[tuple[int, int]] = []
    cur_lo: int | None = None
    cur_hi = 0
    for lo, hi in atoms:
        if cur_lo is None:
            cur_lo, cur_hi = lo, hi
            continue
        # If current is already near target, emit and start fresh.
        if (cur_hi - cur_lo) >= target_chars:
            packed.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
        # If adding this atom would push us over max_chars, emit first.
        elif (hi - cur_lo) > max_chars and (cur_hi - cur_lo) >= min_chars:
            packed.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = hi
    if cur_lo is not None:
        packed.append((cur_lo, cur_hi))

    # Step 4: merge runts into neighbours (short char count or duration).
    timed: list[tuple[int, int, float, float]] = []
    for lo, hi in packed:
        s, e = _chunk_timing(chars, lo, hi, cue_start, cue_end)
        timed.append((lo, hi, s, e))

    changed = True
    while changed and len(timed) > 1:
        changed = False
        for i, (lo, hi, s, e) in enumerate(timed):
            is_runt = (hi - lo) < min_chars or (e - s) < min_duration_s
            if not is_runt:
                continue
            # Merge into left neighbour if it exists, else right.
            if i > 0:
                pl, _, ps, _ = timed[i - 1]
                timed[i - 1] = (pl, hi, ps, e)
                timed.pop(i)
            else:
                _, nh, _, ne = timed[i + 1]
                timed[i + 1] = (lo, nh, s, ne)
                timed.pop(i)
            changed = True
            break

    return timed


def split_cues(
    cues: list[dict],
    *,
    target_chars: int = 22,
    min_chars: int = 8,
    max_chars: int = 32,
    min_duration_s: float = 1.0,
    split_over_chars: int = 32,
    split_over_duration_s: float = 6.0,
    verbose: bool = True,
) -> list[dict]:
    """Split long cues into 2–3 shorter ones using per-char word timings.

    A cue is a candidate for splitting if it exceeds `split_over_chars`
    OR `split_over_duration_s`. Short cues pass through untouched.
    """
    out: list[dict] = []
    n_split = 0
    n_produced = 0
    for cue in cues:
        text = cue["text"]
        chars = cue.get("chars")
        start = cue["start"]
        end = cue["end"]
        duration = end - start

        too_long = (
            len(text) > split_over_chars
            or duration > split_over_duration_s
        )
        if not too_long or not chars or not isinstance(chars, list):
            out.append({"start": start, "end": end, "text": text,
                        "split": False, "chars": chars})
            continue

        pieces = _build_chunks(
            text, chars, start, end,
            target_chars=target_chars, min_chars=min_chars,
            max_chars=max_chars, min_duration_s=min_duration_s,
        )
        if len(pieces) <= 1:
            out.append({"start": start, "end": end, "text": text,
                        "split": False, "chars": chars})
            continue

        # First piece's start and last piece's end inherit the cue bounds
        # so we don't lose audio (alignment is often tight by a few frames).
        for i, (lo, hi, ps, pe) in enumerate(pieces):
            chunk_text = text[lo:hi].strip()
            if not chunk_text:
                continue
            if i == 0:
                ps = min(ps, start)
            if i == len(pieces) - 1:
                pe = max(pe, end)
            # Enforce non-overlap between consecutive pieces.
            if out and out[-1].get("_cue_group") == id(cue):
                prev = out[-1]
                if ps < prev["end"]:
                    ps = prev["end"]
                if pe < ps:
                    pe = ps + min_duration_s
            entry = {"start": ps, "end": pe, "text": chunk_text,
                     "split": True, "chars": chars[lo:hi],
                     "_cue_group": id(cue)}
            out.append(entry)
            n_produced += 1
        n_split += 1

    # Strip bookkeeping key.
    for entry in out:
        entry.pop("_cue_group", None)

    if verbose:
        print(f"[split] {n_split} cues split into {n_produced} new cues "
              f"(total out: {len(out)})")
    return out
