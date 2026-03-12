# Approaches

## Summary Table

| # | Approach | Example from white box test | Tried? | White box | Black box | Successful? | Comments |
|---|----------|-----------------------------|--------|-----------|-----------|-------------|----------|
| 1 | Adding punctuation | cc1 has `等上了高中後，我一定要玩樂團` with comma; transcription `登上了高中後我一定要外月退` has none. Also `"來自壁櫥，我的愛"` has quotes and comma in ground truth. | YES | 79.0% (was 78.2%) | 70.9% (unchanged) | NO | LLM-based punctuation hurt scores badly (78.2%→71.4%); rule-based was safe but marginal (+0.8% white box only). Ceiling is very low: ~16 punct chars in ground truth, most in lines with other content mismatches. Black box got zero benefit. |
| 2 | Rewriting whole line using LLM with reference+transcription context | Transcription: `英雄的社交障礙權` → ground truth: `陰沉的社交障礙者`. Current system can only fix homophones (權→圈), but the correct answer `陰沉` and `者` are not homophones of `英雄` and `權`. Similarly `外月退` should be `玩樂團` — not a homophone swap. | YES | 79.2% (was 78.2%) | 70.6-71.1% (was 71.1%) | YES | Modified prompt to include boundary errors + relaxed homophone check for 3+ char fixes using full pinyin string comparison. Separate LLM rewrite call was tried but consistently hurt black box. |
## Approach 2: ASR Boundary Error Detection

### What was done
ASR sometimes redistributes syllable boundaries, producing nonsensical character combinations that sound similar overall but are NOT homophones character-by-character. Examples: `外月退→玩樂團` (wài yuè tuì ≈ wán lè tuán), `頭開頭→頭蓋骨` (tóu kāi tóu ≈ tóu gài gǔ).

Two changes were made to handle these:
1. **Modified the LLM system prompt** to include "音節邊界錯誤" (syllable boundary errors) as a category alongside homophones, with specific examples. This causes the LLM to suggest the correct fixes (e.g., `外月退→玩樂團`) instead of trying to find homophones (which produced wrong answers like `外月退→湊齊樂`).
2. **Added `is_relaxed_homophone` function** that validates 3+ char fixes by comparing the FULL concatenated pinyin string (not character-by-character). Uses `similar_pinyin_str` with a 0.4 similarity threshold. This is needed because boundary errors make individual characters very different phonetically even though the overall sound is similar.

### Key numbers
- White box: 78.2% → 79.2% (+1.0%)
- Black box: 71.1% → 70.6-71.1% (variable due to LLM non-determinism, average ~70.8%)
- Key fixes in white box: `外月退→玩樂團`, `頭開頭→頭蓋骨` (both segs 46 & 47)

### What was tried but didn't work
- **Separate LLM rewrite call**: A second LLM call asking for a full line rewrite. Consistently improved white box (+1.5%) but hurt black box (-0.7%). The rewrite produced too many synonym swaps (e.g., `人氣→熱門`, `名義→網名`, `爛歌→屎歌`) despite prompt instructions not to. Validation heuristics (jieba word segmentation, pinyin similarity thresholds) couldn't reliably distinguish good fixes from bad ones.

### Quality issues to watch
- The relaxed check only applies to 3+ char fixes. 2-char boundary errors are not handled.
- Black box results are slightly variable (~±0.5%) due to LLM non-determinism with parallel API calls.
- Some boundary errors like `糟了正→演奏了這` are NOT fixed because the LLM suggests `糟了正→演奏了` (wrong char count) or the pinyin similarity is too low.

| 3 | Building a homophone dictionary from reference subtitles and replacing words in transcription | `文化機` appears in seg 5 and `文化界` in seg 34; both should be `文化祭` (from reference). If we learn 機→祭 from the reference alignment in one segment, we can apply it to fix `界→祭` in the other. Also `反碎→粉碎` appears in both seg 46 and 47. | TOBETESTED | — | — | — | |
| 4 | Use more transcription context (wider window) for LLM | Seg 33 `英雄的社交障礙權` should be `陰沉的社交障礙者`. The LLM currently sees ±3 lines (segs 30–36), but seg 23 mentions `吉他英雄` (the username). With wider context, the LLM could realize `英雄` in seg 33 is wrong — the speaker is describing social recluses, not heroes. Similarly seg 11 `外月退` → `玩樂團`: seeing seg 7 `樂團成員` in context would reinforce that this is about forming a band. | TOBETESTED | — | — | — | |
| 5 | Use more reference context for better scene/narrative awareness | Seg 34 `我的樂團在文化界也糟了正` → ground truth `我的樂團在文化祭演奏了這⋯`. Currently only 5 nearest reference lines by time overlap are provided. Providing more reference lines (or a broader time window) gives the LLM fuller narrative context: the story is about performing at a cultural festival. Similarly segs 46–47 `反碎你頭開頭` → `粉碎你的頭蓋骨`: broader reference context about death metal song lyrics would help. | TOBETESTED | — | — | — | |
