# Approaches

## Summary Table

| # | Approach | Example from white box test | Tried? | White box | Black box | Successful? | Comments |
|---|----------|-----------------------------|--------|-----------|-----------|-------------|----------|
| 1 | Adding punctuation | cc1 has `等上了高中後，我一定要玩樂團` with comma; transcription `登上了高中後我一定要外月退` has none. Also `"來自壁櫥，我的愛"` has quotes and comma in ground truth. | TOBETESTED | — | — | — | |
| 2 | Rewriting whole line using LLM with reference+transcription context | Transcription: `英雄的社交障礙權` → ground truth: `陰沉的社交障礙者`. Current system can only fix homophones (權→圈), but the correct answer `陰沉` and `者` are not homophones of `英雄` and `權`. Similarly `外月退` should be `玩樂團` — not a homophone swap. | TOBETESTED | — | — | — | |
| 3 | Building a homophone dictionary from reference subtitles and replacing words in transcription | `文化機` appears in seg 5 and `文化界` in seg 34; both should be `文化祭` (from reference). If we learn 機→祭 from the reference alignment in one segment, we can apply it to fix `界→祭` in the other. Also `反碎→粉碎` appears in both seg 46 and 47. | TOBETESTED | — | — | — | |
