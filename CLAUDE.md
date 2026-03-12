CLAUDE.md:

This project aims to construct professional Closed Captions for Mandarin Chinese video from ASR transcription and reference data (translated subtitles: they have the same meaning but don't match one for one what is being said).

fixtranscribeanime.py is the main script used to construct captions.

test.sh contains two test: white box and black box. Changing the contents of the test is NOT ALLOWED under ANY circumstances. It is allowed to analyse logs and srt-files used in the white box test. You are NOT ALLOWED under ANY circumstances to look at or change logs and srt-files used by the black box test.
After implementing an approach, ALWAYS run test.sh with no exceptions. The goal is to have 80% accuracy, as measured by the black box test.

APPROACHES.md contains a list of approaches to the problem. When trying an approach, it should ALWAYS leave an entry in this file. The entry should contain: short description of the approach, example from white box test the approach is supposed to fix, have we tried it in this project yet (YES/TOBETESTED), white box and black box evaluation numbers, whether the approach was successful (YES/NO), and additional comments on the results. In addition to this table it can optionally contain other sections describing the approaches or additional info.
