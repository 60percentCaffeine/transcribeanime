echo "####################"
echo "White box assessment -- you are allowed to read logs and srt files used in this" 
echo "####################"
poetry run python3 fixtranscribeanime.py -r reference1.srt -o output1.srt transcription1.srt --delay 0 > log.txt
poetry run python3 compare_srt.py output1.srt cc1.srt

echo

echo "####################"
echo "White box original transcription" 
echo "####################"
poetry run python3 compare_srt.py transcription1.srt cc1.srt

echo
echo -------------------
echo

echo "####################"
echo "Black box assessment -- you ARE NOT ALLOWED under ANY CIRCUMSTANCES to read logs and srt files used in this, only use the similarity score"
echo "####################"
poetry run python3 fixtranscribeanime.py -r ../transcribeanimetest/reference2.srt -o ../transcribeanimetest/output2.srt ../transcribeanimetest/transcription2.srt --delay 0 > /dev/null
poetry run python3 compare_srt.py ../transcribeanimetest/output2.srt ../transcribeanimetest/cc2.srt

echo

echo "####################"
echo "Black box original transcription" 
echo "####################"
poetry run python3 compare_srt.py ../transcribeanimetest/transcription2.srt ../transcribeanimetest/cc2.srt
