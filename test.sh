echo "####################"
echo "White box assessment -- you are allowed to read logs and srt files used in this" 
echo "####################"
poetry run python3 fixtranscribeanime.py -r slimetest/sreference04.srt -o soutput04.srt slimetest/stranscription04.srt --delay 0 > log.txt
poetry run python3 compare_srt.py soutput04.srt slimetest/scc04.srt

echo

echo "####################"
echo "White box original transcription" 
echo "####################"
poetry run python3 compare_srt.py slimetest/stranscription04.srt slimetest/scc04.srt

echo
echo -------------------
echo

echo "####################"
echo "Black box assessment -- you ARE NOT ALLOWED under ANY CIRCUMSTANCES to read logs and srt files used in this, only use the similarity score"
echo "####################"
poetry run python3 fixtranscribeanime.py -r ../slimetest/sreference06.srt -o ../slimetest/soutput06.srt ../slimetest/stranscription06.srt --delay 0 > /dev/null
poetry run python3 compare_srt.py ../slimetest/soutput06.srt ../slimetest/scc06.srt

echo

echo "####################"
echo "Black box original transcription" 
echo "####################"
poetry run python3 compare_srt.py ../slimetest/stranscription06.srt ../slimetest/scc06.srt
