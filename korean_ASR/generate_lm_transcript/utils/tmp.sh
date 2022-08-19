
 2640  awk '{print "KoreanSpeech_" NR " :: " $s}' korean_speech.txt > tmp_KoreanSpeech.txt
 2641  awk '{print "KoreanSpeech_" NR " :: " $s}' korean_free_speech.txt > tmp_KoreanFreeSpeech.txt

./utils/data.sh --KoreanSpeech ~/ASR/prepare_lm/tmp_KoreanSpeech.txt --KoreanFreeSpeech ~/ASR/prepare_lm/tmp_KoreanFreeSpeech.txt