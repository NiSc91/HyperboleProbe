# Current parameters: To_sentence_spam=True (I don't know what it means), fraction for the dev set is 0.667 (so only 10 % of the full data is helt out for testing). Best results is F1-score of 0.75 and compression at 1.356 for layer 10.
# Set environment variables
MDL_CODE="source_code/scripts/mdl_probing.py"
export MDL_CODE
EDGE_CODE=source_code/scripts/edge_probing.py
export EDGE_CODE

#python $EDGE_CODE bert-base-uncased-random-weights lcc 0
#python $EDGE_CODE bert-base-uncased-random-weights trofi 0
#python $EDGE_CODE bert-base-uncased-random-weights vua_verb 0
#python $EDGE_CODE bert-base-uncased-random-weights vua_pos 0
#python $EDGE_CODE bert-base-uncased-random-weights hypo_en 0
python $EDGE_CODE bert-base-uncased hypo_en 0
python $EDGE_CODE roberta-base hypo_en 0

#python $EDGE_CODE bert-base-uncased-random-weights lcc 1
#python $EDGE_CODE bert-base-uncased-random-weights trofi 1
#python $EDGE_CODE bert-base-uncased-random-weights vua_verb 1
#python $EDGE_CODE bert-base-uncased-random-weights vua_pos 1
#python $EDGE_CODE bert-base-uncased-random-weights hypo_en 1

#python $EDGE_CODE bert-base-uncased-random-weights lcc 2
#python $EDGE_CODE bert-base-uncased-random-weights trofi 2
#python $EDGE_CODE bert-base-uncased-random-weights vua_verb 2
#python $EDGE_CODE bert-base-uncased-random-weights vua_pos 2
#python $EDGE_CODE bert-base-uncased-random-weights hypo_en 1

#python $MDL_CODE bert-base-uncased lcc 0
#python $MDL_CODE bert-base-uncased trofi 0
#python $MDL_CODE bert-base-uncased vua_verb 0
#python $MDL_CODE bert-base-uncased vua_pos 0
python $MDL_CODE bert-base-uncased hypo_en 0

#python $MDL_CODE roberta-base lcc 0
#python $MDL_CODE roberta-base trofi 0
#python $MDL_CODE roberta-base vua_pos 0
#python $MDL_CODE roberta-base vua_verb 0
python $MDL_CODE roberta-base hypo_en 0

#python $MDL_CODE google/electra-base-discriminator lcc 0
#python3 $MDL_CODE google/electra-base-discriminator trofi 0
#python3 $MDL_CODE google/electra-base-discriminator vua_pos 0
#python3 $MDL_CODE google/electra-base-discriminator vua_verb 0
#python $MDL_CODE google/electra-base-discriminator hypo_en 0

#python $MDL_CODE bert-base-uncased-random-weights lcc 0
#python $MDL_CODE bert-base-uncased-random-weights trofi 0
#python $MDL_CODE bert-base-uncased-random-weights vua_verb 0
#python $MDL_CODE bert-base-uncased-random-weights vua_pos 0
#python $MDL_CODE bert-base-uncased-random-weights hypo_en 0
#python $MDL_CODE roberta-base-random-weights hypo_en 0
