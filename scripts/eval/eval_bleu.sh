
#!/bin/bash
# Adapted from https://github.com/tensorflow/tensor2tensor

PAIR='en-de'
TARGET_LNG='de'
SRC_TOK="./data/wmt16_de_en/newstest2014.tok.en"
REF_TOK="./data/wmt16_de_en/newstest2014.tok.de"


MODEL_PATH=$1
BEAM=${2:-"1"}
TARGET_TOK=${3:-"${MODEL_PATH}/newstest2014.tok.en.translated"}
MOSES=${4:-"./data/wmt16_de_en/mosesdecoder/"}
CHECKPOINT="${MODEL_PATH}/model_best.pth";



echo "Translating ${SRC_TOK} into ${TARGET_TOK}"
python translate.py  -i ${SRC_TOK} \
  -m "${CHECKPOINT}" \
  -o $TARGET_TOK \
  --batch-size 64 \
  --device-ids 0 \
  --use-moses False \
  --max-input-length 200 \
  --max-output-length 200 \
  --beam-size ${BEAM} \
  --length-normalization 0.6;



# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'

perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $REF_TOK > $REF_TOK.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $TARGET_TOK > $TARGET_TOK.atat

# Get BLEU.
perl $MOSES/scripts/generic/multi-bleu.perl $REF_TOK.atat < $TARGET_TOK.atat
