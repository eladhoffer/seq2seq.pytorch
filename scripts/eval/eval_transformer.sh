#!/bin/bash

MODEL_PATH=$1
WMT_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
SRC=${2:-"newstest2014.clean.en"}
TARGET=${3:-"newstest2014.clean.de"}
BEAM=4

# FILES="('${MODEL_PATH}/checkpoint.pth.tar','${MODEL_PATH}/checkpoint1.pth.tar','${MODEL_PATH}/checkpoint2.pth.tar','${MODEL_PATH}/checkpoint3.pth.tar','${MODEL_PATH}/checkpoint4.pth.tar')";
FILES="/home/ehoffer/.torch/models/transformer_en_de-d4bd08ed.pth"
TFILE="${MODEL_PATH}/${SRC}.translation";
rm $TFILE
echo "Translating into ${TFILE}"
python translate.py  ${WMT_DIR}/${SRC} \
  -m "${FILES}" \
  -o $TFILE \
  --batch_size 32 \
  --devices 1 \
  --max_sequence_length 100 \
  --beam_size ${BEAM} \
  --length_normalization 1;
wc -l $TFILE
./scripts/eval/get_en_de_bleu.sh $TFILE
