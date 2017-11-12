#!/bin/bash

MODEL_PATH=$1
WMT_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
SRC=${2:-"newstest2014.tok.clean.en"}
TARGET=${3:-"newstest2014.tok.clean.de"}
BEAM=8

FILES="('${MODEL_PATH}/checkpoint.pth.tar','${MODEL_PATH}/checkpoint1.pth.tar','${MODEL_PATH}/checkpoint2.pth.tar','${MODEL_PATH}/checkpoint3.pth.tar','${MODEL_PATH}/checkpoint4.pth.tar')";
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
  --length_normalization 0.6;
wc -l $TFILE
./scripts/eval/get_en_de_bleu_pretok.sh $TFILE
