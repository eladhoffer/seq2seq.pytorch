#!/bin/bash

MODEL_PATH=$1
BEAM=${2:-"1"}
WMT=${3:-"wmt14"}
LANG=${4:-"en-de"}
SRC=${5:-"${MODEL_PATH}/${WMT}-${LANG}.src"}
TARGET=${6:-"${MODEL_PATH}/${WMT}-${LANG}.target"}
CHECKPOINT="${MODEL_PATH}/checkpoint.pth.tar";

sacrebleu -t ${WMT} -l ${LANG} --echo src > ${SRC}

echo "Translating ${SRC} into ${TARGET}"
rm $TARGET
python translate.py  ${SRC} \
  -m "${CHECKPOINT}" \
  -o $TARGET \
  --batch_size 16 \
  --device_ids 3 \
  --max_sequence_length 80 \
  --beam_size ${BEAM} \
  --length_normalization 0.6;

cat ${TARGET} | sacrebleu -t ${WMT} -l ${LANG}