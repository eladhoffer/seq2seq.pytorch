#!/bin/bash

MODEL_PATH=$1
BEAM=${2:-"1"}
WMT=${3:-"wmt14"}
LANG=${4:-"en-de"}
SRC=${5:-"${MODEL_PATH}/${WMT}-${LANG}.src"}
TARGET=${6:-"${MODEL_PATH}/${WMT}-${LANG}.target"}
CHECKPOINT="${MODEL_PATH}/checkpoint.pth";

sacrebleu -t ${WMT} -l ${LANG} --echo src > ${SRC}

echo "Translating ${SRC} into ${TARGET}"
rm $TARGET
python translate.py  -i ${SRC} \
  -m "${CHECKPOINT}" \
  -o $TARGET \
  --batch-size 32 \
  --device-ids 0 \
  --max-input-length 100 \
  --max-output-length 100 \
  --beam-size ${BEAM} \
  --length-normalization 0.6;

cat ${TARGET} | sacrebleu -t ${WMT} -l ${LANG}