#! /usr/bin/env bash
lang1=${1:-"en"}
lang2=${2:-"he"}
OUTPUT_DIR=${3:-"./data/OpenSubtitles2016"}
DEV_SIZE=${4:-"6000"}
MAX_LENGTH=${5:-"80"}

SITE="http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/"
MOSES_CLEANER="https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl"

OSUB_PREFIX="${OUTPUT_DIR}/${lang1}-${lang2}/all"
OSUB_TRAIN_PREFIX="${OUTPUT_DIR}/${lang1}-${lang2}/train"
OSUB_DEV_PREFIX="${OUTPUT_DIR}/${lang1}-${lang2}/dev"
TMX_FILE="${lang1}-${lang2}.tmx"

mkdir -p $OUTPUT_DIR;
mkdir -p "$OUTPUT_DIR/${lang1}-${lang2}";

wget -nc "${SITE}${TMX_FILE}.gz" -O "${OUTPUT_DIR}/${TMX_FILE}.gz";

echo "Extracting ${OUTPUT_DIR}/${TMX_FILE}.gz..."
gunzip "${OUTPUT_DIR}/${TMX_FILE}.gz";
cat "${OUTPUT_DIR}/${TMX_FILE}" | sed -n "s/.*xml:lang=\"${lang1}\"><seg>\(.*\)<\/seg>.*/\1/p" > "${OSUB_PREFIX}.${lang1}";
cat "${OUTPUT_DIR}/${TMX_FILE}" | sed -n "s/.*xml:lang=\"${lang2}\"><seg>\(.*\)<\/seg>.*/\1/p" > "${OSUB_PREFIX}.${lang2}";
rm "${OUTPUT_DIR}/${TMX_FILE}";

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Clean all corpora
echo "Cleaning ${OSUB_PREFIX}..."
${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl ${OSUB_PREFIX} ${lang1} ${lang2} "${OSUB_PREFIX}.clean" 1 "${MAX_LENGTH}"

tail -n ${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang1}" > "${OSUB_DEV_PREFIX}.clean.${lang1}";
head -n -${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang1}" > "${OSUB_TRAIN_PREFIX}.clean.${lang1}";
rm "${OSUB_PREFIX}.clean.${lang1}";

tail -n ${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang2}" > "${OSUB_DEV_PREFIX}.clean.${lang2}";
head -n -${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang2}" > "${OSUB_TRAIN_PREFIX}.clean.${lang2}";
rm "${OSUB_PREFIX}.clean.${lang2}";

echo "All done."
