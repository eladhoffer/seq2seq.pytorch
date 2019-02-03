#! /usr/bin/env bash
lang1=${1:-"en"}
lang2=${2:-"he"}
OUTPUT_DIR=${3:-"./data/OpenSubtitles2018"}
DEV_SIZE=${4:-"6000"}
MAX_LENGTH=${5:-"250"}

SITE="http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/"
MOSES_CLEANER="https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl"

OSUB_PREFIX="${OUTPUT_DIR}/${lang1}-${lang2}/OpenSubtitles.${lang1}-${lang2}"
OSUB_TRAIN_PREFIX="${OUTPUT_DIR}/${lang1}-${lang2}/train"
OSUB_DEV_PREFIX="${OUTPUT_DIR}/${lang1}-${lang2}/dev"
TXT_FILE="${lang1}-${lang2}.txt"
http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-he.txt.zip
mkdir -p $OUTPUT_DIR;
mkdir -p "$OUTPUT_DIR/${lang1}-${lang2}";

wget -nc "${SITE}${TXT_FILE}.zip" -O "${OUTPUT_DIR}/${TXT_FILE}.zip";

echo "Extracting ${OUTPUT_DIR}/${TXT_FILE}.zip..."
unzip "${OUTPUT_DIR}/${TXT_FILE}.zip" -d "$OUTPUT_DIR/${lang1}-${lang2}";

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Clean all corpora
echo "Cleaning ${OSUB_PREFIX}..."
${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 ${OSUB_PREFIX} ${lang1} ${lang2} "${OSUB_PREFIX}.clean" 1 "${MAX_LENGTH}"

rm "${OSUB_PREFIX}.${lang1}";

tail -n ${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang1}" > "${OSUB_DEV_PREFIX}.clean.${lang1}";
head -n -${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang1}" > "${OSUB_TRAIN_PREFIX}.clean.${lang1}";
rm "${OSUB_PREFIX}.clean.${lang1}";

rm "${OSUB_PREFIX}.${lang2}";
tail -n ${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang2}" > "${OSUB_DEV_PREFIX}.clean.${lang2}";
head -n -${DEV_SIZE} "${OSUB_PREFIX}.clean.${lang2}" > "${OSUB_TRAIN_PREFIX}.clean.${lang2}";
rm "${OSUB_PREFIX}.clean.${lang2}";

echo "All done."
