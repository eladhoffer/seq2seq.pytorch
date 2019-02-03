#! /usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e


OUTPUT_DIR=${1:-"./data/wmt17_de_en"}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

mkdir -p $OUTPUT_DIR_DATA

echo "Downloading Europarl v7. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz \
  http://www.statmt.org/europarl/v7/de-en.tgz

echo "Downloading Common Crawl corpus. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/common-crawl.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

echo "Downloading News Commentary v12. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/nc-v12.tgz \
  http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz

echo "Downloading Rapid 2016. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR_DATA}/rapid2016.tgz \
  http://data.statmt.org/wmt17/translation-task/rapid2016.tgz

echo "Downloading dev/test sets"
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/dev.tgz \
  http://data.statmt.org/wmt17/translation-task/dev.tgz
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/test.tgz \
  http://data.statmt.org/wmt17/translation-task/test.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
mkdir -p "${OUTPUT_DIR_DATA}/nc-v12"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v12.tgz" -C "${OUTPUT_DIR_DATA}/nc-v12"
mkdir -p "${OUTPUT_DIR_DATA}/rapid2016"
tar -xvzf "${OUTPUT_DIR_DATA}/rapid2016.tgz" -C "${OUTPUT_DIR_DATA}/rapid2016"
mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"

# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.en" \
  "${OUTPUT_DIR_DATA}/nc-v12/training/news-commentary-v12.de-en.en" \
  "${OUTPUT_DIR_DATA}/rapid2016/rapid2016.de-en.en" \
  > "${OUTPUT_DIR}/train.en"
wc -l "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/europarl-v7.de-en.de" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.de" \
  "${OUTPUT_DIR_DATA}/nc-v12/training/news-commentary-v12.de-en.de" \
  "${OUTPUT_DIR_DATA}/rapid2016/rapid2016.de-en.de" \
  > "${OUTPUT_DIR}/train.de"
wc -l "${OUTPUT_DIR}/train.de"

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Convert SGM files
# Convert newstest2014 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.en

# Convert newstest2015 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2015-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2015.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2015-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2015.en

# Convert newstest2016 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2017-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2017.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test/newstest2017-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/test/newstest2017.en

# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest20*.de ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest20*.en ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.de ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test/newstest20*.en ${OUTPUT_DIR}

# Tokenize data
for f in ${OUTPUT_DIR}/*.de; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < $f > ${f%.*}.tok.de
done

for f in ${OUTPUT_DIR}/*.en; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean all corpora
for f in ${OUTPUT_DIR}/*.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl -ratio 1.5 $fbase de en "${fbase}.clean" 1 250
done


echo "All done."
