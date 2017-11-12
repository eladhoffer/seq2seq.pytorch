#!/bin/bash
# Adapted from https://github.com/tensorflow/tensor2tensor

decoded_file=$1
WMT_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
mosesdecoder="${WMT_DIR}/mosesdecoder"
tok_gold_targets="${WMT_DIR}/newstest2014.tok.de"


# Tokenize.
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decoded_file > $decoded_file.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $tok_gold_targets > $tok_gold_targets.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decoded_file.tok > $decoded_file.tok.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $tok_gold_targets.atat < $decoded_file.tok.atat
rm $decoded_file.tok.atat
rm $decoded_file.tok
rm $tok_gold_targets.atat
