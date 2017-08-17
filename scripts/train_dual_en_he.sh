DATASET=${1:-"OpenSubtitles2016"}
DATASET_DIR=${2:-"./data/OpenSubtitles2016"}
OUTPUT_DIR=${3:-"./results"}

python main.py \
  --save en_he_dual \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'hidden_size': 512, 'dropout': 0.2, \
                   'tie_embedding': True, 'transfer_hidden': False, \
                   'encoder': {'num_layers': 3, 'bidirectional': True, 'num_bidirectional': 1}, \
                   'decoder': {'num_layers': 4,  'num_pre_attention_layers': 1, 'context_transform': 512, \
                               'attention': {'mode': 'bahdanau', 'normalize': True}}}" \
  --data_config "{'languages': ['he','en'], 'mark_language': True, 'tokenization':'bpe',\
                  'num_symbols':32000, 'shared_vocab':True}" \
  --b 64 \
  --max_length 80 \
  --trainer MultiSeq2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
                          {'epoch': 8, 'optimizer': 'SGD', 'lr': 1e-4}]" \
