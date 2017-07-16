DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}

python main.py \
  --save de_en_wmt16_gnmt \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'hidden_size': 512, 'dropout': 0.2, 'residual': True, \
                   'tie_embedding': True, 'transfer_hidden': False, \
                   'encoder': {'num_layers': 7, 'bidirectional': True, 'num_bidirectional': 1}, \
                   'decoder': {'num_layers': 8, 'context_transform': 256, \
                               'concat_attention': True, 'num_pre_attention_layers': 1, \
                               'attention': {'mode': 'bahdanau', 'normalize': True}}}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 32 \
  --trainer Seq2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 1e-4},  \
                          4: {'optimizer': 'Adam', 'lr': 5e-5},  \
                          8: {'optimizer': 'Adam', 'lr': 1e-5},  \
                          10: {'optimizer': 'SGD', 'lr': 1e-4}}" \
