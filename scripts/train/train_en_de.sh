DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}

python main.py \
  --save de_en_wmt16 \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'hidden_size': 512, 'dropout': 0.2, \
                   'tie_embedding': True, 'transfer_hidden': False, \
                   'encoder': {'num_layers': 3, 'bidirectional': True, 'num_bidirectional': 1}, \
                   'decoder': {'num_layers': 4, 'context_transform': 512, \
                               'attention': {'mode': 'bahdanau', 'normalize': True}}}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 64 \
  --max_length 50 \
  --pack_encoder_inputs \
  --trainer Seq2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 5e-4},  \
                          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},  \
                          {'epoch': 6, 'optimizer': 'Adam', 'lr': 5e-5},  \
                          {'epoch': 8, 'optimizer': 'Adam', 'lr': 1e-5},  \
                          {'epoch': 10, 'optimizer': 'SGD', 'lr': 1e-4}]" \
