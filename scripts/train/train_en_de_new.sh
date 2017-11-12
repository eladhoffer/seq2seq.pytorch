DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"/media/drive/nmt_results"}

python main.py \
  --save de_en_wmt16_better \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'hidden_size': 512, 'dropout': 0.2, \
                   'tie_embedding': True, 'transfer_hidden': False, \
                   'encoder': {'num_layers': 3, 'bidirectional': True, 'num_bidirectional': 1, 'context_transform': 512}, \
                   'decoder': {'num_layers': 3, 'concat_attention': True,\
                               'attention': {'mode': 'dot_prod', 'dropout': 0, 'output_transform': True, 'output_nonlinearity': 'relu'}}}" \
  --data_config "{'moses_pretok': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 128 \
  --max_length 80 \
  --devices 1 \
  --trainer Seq2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 8, 'lr':1e-4},
                          {'epoch': 10, 'lr': 1e-5}]" \
