DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}


python main.py \
  --save recurrent_attention_small \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'num_layers': 2, 'hidden_size': 256, 'dropout': 0.2, 'tie_embedding': True}" \
  --data_config "{'moses_pretok': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 192 \
  --max_length 50 \
  --devices 1 \
  --label_smoothing 0.1 \
  --trainer Seq2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 4, 'optimizer': 'Adam', 'lr': 5e-4},
                          {'epoch': 8, 'optimizer': 'Adam', 'lr': 1e-4}]" \
