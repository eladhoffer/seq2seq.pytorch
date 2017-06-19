DATASET='WMT16_de_en'
DATASET_DIR='./datasets/data/wmt16_de_en'

python main.py \
  --save recurrent_attention_wmt16 \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'num_layers': 2, 'hidden_size': 256, 'dropout': 0.2}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 128 \
  --trainer Seq2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 1e-3},
                          1: {'optimizer': 'Adam', 'lr': 1e-4},
                          2: {'optimizer': 'SGD', 'lr': 1e-4}}" \
