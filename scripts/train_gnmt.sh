DATASET='WMT16_de_en'
DATASET_DIR='./datasets/data/wmt16_de_en'

python main.py \
  --save gnmt_wmt16 \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --model GNMT \
  --model_config "{'num_layers': 8, 'hidden_size': 256}" \
  --devices "{'input': 0, 'encoder': 1, 'decoder': 2}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --trainer Seq2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 1e-3},
                          1: {'optimizer': 'Adam', 'lr': 1e-4},
                          2: {'optimizer': 'SGD', 'lr': 1e-4}}" \


#
