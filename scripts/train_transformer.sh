DATASET='OpenSubtitles2016'
DATASET_DIR='./datasets/data/OpenSubtitles2016'

python main.py \
  --save transformer_en_he_small_dual \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --model Transformer \
  --model_config "{'num_layers': 3, 'hidden_size': 256, 'num_heads': 4}" \
  --data_config "{'languages': ['he','en'], 'mark_language': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 32 \
  --trainer MultiSeq2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'SGD', 'lr': 0.1},
                          1: {'optimizer': 'SGD', 'lr': 1e-2},
                          2: {'optimizer': 'SGD', 'lr': 1e-3}}" \
