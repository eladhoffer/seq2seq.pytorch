DATASET='CocoCaptions'
DATASET_DIR='/media/ssd/Datasets/COCO'

python main.py \
  --save captions \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --model ResNetCaptionGenerator \
  --model_config "{'num_layers': 2, 'hidden_size': 256, 'dropout': 0.2}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 32 \
  --trainer Img2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 1e-3},
                          1: {'optimizer': 'Adam', 'lr': 1e-4},
                          2: {'optimizer': 'SGD', 'lr': 1e-4}}" \
