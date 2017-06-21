DATASET='CocoCaptions'
DATASET_DIR=${1:-"/media/ssd/Datasets/COCO"}
OUTPUT_DIR=${2:-"./results"}

python main.py \
  --save captions_no_finetune \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model ResNetCaptionGenerator \
  --model_config "{'num_layers': 1, 'hidden_size': 256, 'dropout': 0.2}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000}" \
  --b 64 \
  --trainer Img2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 1e-3},
                          4: {'optimizer': 'Adam', 'lr': 1e-4},
                          8: {'optimizer': 'SGD', 'lr': 1e-4}}" \
