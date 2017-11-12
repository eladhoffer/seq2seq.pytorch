DATASET='CocoCaptions'
DATASET_DIR=${1:-"/media/ssd/Datasets/COCO"}
OUTPUT_DIR=${2:-"./results"}

python main.py \
  --save captions_resnet50 \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Img2Seq \
  --model_config "{'encoder': {'model': 'resnet50', 'finetune': False}, \
                   'decoder': {'num_layers': 2, 'hidden_size': 256, 'dropout': 0, \
                               'context_transform': 256, 'tie_embedding': False, \
                               'attention': {'mode': 'bahdanau', 'normalize': True}}}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':8000}" \
  --b 128 \
  --epochs 10\
  --trainer Img2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
                          {'epoch': 8, 'optimizer': 'SGD', 'lr': 1e-4, 'momentum': 0.9}]"
