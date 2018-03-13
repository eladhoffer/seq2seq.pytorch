DATASET='CocoCaptions'
DATASET_DIR=${1:-"/home/ehoffer/Datasets/COCO"}
OUTPUT_DIR=${2:-"./results"}

python main.py \
  --save captions_resnet50 \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Img2Seq \
  --model_config "{'encoder': {'model': 'resnet50', 'finetune': False, 'context_transform': 256, 'spatial_context': False}, \
                   'transfer_hidden': True,
                   'decoder': {'type': 'recurrent', 'num_layers': 1, 'hidden_size': 256, 'embedding_dropout': 0.1, 'dropout': 0.2,  'tie_embedding': False}}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':2000}" \
  --b 128 \
  --epochs 10\
  --devices 2 \
  --trainer Img2SeqTrainer \
  --optimization_config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
                          {'epoch': 8, 'optimizer': 'SGD', 'lr': 1e-4, 'momentum': 0.9}]"
