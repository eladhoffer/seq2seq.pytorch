DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}

python main.py \
  --save transformer \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Transformer \
  --model_config "{'num_layers': 6, 'hidden_size': 512, 'num_heads': 8, 'inner_linear': 2048}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 32 \
  --grad_clip 0 \
  --trainer Seq2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 1e-4, 'betas': (0.9, 0.98), 'eps':1e-9},
                          4: {'lr': 5e-5},
                          6: {'lr': 1e-5},
                          8: {'optimizer': 'SGD', 'lr': 1e-4, 'momentum':0.9},
                          10: {'lr': 1e-5, 'momentum':0}}" \
