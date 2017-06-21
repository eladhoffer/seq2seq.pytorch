DATASET=${1:-"OpenSubtitles2016"}
DATASET_DIR=${2:-"./data/OpenSubtitles2016"}
OUTPUT_DIR=${3:-"./results"}

python main.py \
  --save transformer_en_he_dual \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Transformer \
  --model_config "{'num_layers': 4, 'hidden_size': 256, 'num_heads': 4, 'inner_linear': 1024}" \
  --data_config "{'languages': ['he','en'], 'mark_language': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 32 \
  --grad_clip 0 \
  --trainer MultiSeq2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 1e-4, 'betas': (0.9, 0.98), 'eps':1e-9},
                          4: {'lr': 1e-4},
                          8: {'optimizer': 'SGD', 'lr': 1e-4}}" \
