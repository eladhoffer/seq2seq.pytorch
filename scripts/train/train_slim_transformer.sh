DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/home/ehoffer/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}

WARMUP="4000"
LR0="512**(-0.5)"

python main.py \
  --save slim_transformer \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model Transformer \
  --model_config "{'num_layers': 3, 'hidden_size': 512, 'num_heads': 8, 'inner_linear': 4096, 'inner_groups': 2, 'stateful': True, 'layer_norm': False}" \
  --data_config "{'moses_pretok': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 32 \
  --max_length 50 \
  --device_ids 0 \
  --label_smoothing 0.1 \
  --trainer Seq2SeqTrainer \
  --optimization_config "[{'step_lambda':
                          \"lambda t: { \
                              'optimizer': 'Adam', \
                              'lr': ${LR0} * min(t ** -0.5, t * ${WARMUP} ** -1.5), \
                              'betas': (0.9, 0.98), 'eps':1e-9}\"
                          }]"
