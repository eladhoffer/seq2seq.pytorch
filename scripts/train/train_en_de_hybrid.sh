DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"/media/drive/nmt_results"}

WARMUP="4000"
LR0="512**(-0.5)"

python main.py \
  --save de_en_wmt16_hybrid \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --model HybridSeq2Seq \
  --model_config "{'tie_embedding': True, \
                   'encoder': {'type': 'transformer', 'num_layers': 6, 'hidden_size': 512, 'num_heads': 8, 'inner_linear': 2048}, \
                   'decoder': {'type': 'recurrent', 'hidden_size': 512, 'dropout': 0.2, 'tie_embedding': True, 'num_layers': 3, 'concat_attention': True,\
                               'attention': {'mode': 'dot_prod', 'dropout': 0, 'output_transform': True, 'output_nonlinearity': 'relu'}}}" \
  --data_config "{'moses_pretok': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 128 \
  --max_length 50 \
  --devices 1 \
  --grad_clip "{'decoder': 5.}" \
  --trainer Seq2SeqTrainer \
  --optimization_config "[{'step_lambda':
                          \"lambda t: { \
                              'optimizer': 'Adam', \
                              'lr': ${LR0} * min(t ** -0.5, t * ${WARMUP} ** -1.5), \
                              'betas': (0.9, 0.98), 'eps':1e-9}\"
                          }]"
