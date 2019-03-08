DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"/media/drive/Datasets/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}
NUM_GPUS=4

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} main.py \
  --save de_en_wmt17 \
  --dataset ${DATASET} \
  --dataset-dir ${DATASET_DIR} \
  --results-dir ${OUTPUT_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model-config "{'hidden_size': 512, 'dropout': 0.2, \
                   'tie_embedding': True, 'transfer_hidden': False, \
                   'encoder': {'num_layers': 3, 'bidirectional': True, 'num_bidirectional': 1, 'context_transform': 512, 'pack_inputs': False}, \
                   'decoder': {'num_layers': 3, 'concat_attention': True,\
                               'attention': {'mode': 'dot_prod', 'dropout': 0, 'output_transform': True, 'output_nonlinearity': 'relu'}}}" \
  --data-config "{'moses_pretok': True, 'tokenization':'bpe', 'tokenization_config':{'num_symbols':32000}, 'shared_vocab':True}" \
  --b 32 \
  --max-length 50 \
  --trainer Seq2SeqTrainer \
  --optimization-config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 6, 'lr': 5e-4},
                          {'epoch': 8, 'lr':1e-4},
                          {'epoch': 10, 'lr': 5e-5},
                          {'epoch': 12, 'lr': 1e-5}]" \
