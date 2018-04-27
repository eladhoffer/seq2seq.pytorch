DATASET=${1:-"IWSLT15"}
DATASET_DIR=${2:-"/media/drive/Datasets/iwslt15"}
OUTPUT_DIR=${3:-"/media/drive/nmt_results"}

python main.py \
  --save en_vi_word_tf \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --results_dir ${OUTPUT_DIR} \
  --epochs 12 \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'hidden_size': 512, 'dropout': 0.2, 'forget_bias': 1, 'embedding_dropout': 0.2, \
                   'tie_embedding': False, 'transfer_hidden': True, 'bias_classifier': False, \
                   'encoder': {'num_layers': 1, 'bidirectional': True, 'num_bidirectional': 1, 'context_transform': 512}, \
                   'decoder': {'num_layers': 2, 'concat_attention': True, \
                               'attention': {'mode': 'dot_prod', 'bias': False, 'normalize': True, 'dropout': 0, 'query_transform': False, 'output_transform': True, 'output_nonlinearity': ''}}}" \
  --data_config "{'tokenization':'word', 'shared_vocab':False}" \
  --b 256 \
  --max_length 50 \
  --device_ids 0 \
  --uniform_init 0.1 \
  --trainer Seq2SeqTrainer \
  --optimization_config "[{'step': 0, 'optimizer': 'SGD', 'lr': 1},
                          {'step': 9000, 'lr': 0.5},
                          {'step': 10000, 'lr': 0.25},
                          {'step': 11000, 'lr': 0.125},
                          {'step': 12000, 'lr': 0.0625},
                          {'step': 13000, 'lr': 0.03125}]" \
