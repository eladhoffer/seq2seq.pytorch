DATASET='WMT16_de_en'
DATASET_DIR='/media/drive/Datasets/wmt16_de_en'

python main.py \
  --save de_en_wmt16 \
  --dataset ${DATASET} \
  --dataset_dir ${DATASET_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model_config "{'num_layers_encoder': 2, 'num_layers_decoder': 4, \
                   'bidirectional_encoder': True, \
                   'hidden_size': 512, 'dropout': 0.2, \
                   'tie_embedding': True, \
                   'transfer_hidden': False, 'attention': 'bahdanau'}" \
  --data_config "{'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 32 \
  --trainer Seq2SeqTrainer \
  --optimization_config "{0: {'optimizer': 'Adam', 'lr': 5e-4},
                          4: {'optimizer': 'Adam', 'lr': 1e-4},
                          6: {'optimizer': 'Adam', 'lr': 5e-5},
                          8: {'optimizer': 'Adam', 'lr': 1e-5},
                          10: {'optimizer': 'SGD', 'lr': 1e-4}}" \


#
