# Seq2Seq in PyTorch
This is a complete suite for training sequence-to-sequence models in [PyTorch](www.pytorch.org). It consists of several models and code to both train and infer using them.

Using this code you can train:
* Neural-machine-translation (NMT) models
* Language models
* Image to caption generation
* Skip-thought sentence representations
* And more...
 
 ## Installation
 ```
 git clone --recursive https://github.com/eladhoffer/seq2seq.pytorch
 cd seq2seq.pytorch; python setup.py develop
 ```
 
## Models
Models currently available:
* Simple Seq2Seq recurrent model
* Recurrent Seq2Seq with attentional decoder
* [Google neural machine translation](https://arxiv.org/abs/1609.08144) (GNMT) recurrent model
* Transformer - attention-only model from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)

## Datasets
Datasets currently available:

* WMT16
* WMT17
* OpenSubtitles 2016
* COCO image captions
* [Conceptual captions](https://ai.googleblog.com/2018/09/conceptual-captions-new-dataset-and.html)

All datasets can be tokenized using 3 available segmentation methods:

* Character based segmentation
* Word based segmentation
* Byte-pair-encoding (BPE) as suggested by [bpe](https://arxiv.org/abs/1508.07909) with selectable number of tokens.  

After choosing a tokenization method, a vocabulary will be generated and saved for future inference.


## Training methods
The models can be trained using several methods:

* Basic Seq2Seq - given encoded sequence, generate (decode) output sequence. Training is done with teacher-forcing.
* Multi Seq2Seq - where several tasks (such as multiple languages) are trained simultaneously by using the data sequences as both input to the encoder and output for decoder.
* Image2Seq - used to train image to caption generators.

## Usage
Example training scripts are available in ``scripts`` folder. Inference examples are available in ``examples`` folder.

* example for training a [transformer](https://arxiv.org/abs/1706.03762)
 on WMT16 according to original paper regime:
```
DATASET=${1:-"WMT16_de_en"}
DATASET_DIR=${2:-"./data/wmt16_de_en"}
OUTPUT_DIR=${3:-"./results"}

WARMUP="4000"
LR0="512**(-0.5)"

python main.py \
  --save transformer \
  --dataset ${DATASET} \
  --dataset-dir ${DATASET_DIR} \
  --results-dir ${OUTPUT_DIR} \
  --model Transformer \
  --model-config "{'num_layers': 6, 'hidden_size': 512, 'num_heads': 8, 'inner_linear': 2048}" \
  --data-config "{'moses_pretok': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 128 \
  --max-length 100 \
  --device-ids 0 \
  --label-smoothing 0.1 \
  --trainer Seq2SeqTrainer \
  --optimization-config "[{'step_lambda':
                          \"lambda t: { \
                              'optimizer': 'Adam', \
                              'lr': ${LR0} * min(t ** -0.5, t * ${WARMUP} ** -1.5), \
                              'betas': (0.9, 0.98), 'eps':1e-9}\"
                          }]"
```

* example for training attentional LSTM based model with 3 layers in both encoder and decoder:
```
python main.py \
  --save de_en_wmt17 \
  --dataset ${DATASET} \
  --dataset-dir ${DATASET_DIR} \
  --results-dir ${OUTPUT_DIR} \
  --model RecurrentAttentionSeq2Seq \
  --model-config "{'hidden_size': 512, 'dropout': 0.2, \
                   'tie_embedding': True, 'transfer_hidden': False, \
                   'encoder': {'num_layers': 3, 'bidirectional': True, 'num_bidirectional': 1, 'context_transform': 512}, \
                   'decoder': {'num_layers': 3, 'concat_attention': True,\
                               'attention': {'mode': 'dot_prod', 'dropout': 0, 'output_transform': True, 'output_nonlinearity': 'relu'}}}" \
  --data-config "{'moses_pretok': True, 'tokenization':'bpe', 'num_symbols':32000, 'shared_vocab':True}" \
  --b 128 \
  --max-length 80 \
  --device-ids 0 \
  --trainer Seq2SeqTrainer \
  --optimization-config "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
                          {'epoch': 6, 'lr': 5e-4},
                          {'epoch': 8, 'lr':1e-4},
                          {'epoch': 10, 'lr': 5e-5},
                          {'epoch': 12, 'lr': 1e-5}]" \
```
