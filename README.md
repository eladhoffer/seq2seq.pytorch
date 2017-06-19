# seq2seq.pytorch
This is a complete suite for training sequence-to-sequence models in [PyTorch](www.pytorch.org). It consists of several models and code to both train and infer using them.

Using this code you can train:
* Neural-machine-translation (NMT) models
* Language models
* Image to caption generation
* Skip-thought sentence representations
* And more..


## Datasets
Datasets currently available:

* WMT16
* OpenSubtitles 2016
* English books from [skipthought](www.?.com)
* COCO image captions

All datasets can be tokenized using 3 available segmentation methods:

* Character based segmentation
* Word based segmentation
* Byte-pair-encoding (BPE) as suggested by [bpe](?) with selectable number of tokens.  

After choosing a tokenization method, a vocabulary will be generated and saved for future inference.

## Models
Models currently available:
* Simple Seq2Seq recurrent model
* Recurrent Seq2Seq with attentional decoder
* Google neural machine translation (GNMT) recurrent model
* Transformer - attention-only model from [transformer](?)
* ByteNet - convolution based encoder+decoder

## Training methods
The models can be trained using several methods:

* Basic Seq2Seq - given encoded sequence, generate (decode) output sequence. Training is done with teacher-forcing.
* Multi Seq2Seq - where several tasks (such as multiple languages) are trained simultaneously by using the data sequences as both input to the encoder and output for decoder.

## Installation
### Dependencies

### Data preparing
