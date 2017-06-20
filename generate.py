#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from tools.config import EOS, BOS
from tools.inference import Translator
import math
from tools.quantize import quantize_model, dequantize_model

cuda = False
checkpoint = torch.load('./results/gnmt_wmt16/checkpoint.pth.tar')
model = checkpoint['model']
# quantize_model(model)
# checkpoint['state_dict'] = checkpoint['model'].state_dict()
# torch.save(checkpoint,
#            './results/en_he_onmt/checkpoint_quantized.pth.tar')
# dequantize_model(model)
src_tok, target_tok = checkpoint['tokenizers'].values()

translation_model = Translator(model,
                               src_tok=src_tok,
                               target_tok=target_tok,
                               beam_size=5,
                               length_normalization_factor=0,
                               cuda=cuda)

# print(translation_model.translate('hello world')[::-1])

while True:
    src= raw_input()
    print(translation_model.translate(src))
