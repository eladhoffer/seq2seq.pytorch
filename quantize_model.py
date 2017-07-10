import torch
from seq2seq.tools.quantize import quantize_model
name = 'captions_no_finetune'
checkpoint = torch.load('./results/captions_no_finetune/checkpoint.pth.tar')
quantize_model(checkpoint['model'])
checkpoint['state_dict'] = checkpoint['model'].state_dict()
torch.save(checkpoint, './results/captions_no_finetune/%s.pth.tar' % name)
