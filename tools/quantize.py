from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = round(initial_zero_point)

    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x, num_bits=8):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def quantize_state_dict(float_dict, num_bits=8):
    quant_dict = {}
    for k, p in float_dict.items():
        quant_dict[k] = quantize_tensor(p, num_bits)
    return quant_dict


def dequantize_state_dict(quant_dict):
    float_dict = {}
    for k, p in quant_dict.items():
        float_dict[k] = dequantize_tensor(p)
    return float_dict

def quantize_model(model):
    qparams = {}

    for n,p in model.state_dict():
        qp = quantize_tensor(p)
        qparams[n +'.quantization.scale'] =  torch.FloatTensor([qp.scale])
        qparams[n +'.quantization.zero_point'] = torch.ByteTensor([qp.zero_point])
        p.copy_(qp.tensor)
    model.type('torch.ByteTensor')
    for n,p in qparams.items():
        model.register_buffer(n,p)

def dequantize_model(model):
    model.float()

    qparams = {}

    for n,p in model.state_dict():
        qp = quantize_tensor(p)
        qparams[n +'.quantization.scale'] =  torch.FloatTensor([qp.scale])
        qparams[n +'.quantization.zero_point'] = torch.ByteTensor([qp.zero_point])
        p.copy_(qp.tensor)
    for n,p in qparams.items():
        model.register_buffer(n,p)
