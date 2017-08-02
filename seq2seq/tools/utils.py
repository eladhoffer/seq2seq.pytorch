import os
import logging.config
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.charts import Line, defaults
from .config import PAD

defaults.width = 800
defaults.height = 400
defaults.tools = 'pan,box_zoom,wheel_zoom,box_select,hover,resize,reset,save'


def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    def plot(self, *kargs, **kwargs):
        line = Line(data=self.results, *kargs, **kwargs)
        self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = torch.optim.__dict__[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def batch_sequences(seqs, max_length=None, batch_first=False, sort=False, pack=False):
    max_length = max_length or float('inf')
    batch_dim, time_dim = (0, 1) if batch_first else (1, 0)
    if len(seqs) == 1:
        lengths = [min(len(seqs[0]), max_length)]
        seq_tensor = seqs[0][:lengths[0]]
        seq_tensor = seq_tensor.unsqueeze(batch_dim)
    else:
        if sort:
            seqs.sort(key=len, reverse=True)
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        tensor_size = (len(seqs), batch_length) if batch_first \
            else (batch_length, len(seqs))
        seq_tensor = torch.LongTensor(*tensor_size).fill_(PAD)
        for i, seq in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor.narrow(time_dim, 0, end_seq).select(batch_dim, i)\
                .copy_(seq[:end_seq])
    if pack:
        seq_tensor = pack_padded_sequence(
            seq_tensor, lengths, batch_first=batch_first)
    return (seq_tensor, lengths)
