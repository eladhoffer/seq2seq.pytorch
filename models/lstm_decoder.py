import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from beam_search import CaptionGenerator
from data import __normalize as normalize_values
from torchvision import transforms


class LSTMDecoder(nn.Module):

    def __init__(self, vocab, embedding_size=256, rnn_size=256, num_layers=2,
                 share_embedding_weights=False):
        super(LSTMDecoder, self).__init__()
        self.vocab = vocab
        self.rnn = nn.LSTM(embedding_size, rnn_size, num_layers=num_layers)
        self.classifier = nn.Linear(rnn_size, len(vocab))
        self.embedder = nn.Embedding(len(self.vocab), embedding_size)
        if share_embedding_weights:
            self.embedder.weight = self.classifier.weight

    def forward(self, inputs, state=None):
        embeddings = torch.cat([img_feats, embeddings], 0)
        packed_embeddings = pack_padded_sequence(embeddings, lengths)
        feats, state = self.rnn(packed_embeddings)
        pred = self.classifier(feats[0])

        return pred, state
