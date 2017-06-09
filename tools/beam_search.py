"""Class for generating sequences from an image-to-text model.
Adapted from https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/sequence_generator.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax
import heapq
import math
from .config import EOS


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Sequence.

        Args:
          sentence: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
          metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class SequenceGenerator(object):
    """Class to generate sequences from an image-to-text model."""

    def __init__(self,
                 model,
                 eos_id=EOS,
                 beam_size=3,
                 max_sequence_length=50,
                 length_normalization_factor=0.0):
        """Initializes the generator.

        Args:
          model: recurrent model, with inputs: (input, state) and outputs len(vocab) values
          beam_size: Beam size to use when generating sequences.
          max_sequence_length: The maximum sequence length before stopping the search.
          length_normalization_factor: If != 0, a number x such that sequences are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of sequences depending on their lengths. For example, if
            x > 0 then longer sequences will be favored.
        """
        self.model = model
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.max_sequence_length = max_sequence_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, initial_input, initial_state=None):
        """Runs beam search sequence generation on a single image.

        Args:
          initial_state: An initial state for the recurrent model

        Returns:
          A list of Sequence sorted by descending score.
        """

        def get_topk(inputs, states):
            logits, new_states = self.model(inputs, states)

            logprobs = log_softmax(logits.view(-1, logits.size(-1)))
            logprobs, words = logprobs.topk(self.beam_size, 1)
            return words.data, logprobs.data, new_states

        partial_sequences = TopN(self.beam_size)
        complete_sequences = TopN(self.beam_size)

        words, logprobs, new_state = get_topk(initial_input, initial_state)

        for k in range(self.beam_size):
            cap = Sequence(
                sentence=[words[0, k]],
                state=new_state,
                logprob=logprobs[0, k],
                score=logprobs[0, k])
            partial_sequences.push(cap)

        # Run beam search.
        for _ in range(self.max_sequence_length - 1):
            partial_sequences_list = partial_sequences.extract()
            partial_sequences.reset()
            input_feed = torch.LongTensor([c.sentence[-1]
                                           for c in partial_sequences_list])
            input_feed = input_feed.view(1, -1)
            if initial_input.is_cuda:
                input_feed = input_feed.cuda()
            input_feed = Variable(input_feed, volatile=True)
            state_feed = [c.state for c in partial_sequences_list]
            state_feed = merge_states(state_feed)

            words, logprobs, new_states = get_topk(input_feed, state_feed)
            for i, partial_sequence in enumerate(partial_sequences_list):
                state = select_state(new_states, i)
                for k in range(self.beam_size):
                    w = words[i, k]
                    sentence = partial_sequence.sentence + [w]
                    logprob = partial_sequence.logprob + logprobs[i, k]
                    score = logprob
                    if w == self.eos_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence)**self.length_normalization_factor
                        beam = Sequence(sentence, state, logprob, score)
                        complete_sequences.push(beam)
                    else:
                        beam = Sequence(sentence, state, logprob, score)
                        partial_sequences.push(beam)
            if partial_sequences.size() == 0:
                # We have run out of partial candidates; happens when beam_size
                # = 1.
                break

        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.
        if not complete_sequences.size():
            complete_sequences = partial_sequences

        caps = complete_sequences.extract(sort=True)

        return [c.sentence for c in caps], [c.score for c in caps]


def merge_states(state_list):
    if isinstance(state_list[0], tuple):
        return tuple([merge_states(s) for s in zip(*state_list)])
    else:
        batch_dim = 1 if state_list[0].dim() == 3 else 0
        return torch.cat(state_list, batch_dim)


def select_state(state, i):
    if isinstance(state, tuple):
        return tuple(select_state(s, i) for s in state)
    else:
        batch_dim = 1 if state.dim() == 3 else 0
        return state.narrow(batch_dim, i, 1)
