"""Class for generating sequences
Adapted from https://github.com/tensorflow/models/blob/master/im2txt/im2txt/inference_utils/sequence_generator.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import heapq
from .config import EOS


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, sentence, state, logprob, score, attention=None):
        """Initializes the Sequence.

        Args:
          sentence: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.attention = attention

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
                 batch_first=False,
                 get_attention=False,
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
        self.get_attention = get_attention

    def beam_search(self, initial_input, initial_state=None):
        """Runs beam search sequence generation on a single image.

        Args:
          initial_state: An initial state for the recurrent model

        Returns:
          A list of Sequence sorted by descending score.
        """
        batch_size = len(initial_input)
        partial_sequences = [TopN(self.beam_size) for _ in range(batch_size)]
        complete_sequences = [TopN(self.beam_size) for _ in range(batch_size)]

        words, logprobs, new_state = self.model.generate(
            initial_input, initial_state,
            k=self.beam_size,
            feed_all_timesteps=True,
            get_attention=self.get_attention)
        for b in range(batch_size):
            for k in range(self.beam_size):
                seq = Sequence(
                    sentence=initial_input[b] + [words[b][k]],
                    state=new_state[b],
                    logprob=logprobs[b][k],
                    score=logprobs[b][k],
                    attention=None if not self.get_attention else [new_state[b].attention_score])
                partial_sequences[b].push(seq)

        # Run beam search.
        for _ in range(self.max_sequence_length - 1):
            partial_sequences_list = [p.extract() for p in partial_sequences]
            for p in partial_sequences:
                p.reset()
            flattened_partial = [
                s for sub_partial in partial_sequences_list for s in sub_partial]

            input_feed = [c.sentence for c in flattened_partial]
            state_feed = [c.state for c in flattened_partial]
            if len(input_feed) == 0:
                # We have run out of partial candidates; happens when beam_size=1
                break
            words, logprobs, new_states \
                = self.model.generate(
                    input_feed, state_feed,
                    k=self.beam_size, get_attention=self.get_attention)

            idx = 0
            for b in range(batch_size):
                for partial in partial_sequences_list[b]:
                    state = new_states[idx]
                    if self.get_attention:
                        attention = partial.attention + [new_states[idx].attention_score]
                    else:
                        attention = None
                    for k in range(self.beam_size):
                        w = words[idx][k]
                        sentence = partial.sentence + [w]
                        logprob = partial.logprob + logprobs[idx][k]
                        score = logprob

                        if w == self.eos_id:
                            if self.length_normalization_factor > 0:
                                score /= len(sentence)**self.length_normalization_factor
                            beam = Sequence(sentence, state,
                                            logprob, score, attention)
                            complete_sequences[b].push(beam)
                        else:
                            beam = Sequence(sentence, state,
                                            logprob, score, attention)
                            partial_sequences[b].push(beam)
                    idx += 1

        # If we have no complete sequences then fall back to the partial sequences.
        # But never output a mixture of complete and partial sequences because a
        # partial sequence could have a higher score than all the complete
        # sequences.
        for b in range(batch_size):
            if not complete_sequences[b].size():
                complete_sequences[b] = partial_sequences[b]
        seqs = [complete.extract(sort=True)[0]
                for complete in complete_sequences]
        return seqs
