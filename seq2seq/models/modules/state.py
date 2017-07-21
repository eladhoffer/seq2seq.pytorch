import torch
from torch.autograd import Variable


class State(object):
    __slots__ = ['batch_first', 'hidden', 'inputs', 'outputs', 'context',
                 'attention', 'attention_score', 'mask']

    def __init__(self, hidden=None, inputs=None, outputs=None, context=None, attention=None,
                 attention_score=None, mask=None, batch_first=False):
        self.hidden = hidden
        self.outputs = outputs
        self.inputs = inputs
        self.context = context
        self.attention = attention
        self.mask = mask
        self.batch_first = batch_first
        self.attention_score = attention_score

    def __select_state(self, state, i, type_state='hidden'):
        if isinstance(state, tuple):
            return tuple(self.__select_state(s, i, type_state) for s in state)
        elif isinstance(state, Variable) or torch.is_tensor(state):
            if type_state == 'hidden':
                batch_dim = 0 if state.dim() < 3 else 1
            else:
                batch_dim = 0 if self.batch_first else 1
            return state.narrow(batch_dim, i, 1)
        else:
            return state

    def __merge_states(self, state_list, type_state='hidden'):
        if state_list is None:
            return None
        if isinstance(state_list[0], State):
            return State().from_list(state_list)
        if isinstance(state_list[0], tuple):
            return tuple([self.__merge_states(s, type_state) for s in zip(*state_list)])
        else:
            if isinstance(state_list[0], Variable) or torch.is_tensor(state_list[0]):
                if type_state == 'hidden':
                    batch_dim = 0 if state_list[0].dim() < 3 else 1
                else:
                    batch_dim = 0 if self.batch_first else 1
                return torch.cat(state_list, batch_dim)
            else:
                assert state_list[1:] == state_list[:-1]  # all items are equal
                return state_list[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            state_list = [self[idx] for idx in range(
                index.start or 0, index.stop or len(self), index.step or 1)]
            return State().from_list(state_list)
        else:
            item = State()
            for s in self.__slots__:
                value = getattr(self, s, None)
                if isinstance(value, State):
                    selected_value = value[index]
                else:
                    selected_value = self.__select_state(value, index, s)
                setattr(item, s, selected_value)
            return item

    def from_list(self, state_list):
        for s in self.__slots__:
            values = [getattr(item, s, None) for item in state_list]
            setattr(self, s, self.__merge_states(values, s))
        return self
