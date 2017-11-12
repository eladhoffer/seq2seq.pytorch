import torch
from seq2seq.tools import batch_sequences

s1 = torch.LongTensor([1,2,3,4,5,6])
s2 = torch.LongTensor([10,20,30])

seqs = [s1,s2]
batch = batch_sequences(seqs, max_length=4, augment=True)
print(batch)
