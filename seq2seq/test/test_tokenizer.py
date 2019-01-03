from seq2seq.tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer

test_file = '../../README.md'
text = 'machine learning - hello world'

tokenizer = Tokenizer(vocab_file='test.vocab')
tokenizer.get_vocab([test_file], from_filenames=True)
tokenized = tokenizer.tokenize(text)
print(tokenized, tokenizer.detokenize(tokenized))

char_tokenizer = CharTokenizer(vocab_file='test_char.vocab')
char_tokenizer.get_vocab([test_file], from_filenames=True)
tokenized = char_tokenizer.tokenize(text)
print(tokenized, char_tokenizer.detokenize(tokenized))

bpe_tokenizer = BPETokenizer('test_bpe.codes', 'test_bpe.vocab', num_symbols=100, use_moses=True)
bpe_tokenizer.learn_bpe([test_file], from_filenames=True)
bpe_tokenizer.get_vocab([test_file], from_filenames=True)

tokenized = bpe_tokenizer.tokenize(text)
print(tokenized, bpe_tokenizer.detokenize(tokenized))
