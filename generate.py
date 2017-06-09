import torch
from tools.config import EOS, BOS
from tools.beam_search import SequenceGenerator
checkpoint = torch.load('./results/en_de_onmt/checkpoint.pth.tar')
model = checkpoint['model']
src_tok, target_tok = checkpoint['tokenizers'].values()

src = 'hellow world'
src = src_tok.tokenize(src, insert_start=[BOS], insert_end=[EOS])

enc_hidden, context = model.encoder(src.view(-1, 1))
init_output = model.make_init_decoder_output(context)

enc_hidden = (model._fix_enc_hidden(enc_hidden[0]),
              model._fix_enc_hidden(enc_hidden[1]))


def gen_func(x, s):
    out, dec_hidden, _attn = self.decoder(x, s,
                                          context, init_output)
    return out, dec_hidden
generator = SequenceGenerator(model=gen_func)
SequenceGenerator.beam_search(src, enc_hidden)
