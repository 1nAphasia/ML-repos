import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())

corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(f"vocab.token_freqs[:10]:{vocab.token_freqs[:10]}")
