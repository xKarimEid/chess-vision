"""
An encoder only transformer implementation with prenorm layers
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from tokenizer.basic_tokenizer import CONTEXT_SIZE


# Global variables
N_EMBED = 512
HEAD_SIZE = 32
N_HEADS = 16
N_CHARS = 25
N_BINS = 64
N_BLOCKS = 8
DROPOUT = 0.05

# Configure to use gpu if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NewGELU(nn.Module):
    """
    The GELU activation function, as applied in the Google BERT repository.
    For further details, please refer to the Gaussian Error Linear Units (GELU) 
    paper at https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MultiHeadAttention(nn.Module):
    """
    Implementation of batched multihead attention to fully utilize
    gpu
    """

    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(N_EMBED, 3 * N_EMBED)
        self.proj = nn.Linear(N_EMBED, N_EMBED)

    def forward(self, x):
        """
        calculate query, key, values for all heads in batch and move head 
        forward to be the batch dim Split the result to three vectors 
        each having N_EMBED channels
        """

        B, T, C = x.size()

        q, k ,v  = self.attn(x).split(N_EMBED, dim=2)
        k = k.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, N_HEADS, C // N_HEADS).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.proj(y)

        return y

class Block(nn.Module):
    """
    Implementation of a decoder block. The block contains layer norm, self attention,
    and a multilayer perceptron
    """

    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBED)
        self.attn = MultiHeadAttention()
        self.ln_2 = nn.LayerNorm(N_EMBED)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(N_EMBED, 4 * N_EMBED),
            c_proj  = nn.Linear(4 * N_EMBED, N_EMBED),
            act     = NewGELU(),
            ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class ModelHead(nn.Module):
    """
    A MLP network mounted on the head of the transformer.
    Transforms a (B, T, C) tensor to (B, N_BINS) by pooling along 
    the T dimension
    """

    def __init__(self):
        super().__init__()
        self.head = nn.ModuleDict(dict(
            h1 = nn.Linear(N_EMBED, N_EMBED),
            act = NewGELU(),
            h2 = nn.Linear(N_EMBED, N_BINS, bias = False)
        ))

    def forward(self, x):
        x = self.head.act(self.head.h1(x)) # B, T, C
        # Calculates the channel means and forwards to the final layer
        x = self.head.h2(x.mean(dim = 1)) # B, N_BINS

        return x


class Network(nn.Module):
    """
    Putting together the modules creating a sequence classifier
    using encoder architecture utilizing self attention
    """

    def __init__(self):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(N_CHARS, N_EMBED),
                wpe = nn.Embedding(CONTEXT_SIZE, N_EMBED),
                h = nn.ModuleList([Block() for _ in range(N_BLOCKS)]),
                ln_f = nn.LayerNorm(N_EMBED),
            ))
        self.lm_head = ModelHead()

    def forward(self, x, targets):
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # shape (1, T)

        tok_emb = self.transformer.wte(x) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss
