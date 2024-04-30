"""
Add docstring
"""

import torch
from torch import nn
import torch.nn.functional as F

# Global params
N_EMBED = 32
HEAD_SIZE = 8
N_HEADS = 4
N_CHARS = 25
N_BINS = 64
N_BLOCKS = 4
DROPOUT = 0.05
CONTEXT_SIZE = 68

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionHead(nn.Module):
    """
    Add description here
    """

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(N_EMBED, HEAD_SIZE, bias = False)
        self.k = nn.Linear(N_EMBED, HEAD_SIZE, bias = False)
        self.v = nn.Linear(N_EMBED, HEAD_SIZE, bias = False)
        #self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Add description here
        """

        B, T, C = x.shape
        queries = self.q(x) # (B, T, HEAD_SIZE)
        keys = self.k(x) # (B, T, HEAD_SIZE)
        values = self.v(x) # (B, T, HEAD_SIZE)

        # The communication between the nodes
        wei = keys @ queries.transpose(-2, -1) * C**-0.5 # (B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        #wei = self.dropout(wei)

        out = wei @ values # (B, T, HEAD_SIZE)
        return out

class MultiHeadAttention(nn.Module):
    """
    Add docstring
    """

    def __init__(self):
        super().__init__()
        assert N_EMBED == N_HEADS * HEAD_SIZE
        self.sa_heads = nn.ModuleList([AttentionHead() for _ in range(N_HEADS)])
        self.proj = nn.Linear(N_EMBED, N_EMBED, bias = True)
        #self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Add description here
        """
        x = torch.cat([h(x) for h in self.sa_heads], dim=-1)
        x = self.proj(x)
        return x

class FFWD(nn.Module):
    """
    Add docstring
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, N_EMBED*4, bias = True),
            nn.ReLU(),
            nn.Linear(N_EMBED*4, N_EMBED, bias = True),
            #nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        """
        Add docs
        """

        return self.net(x)

class Block(nn.Module):
    """
    Add docstring
    """

    def __init__(self):
        super().__init__()
        self.mh_attention = MultiHeadAttention()
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ffwd = FFWD()
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        """
        Add doc
        """
        x = self.mh_attention(self.ln1(x)) + x
        x = self.ffwd(self.ln2(x)) + x
        return x

class Network(nn.Module):
    """
    Add docstring
    """
    def __init__(self):
        super().__init__()
        # Chess piece embedding
        self.embed = nn.Embedding(N_CHARS, N_EMBED)
        # Chess squares embedding
        self.positional_embedding = nn.Embedding(CONTEXT_SIZE, N_EMBED)
        # Self Attention
        self.blocks = nn.Sequential(*[Block() for _ in range(N_BLOCKS)])
        # Flatten out to stretch the rows to all channels for all T
        self.flatten = nn.Flatten(start_dim = 1)
        self.ln = nn.LayerNorm(CONTEXT_SIZE*N_EMBED)
        self.head = nn.Linear(CONTEXT_SIZE*N_EMBED, N_BINS)

    def forward(self, x, targets = None):
        """
        Add docstring
        """
        # x is (B, T)
        piece_embed = self.embed(x) # (B, T, N_EMBED)
        positional_embed = self.positional_embedding(torch.arange(CONTEXT_SIZE, device = device))
        # Adding together piece embedding and positional embedding
        x = piece_embed + positional_embed # (B, T, N_EMBED)
        x = self.blocks(x) # (B, T, HEAD_SIZE)
        x = self.flatten(x) # (B, T*HEAD_SIZE)
        x = self.ln(x)
        logits = self.head(x) # (B, N_BINS)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss
