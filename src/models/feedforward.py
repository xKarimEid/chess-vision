"""
Basic ffwd model

context length 68

Takes in a fen ->
fen is int encoded by using a standard encoding method
('p':1 and so forth)
no positional encoding
padded to fixed length

embedding matrix
dense hidden layer with activation
final output layer as binn

"""


import torch
import torch.nn as nn
import torch.nn.functional as F

# Network configuration settings
N_CHARS = 25
N_EMBED = 8
N_BINS = 64
CONTEXT_LENGTH = 68

class FeedForward(nn.Module):
    """
    Basic Neural network architecture
    """
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_CHARS, N_EMBED)
        self.flatten = nn.Flatten(end_dim = -1)
        self.hidden = nn.Linear(N_EMBED*CONTEXT_LENGTH, N_BINS)

    def forward(self, x):
        """
        Forward function and get the probs
        """

        x = self.emb(x) # B, T, C
        x = self.flatten(x) # B, T*C
        logits = self.hidden(x) # B, N_BINS
        probs = F.softmax(logits, dim = -1) # B, N_BINS

        return probs

    def get_best_position_idx(self, x):
        """
        We need to find the position with the highest evaluation
        from all given positions. To do this we first need to find the 
        binned evaluation for each position and then choose the position with the
        highest binned evaluation
        """

        # Find probs for each position
        probs = self.forward(x) # B, n_bins
        # Assign position evaluation based on the highest probability
        binned_eval = torch.argmax(probs, dim = 1)
        # Find the position idx with the best evaluation (highest binned evaluation)
        best_position = torch.argmax(binned_eval)

        return best_position
