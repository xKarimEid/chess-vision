"""
Script for training model
"""


import os
import numpy as np
import torch

from tokenizer.basic_tokenizer import CONTEXT_SIZE
from models.feedforward import FeedForward



BATCH_SIZE = 32
# Directory where data is stored
data_dir = "data"
# Set the correct device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split):
    """
    Get train/test batch, create a memory mapping every time to 
    avoid a memory leak.
    https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    """

    # Recreate np.memmap every batch to avoid memory leak
    ids_path = os.path.join(data_dir, f"{split}_ids.bin")
    # Create path for binned scores
    scores_path = os.path.join(data_dir, f"{split}_percentages.bin")
    # Create memory mapped array
    x = np.memmap(ids_path, dtype=np.uint8, mode = 'r')
    y = np.memmap(scores_path, np.uint8, mode = 'r')
    # Reshape using context size used to tokenize data
    # B here is the total arr length
    B = int(x.shape[0] / CONTEXT_SIZE)
    x = x.reshape(B, CONTEXT_SIZE)

    # Generate random ids
    ix = torch.randint(B, (BATCH_SIZE,))
    # Convert to pytorch and convert type to int32 to make it compatible with model
    x, y = torch.from_numpy(x[ix].astype(np.int32)), torch.from_numpy(y[ix].astype(np.int32))

    return x.to(device), x.to(device)

for _ in range(100):
    x, y = get_batch('train')
    print("done")