"""
Script for training model
"""


import os
import numpy as np
import torch

from tokenizer.basic_tokenizer import CONTEXT_SIZE

# Config params
lr = 1e-4
N_ITER = 60000
EVAL_ROUND = 1000
EVAL_ITER = 20
BATCH_SIZE = 32

# Directory where data is stored
data_dir = "data"
# Directory where model weights are stored
model_dir = '/content/drive/My Drive/chess/models/model.pkl'
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

@torch.no_grad()
def evaluate_model(model):
    """
    Evaluates model on both train and test data. The loss
    is calculated EVAL_ITER times, and the mean is returned
    for both splits.
    """

    out = {}
    model.eval()
    for split in ['train', 'test']:
        lossi = torch.zeros(EVAL_ITER)
        for k in range(EVAL_ITER):
            # Get batch
            xb, yb = get_batch(split)
            # Forward pass
            logits, loss = model(xb, yb)
            # Save loss
            lossi[k] = loss.item()

        out[split] = lossi.mean()

        # Convert state back to train
        model.train()

    return out


def train_loop(model):
    """
    Simple train loop which trains the model N_ITER times.
    Evaluation happens every EVAL_ROUND. If test loss is better than the
    previous loss, the weights are checkpointed.
    """

    min_test = float("inf")
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    for i in range(N_ITER):
        xb, yb = get_batch('train')
        optimizer.zero_grad()
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        if i % EVAL_ROUND == 0:
            loss = evaluate_model(model)
            train_loss, test_loss = loss['train'], loss['test']
            print(f"{i}/{N_ITER} Train loss: {train_loss:.4f} Test loss: {test_loss:.4f}")

            if test_loss < min_test:
                min_test = test_loss
                checkpoint = {'model': model.state_dict()}
                torch.save(checkpoint, model_dir)
                print(f"writing new weights to disk with loss {min_test}")
