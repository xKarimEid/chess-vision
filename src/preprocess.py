"""
Downloads and preprocesses training/test data
"""

import os
import numpy as np
from datasets import load_dataset
from tokenizer.basic_tokenizer import BasicTokenizer, CONTEXT_SIZE

# Global variables
TOTAL_SIZE = 1e7 # 10MB * 65 = 650 MB
TOT_ARR_LEN = int(TOTAL_SIZE/2) # each entry takes up two bytes (np.16int)
TEST_SIZE = 0.05 # Size of test data
N_BINS = 64 # Used for bucketing percentages

def convert_scores_percentages(scores):
    """
    Convert scores to winning percentages
    """

    winning_percentages = np.zeros(shape=len(scores), dtype = np.float16)
    for i, score in enumerate(scores):
        win_percentage = 0.0
        if score != 'C':
            win_percentage = 50 + 50 * (2 / (1 + np.exp(-0.00368208 * float(score))) - 1)

        winning_percentages[i] = win_percentage

    return winning_percentages

def bin_percentages(percentages):
    """
    Describe the binning
    """

    boundaries = np.linspace(0, 100, N_BINS)
    # Substract one to make buckets start from 0
    buckets = np.digitize(percentages, boundaries) - 1

    return buckets


def preprocess(game):
    """
    Convert scores to winning percentages
    """

    # Encode fens to integers
    ids = encoder.encode(game['fens'])
    # Convert scores to winning percentages
    winning_percentages = convert_scores_percentages(game['scores'])
    # Bin the percentages to buckets
    buckets = bin_percentages(winning_percentages)
    # Gather ids and percentages in dictionary
    out = {'ids': ids, 'win_percentages': buckets}

    return out

if __name__ == "__main__":

    # Initialize encoder
    encoder = BasicTokenizer()
    # Initialize dataset connection
    dataset = load_dataset(path="mauricett/lichess_sf", trust_remote_code=True, streaming = True)
    # Streaming list of processed games
    data = dataset.map(preprocess)
    # Train, test splits
    splits = [('train', 1 - TEST_SIZE), ('test', TEST_SIZE)]

    # # Create filepaths and storage arrays for current split
    for split, p in splits:

        arr_len = int(TOT_ARR_LEN*p)
        # Creating filepath to data
        encodings_path = os.path.join(os.path.dirname(
             os.path.dirname(__file__)), f'data/{split}_ids.bin'
             )

        # Creating memory mapped array at filepath
        ids_arr = np.memmap(encodings_path, dtype=np.uint8, mode='w+', shape=(arr_len, CONTEXT_SIZE))

        # Creating winning percentages filepath
        scores_path = os.path.join(
             os.path.dirname(os.path.dirname(__file__)), f'data/{split}_percentages.bin'
             )

        # Creating memory mapped array at filepath
        percentages_arr = np.memmap(scores_path, dtype=np.uint8, mode='w+', shape=(arr_len,))

        # Keep track of the indices
        write_idx = 0
        i = 0

        for game in data['train']:
            # Create mini batches
            batch_ids = game['ids']
            batch_scores = game['win_percentages']

            # Number of positions in new batch, this varies from game to game
            B = len(batch_ids)

            # If the new batch does not fit into the allocated space
            if write_idx + B >= arr_len:
                # Cut the overflowing bits
                B = arr_len - write_idx

            ids_arr[write_idx: write_idx + B, :] = batch_ids[:B, :]
            percentages_arr[write_idx: write_idx + B] = batch_scores[:B]

            # Increment starting position of batch_idx
            write_idx += B

            if write_idx == arr_len:
                break

            if i % 1000 == 0:
                print(write_idx, (write_idx/arr_len)*100)

            i += 1

