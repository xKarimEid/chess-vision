"""
Tokenizer converts FEN notation to ints and
padds the output to a specific length
"""

import numpy as np

# All possible characters in a fen, adding '.' for padding purposes
STOI = '. -/12345678BKNPQRbknpqrw'
CONTEXT_SIZE = 71

class BasicTokenizer:
    """
    Basic tokenizer for encoding/decoding a fen position
    """

    def __init__(self):
        self.stoi = {char: idx for idx, char in enumerate(STOI)}
        self.itos = {idx: char for idx, char in enumerate(STOI)}

    def _encode_fen(self, fen):
        """
        Encodes the first part of fen to a fixed length of 
        integers. The encodings are padded to a constant length of 71
        """

        # Disregards castling and move number information
        parts = fen.split(" ")
        fen_part = ' '.join(parts[:2])
        # Convert chars to integers
        ids = [self.stoi[char] for char in fen_part]
        # Padds to a constand length
        padd_length = 71 - len(ids)
        padded = [0 for _ in range(padd_length)]
        # Add the paddings to the actual integers
        padded.extend(ids)

        return np.array(padded)

    def encode(self, fens):
        """
        Takes in a list of fen and encodes them
        """
        # Makes sure fens is a list
        if isinstance(fens, str):
            fens = [fens]

        # Store the encoded fens
        encoded_fens = np.zeros(shape=(len(fens), CONTEXT_SIZE))
        # Iterate and encode each fen
        for i, fen in enumerate(fens):
            encoded = self._encode_fen(fen)
            encoded_fens[i, :] = encoded[:]

        return encoded_fens

    def decode(self, ids):
        """
        Takes a list of encoded positions and decodes them
        """

        assert isinstance(ids[0], list)

        decoded_fens = []
        # Decode each integer except for the padding integer
        for encoding in ids:
            chars = [self.itos[idx] for idx in encoding if idx != 0]
            decoded_fens.append(chars)

        fens = []
        # Join the chars of the fen together
        for fen in decoded_fens:
            f = ''.join(fen)
            fens.append(f)

        # Return a single fen string if only one fen is present
        if len(fens) == 1:
            return fens[0]

        return fens