"""
Character level tokenizer
"""

import numpy as np

def char_tokenize(strings: list[str], block_size: int):
    """
    Tokenize the list of strings by building blocks and also return a vocabulary of valid outputs
    """
    vocab = set(list("".join(strings)))
    tokens = []
    for string in strings:
        inp = "." * block_size + string + "." * block_size
        for i in range(len(inp) - block_size):
            tokens.append(list(inp[i: i + block_size]))

    return np.array(tokens) , vocab