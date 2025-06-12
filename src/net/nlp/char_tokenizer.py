"""
Character level tokenizer
"""

import numpy as np

def char_tokenize(strings: list[str], block_size: int):
    """
    Tokenize the list of strings by building blocks and also return a translator from character to index
    """
    vocab = set(list(".".join(strings)))
    str_to_int = stoi(vocab)
    tokens = []
    for string in strings:
        inp = "." * block_size + string + "." * block_size
        for i in range(len(inp) - block_size):
            tokens.append([str_to_int[s] for s in list(inp[i: i + block_size])])

    return np.array(tokens) , vocab

def stoi(vocab: set[str]) -> dict[str, int]:
    """
    Build string to int converter
    """
    return {v: i for i, v in enumerate(vocab)}

def itos(vocab: set[str]) -> dict[str, int]:
    """
    Build int to string converter
    """
    return dict(enumerate(vocab))