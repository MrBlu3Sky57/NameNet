"""
Testing grad update for embeddings with dummy dataset.
      --- This test was written by Chat GPT ---
"""

import numpy as np
from net import MLP


def test_embedding_backward():
    np.random.seed(0)

    vocab_size = 10
    embed_dim = 3
    seq_len = 2
    hidden_size = 5
    output_size = 2
    batch_size = 4

    # Create dummy embeddings, input, and labels
    emb = np.random.randn(vocab_size, embed_dim)
    xs = np.random.randint(0, vocab_size, size=(batch_size, seq_len))  # (4, 2)
    ys = np.random.randint(0, output_size, size=batch_size)

    def dummy_sigma(x): return x  # Identity for simplicity
    def dummy_dsigma(x): return np.ones_like(x)

    # Create model
    model = MLP(size=(seq_len * embed_dim, hidden_size, output_size),
                sigma=dummy_sigma,
                dsigma=dummy_dsigma,
                emb=emb.copy())

    # Forward and backward
    model.forward(xs)
    y_onehot = np.eye(output_size)[ys]
    model.backward(y_onehot)

    # Embedding gradient check
    grad = model.emb.grad
    used_indices = xs.flatten()

    # Check: only used indices have non-zero gradients
    for idx in range(vocab_size):
        if idx in used_indices:
            assert not np.allclose(grad[idx], 0), f"Grad for used index {idx} is zero"
        else:
            assert np.allclose(grad[idx], 0), f"Grad for unused index {idx} is nonzero"

test_embedding_backward()
