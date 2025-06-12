"""
Testing full embedding MLP on toy dataset
      --- This test was written by Chat GPT ---
"""

import numpy as np
from net.util import tanh, dtanh, SEED
from net.model.mlp import MLP
from net.train import grad_descent  # assuming that's the module name

def test_embedding_learning():
    np.random.seed(SEED)

    # 1. Toy character vocab: a=0, b=1, c=2, d=3
    vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    vocab_size = len(vocab)
    embed_dim = 4  # small embedding size

    # 2. Dataset: simple binary classification task
    # Class 0: 'ab', 'ac', 'ad'
    # Class 1: 'ba', 'ca', 'da'
    samples = [("ab", 0), ("ac", 0), ("ad", 0),
               ("ba", 1), ("ca", 1), ("da", 1)]
    
    xs = np.array([[vocab[c] for c in s] for s, _ in samples])  # shape: (6, 2)
    ys = np.array([label for _, label in samples])              # shape: (6,)

    # 3. Initial embeddings
    emb_init = np.random.randn(vocab_size, embed_dim)

    # 4. Build MLP: input size = 2 * embed_dim (flattened pair), one hidden layer
    model = MLP(size=(2 * embed_dim, 6, 2), sigma=tanh, dsigma=dtanh, emb=emb_init.copy())

    # 5. Train using your gradient descent
    grad_descent(
        model=model,
        xs=xs,
        ys=ys,
        steps=300,
        batch_size=3,
        lr_start=0.1
    )

    # 6. Evaluate results
    model.forward(xs)
    preds = np.argmax(model.layers[-1].value, axis=1)
    print("\nPredictions:", preds)
    print("Ground Truth:", ys)

    # 7. Sanity check: embeddings changed
    delta = np.linalg.norm(model.emb.value - emb_init)
    print("Embedding Change Magnitude:", delta)
    assert delta > 1e-3, "Embeddings did not update!"

    # 8. Show embeddings
    print("\nFinal Embeddings:")
    for char, idx in vocab.items():
        print(f"'{char}': {model.emb.value[idx]}")

    # 9. Optional: check training success
    accuracy = np.mean(preds == ys)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    assert accuracy > 0.8, "Model failed to learn pattern!"

    print("\nEmbedding learning test passed.")

test_embedding_learning()
