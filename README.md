# NameNet
A deep feedforward network for name generation.

## Table of Contents
- [Plan](#plan)
- [Implementation](#implementation)
- [Takeaways](#takeaways)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Plan
In this project I will build up the mathematical formulations of a deep feedforward network
(See the writeups folder) then use this theory to build a character level language model for name generation.

## Implementation
I discuss the mathematical details of my training and overall model design
in my Backprop write up. The main idea of this project was to get an understanding of deep feed forward networks from the ground up, using NumPy to build the model from scratch. I modelled this as a baby version
of the neural probabilistic generative model proposed in the Bengio et al.
paper without any recurrent formulations. I will come back to this later
building a character level RNN then upgrading it to a word level one.

## Takeaways
I was quite proud of the results of my model, it produced a lot of real names and had no junk output. I thoroughly enjoyed building this model and learned a lot about how to optimize a model once the core framework was built, as I think a lot of performance can be attributed to hyperparameter tuning and regularization. This also the first time I built
a machine learning project as a proper python project not solely in jupyter notebooks. This made my code a lot easier to work with and made implementing integrated testing very easy, which saved me a lot of headache!

## Acknowledgments

The ideas in **NameNet** are based on foundational research and inspiring implementations, including:

- [Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — *A Neural Probabilistic Language Model* provided the conceptual backbone for building a statistical language model using neural networks.
- [Andrej Karpathy's `char-rnn`](https://github.com/karpathy/char-rnn) — This repo inspired the character-level generation format and workflow.
- [Cybenko (1989)](https://www.sciencedirect.com/science/article/pii/0893608089900208) — *Approximation by superpositions of a sigmoidal function* laid the theoretical groundwork for universal function approximation with neural networks.

While all code and mathematical derivations were developed independently, these resources deeply informed the design and motivation of the project.

## License
This project is licensed under the [MIT License](LICENSE).