import torch
from torch import nn
import math


class Embedding(nn.Module):
    """
    Converts token IDs into dense vectors (embeddings)
    A neural network can't able to do meaningful math on integer ID's
    like [9246, 6734]. it needs continous numbers in vectors.

    think of it as giant lookup table
    Row 9246 -> vector of 768 floats (the 'meaning' of 'cat')
    Row 6734 -> vector of 768 floats (the 'meaning' of 'sat')

    This table is LEARNED. Initially random, backpropagation gradually moves related tokens closer together in the 768 dimensional space

    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Create the embedding table ( a learnable matrix)

        Args:
            vocab_size: How many unique tokens exist (50,257 for GPT-2)
            d_model: size of each embedding vector

        Examples by model scale:
            GPT-2 small: vocab=50257, d_model=768   -> table is 50257 x 768
            GPT-2 medium: vocab=50257, d_model=1024  -> table is 50257 x 1024
            GPT-3 small:  vocab=50257, d_model=4096  -> table is 50257 x 4096
            GPT-3 large:  vocab=50257, d_model=12288 -> table is 50257 x 12288

        The embedding dimension determines how much 'space' each word has to express
        its meaning. Bigger d_model = more nuanced menaings can be captured, at the cost of
        more parameters and slower training.
        """
        super().__init__()

        """
        The actual embedding weights - a [vocab_size, d_model] matrix
        nn.Embeddng is an optimized lookup table. When you pass a tensor of token ID's,
        it returns the corresponding rows. It's backed by a standard weight matrix, so gradients
        flow through it just like any nn.Linear layer.

        Internally nn.Embedding is essentially:
            def forward(self, x):
                return self.weight[x] # index into weight matrix
        """

        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for each token ID in the input
        Input shape: [batch_size, seq_len] - each cell is a token ID
        Output shape: [batch_size, sew_len, d_model] - each cell is a vector

        Example walkthrough:
            input: [[464, 3797]] # ['The', 'cat']
            Step 1: Look up row 464 → [768 floats] for "The"
                    Look up row 3797 → [768 floats] for "cat"
            Step 2: Scale by sqrt(768) ≈ 27.7
            Output: [[[v0..v767], [v0..v767]]] # 2 vectors of 768 numbers

        WHY each dimension:
            batch_size = how many sequences we process at once (parallelism)
            seq_len    = how many tokens per sequence (context window)
            d_model    = how rich each token's representation is (expressiveness)
        """

        # index into embedding matrix
        # for each token id, return its row. this is an O(1) lookup operation - very fast, even for 50K+ vocab
        embeddings = self.embed(x) # [batch, seq_len, d_model]

        # Return embeddings unchanged
        # we use RoPE (Rotary Positional Emebddings) for position embedding. RoPE rotates rather than adds, so no scaling is needed. LLaMA and Mistral follow the same convention
        return embeddings