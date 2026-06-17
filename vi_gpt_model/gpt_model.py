import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt_model_config import GPTConfig
from v_transformer.transformer import TransformerBlock
from v_transformer.rms_norm import RMSNorm

class GPT(nn.Module):
    """
    a complete decoder-only transformer language model
    this single class combines everything we built so far
    embedding -> N x transformer blocks -> output projection

    decoder-only means it generates text left to right
    (causal/autoregressive), without encoder (encoder looks at the full response)

    This is the same architecture family as:
    GPT-2 (12 layers, 768 dims), GPT-3 (96 layers, 12288 dims),
    LLaMA 3 (32-80 layers), Mistral (32 layers)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # token embedding
        # lookup table: token ID -> dense vector
        # converts integers (id's) into the continuous vectors
        # that neural networks can work with.
        # shape [50257, 768] - one row per vocabulary token
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model)

        # dropout applied to embeddings
        # early dropout prevents the model from overfitting to specific embedding values during training
        self.embd_dropout = nn.Dropout(p=config.embd_dropout)

        # transformer blocks
        # stack N identical transformer layers
        # nn.ModuleList registers each block so pytorch tracks their params for training.
        # A regualar python list would not be tracked
        # each block: RMSNorm -> Attention(+residual) -> RMSNorm -> FFN(+residuals)
        self.layers = nn.ModuleList([
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
        ])

        # final normalization
        # one last RMSNorm before the output head
        # the output of the last transformer block is raw (unnormalized)
        # we normalize before projecting to vocabulary so the LM head gets clean, well-scaled inputs
        self.final_norm = RMSNorm(d_model=config.d_model)

        # LM HEAD (output projection)
        # linear projection: d_model -> vocab size
        # transofrms each token's 768 dim 'understanding' into a
        # 50257 dim vector of scores - one score per possible next token
        # logits[b,t,v] = 'score for token v being the next word after position t in batch b'

        self.lm_head = nn.Linear(in_features=config.d_model, out_features=config.vocab_size, bias=False)

        # weight tying
        # share weight matrix between embedding and LM head
        # the embdedding maps token -> vector. The LM head maps vector->token.
        # these are inverse operations
        # sharing weights has 3 benifits
        #  1. parameter efficiency: saves 50257 x 768 = 38.6M params
        #   (30% of the total for GPT-2 small)
        #  2. better regularization: the shared matrix gets 
        #      gradient signals from both directions, improving the quality of token representations
        # 3. theoretical elegance. input and output tokens live in the same semantic space
        self.token_embedding.weight = self.lm_head.weight

        # weight initialization
        # initialize all weights with normal(0, 0.02)
        # starting from the right distribution is criticla
        # too small -> gradients vanish, model never learns
        # too large -> activations saturate, gradients explode
        # 0.02 std gives values mostly in [-0.04, 0.04],
        # which is sweet spot for Transformers
        self.apply(self._init_weights)
        print(f'GPT Initialized with {self.get_num_params():, } parameters')

    def _init_weights(self, module: nn.Module):
        """Initialize weights using the GPT-2 scheme"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self) -> int:
        """Count total trainable parameters (weights + biases)."""
        return sum(p.numel() for p in self.parameters())
    
    def create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.view(1,1,seq_len,seq_len)
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None) -> tuple:
        """
        Process a batch of token sequences through the GPT model
        Args:
            input_ids: [batch_size, seq_len] - token ids for each sequence
            targets: [batch_size, seq_len] - same tokens used for loss
                                (the model predicts input_ids[t+1] from input_ids[t])

        Returns:
            logits: [batch_size, seq_len, vocab_size] - raw prediction scores
            loss: scalar - cross entropy (None if targets not provided)

        The shift-by-one trick:
            input:  [The, cat, sat, on, the, mat]
                      |    |    |    |   |    |
            target: [cat, sat, on,  the, mat, ?]
            predict P(cat|the) P(sat|the, cat) .... P(mat|the,cat,sat,on,the)
        The dataset already provides shifted targets so we compute loss on all positions
        """

        batch_size, seq_len = input_ids.shape

        # embed tokens
        # input: [batch, seq] token ids
        # output: [batch, seq, d_mdoel] continuous vectors
        x = self.token_embedding(input_ids)
        x = self.embd_dropout(x)

        # create a causal mask
        # lower traiangular mask: token i can only see tokens 0...i
        # without this, the model would 'cheat' by looking at future tokens 
        # when predicting the next one
        mask = self.create_causal_mask(seq_len, seq_len, input_ids.device)

        # transformer layers
        # pass through all N transformer blocks sequentially
        # each layer refined the representations. early layers
        # capture syntax. later layers capture semantics

        for layer in self.layers:
            x = layer(x, mask)
        # final normalization
        x = self.final_norm(x)

        # project to vocabulary
        # convert from d_model-dim 'understanding' to vocab_size-dim scores
        # each position gets a score for every possible next token
        # example: logits[0, 3, 2603] = 9.2 means:
        # for batch 0, position 3, the score for token 2603 ('mat') is 9.2
        # higher score = model thinks this token is more likely
        logits = self.lm_head(x) # [batch, seq_len, vocab_size]

        # compute loss (training only)

        loss  = None
        if targets is not None:
            # WHAT: Align predictions with targets using shift-by-one
            #
            # logits[:, :-1, :]:  predictions for positions 0..seq-2
            # targets[:, 1:]:      true tokens for positions 1..seq-1
            #
            #          Position:  0      1      2      3
            #          Input:     The    cat    sat    on
            #          Target:    cat    sat    on     the
            #          Logits:   P(cat) P(sat) P(on)  P(the)
            #                                        ^
            #                                   We drop this
            #                                   (no target for it)
            logits_flat = logits.contiguous().view(
                -1, self.config.vocab_size
            )
            targets_flat = targets.contiguous().view(
                -1
            )

            loss = F.cross_entropy(logits_flat, targets_flat)


        return logits, loss
    
