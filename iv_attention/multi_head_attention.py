import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from iii_position_encoding.rope import RotaryPositionEmbedding

class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention with RoPE and causal masking.

    Transformers would be useless without attention. This is the
    mechanism that lets each token 'look at' every other token and 
    decide how much each matters for understanding the current context.

    Each attention head:
        1. Projects input into Query, Key, Value spaces
        2. Computes Q.K^T / sqrt(d_k) -> how well each query matches each key
        3. Applies causal mask -> no peeking at future tokens
        4. Softmax -> converts scores to a probability distribution
        5. Weighted sum of values -> builds context aware representation.

        Doing this with multiple heads in parallel lets each head specialize
        in different linguistic patterns

    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        d_model: Totall embedding dimension (e.g. 768 for GPT-2 small)
        num_heads: Number of parallel attention heads (e.g 12)
        dropout: Probability of randomly zeroing attention weights

        d_model must be divisible by num_heads because each head operates
        on d_model/num_heads dimensions (64 for GPT-2 small).
        This split-then-concat starategy lets heads specialize while keeping
        total parameter count the same as single large head.
        """
        super().__init__()

        # validate that heads evenly divide the model dimension
        assert d_model % num_heads == 0, (
            f'd_model ({d_model}) must be divisible by num_heads ({num_heads}).'
            f'This ensures each head has an equal dimension.'
        )

        self.d_model = d_model
        self.num_heads = num_heads
        # 768/12 = 64 dimensions per head
        # 64 is the sweet spot enough to capture meaning, smal enough for efficient compute
        self.head_dim = d_model // num_heads

        """
        QKV Projection

        one big linear layer that projects input Q, K, V simultaneously
        why: 3 seperate linear (768->768) layers = 3 matrix multiplies
        one combined linear (768->2304) = 1 bigger matrix multiply.
        on GPU, 1 big operation is much faster than 3 small ones
        due to better parallelism and fewer kernel launches
        shape: [d_model, 3*d_model] = [768, 2304]
        """

        self.qkv_proj = nn.Linear(in_features=d_model, out_features=3*d_model, bias=False)

        """
        Output Projection
        Project concateneated head outputs back to d_model
        why: after concatenation: [batch, seq_len, d_model] but each head's output
        was computed independently. this linear layer mixes information across heads, letting
        them communicate. without it, heads would stay isolated - like 12 experts
        who never talk to each other.
        """
        self.out_proj = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        """
        RoPE (Rotary Position Embeddings)
        Applies rotation based position encoding to Q and K only
        Why: RoPE encodes position into Q and K vectors so that the 
        dot product Q.K naturally depends on RELATIVE position.
        we apply to the head_dim (d_model) because each head needs its own
        position info in its subspace. V does not get RoPE because values carry content
        but not position. position ony relevant for deciding which values to attend to,
        not the values themselves
        """
        self.rotary = RotaryPositionEmbedding(d_model=self.head_dim)

        """
        Dropout
        Randomly zero out attention weights during training.
        whithout dropout, the model can become overconfident.
        one token always dominates attention, ignoring other potentially
        useful context. Dropout forces the model to learn
        redundant attention patterns (backup plans)
        """

        self.attn_dropout = nn.Dropout(dropout) # applied to attention weights
        self.resid_dropout = nn.Dropout(dropout) # applied to final output

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        compute multi head self attention
        input: x [batch, seq_len, d_model] -> token embeddings
        mask: [batch, 1, seq, seq] -> causal mask (1=visible, 0=masked)
        output = [batch, seq_len, d_model] - conetxt aware representations
        the forward pass has 8 steps, each critical;
        """

        batch_size, seq_len, _ = x.shape
        """
        step1: project input to Q, K, V - all at once
        linearly transform input into query, key and value spaces
        combined projection is faster on GPU than 3 seperate ones
        after this: [batch, seq, 3*d_model] where the last dim has Q values first,
        then K values and then V values
        """

        qkv = self.qkv_proj(x) # [batch, seq, 3*d_model]

        """
        step2: reshape to expose the head dimension
        split 3*d_model into seperate Q,K,V and seperate heads
        we need shape [batch, num_heads, seq, head_dim] for
        parallel computation. The reshape + permute does this in
        two efficient operations without data copies
        transform: [batch, seq, 3, heads, head_dim]
        then permute: [3, batch, heads, seq, head_dim]
        """

        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, batch, heads, seq, head_dim]

        # unpack 3 projections
        q = qkv[0] # Query:  [batch, heads, seq, head_dim] — "what I'm looking for"
        k = qkv[1] # Key:    [batch, heads, seq, head_dim] — "what I offer to match"
        v = qkv[2] # Value:  [batch, heads, seq, head_dim] — "my actual content"

        """
        step3: apply rotary position embeddings
        rotate Q and K by position-dependent angles
        after rotation, the dot product q_i.k_j depends on
        cos(i-j) and sin(i-j) -- the relative distance between tokens
        i and j. this is what we want: attention should care about
        'how far apart are these tokens?' not 'what are their absolute positions?'
        """
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)

        """
        step4: compute attention scores (Q.K^T)
        for each query token, compute dot product with every key token
        dot product measures cosine similarity (if vectors normalized)
        higher dot product = query 'wants' what key 'offers'
        shape: [batch, heads, query_seq, key_seq]
        attn_scores[b,h,i,j]=how much token i attends to token j

        DIVIDE by sqrt(head_dim): critical for stable training.
        Without this, the variance of dot products grows with d_k,
        making softmax too "peaky" → gradients vanish → model dies.
        See Part 4 above for the mathematical derivation.
        """
        attn_scores = (q @  k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        """
        step5: Apply causal mask — no peeking at future tokens
        Set attention scores to future tokens to -infinity
        During training, the model must predict token[i+1] from
        tokens[0..i]. If token[i] can see token[i+1], it's like
        seeing the answer before the question — cheating.
        -infinity → e^(-inf) = 0.0 after softmax = zero attention
        The mask is lower-triangular:
        Token 0 → sees [0]        (itself only)
        Token 1 → sees [0, 1]     (itself + previous)
        Token 2 → sees [0, 1, 2]  (itself + all previous)
        Token 3 → sees [0, 1, 2, 3]
        """

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        """
        Step6: Softmax — scores become attention weights 
        Convert raw scores to a probability distribution over keys
        softmax(scores)[j] = e^score[j] / sum(e^score[k] for k in all keys)
        This makes all weights:
        - Positive (e^x > 0 always)
        - Sum to 1.0 (proper probability distribution)
        - Differentiable (we can compute gradients through it)
        The softmax is applied over the LAST dimension (dim=-1),
        which is the "key" dimension — so each query gets a
        distribution over all keys it can see.
        """
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        """
        step7: Weighted sum of values
        Mix the value vectors according to attention weights
        This is WHERE attention actually happens. Each query
        token gets a NEW vector that is a weighted blend of
        all visible value vectors.
        
        High attention to token j → V_j has large influence
        Low attention to token j → V_j has small influence
        
        The result is "context-aware" — each token now "knows"
        about the other relevant tokens in the sequence.
        
        [batch, heads, seq, head_dim] @ [batch, heads, seq, head_dim]
        → [batch, heads, seq, head_dim]
        """
        attn_output = attn_weights @ v

        """
        step8: Merge heads and project
        Combine all head outputs into one d_model vector per token
        Currently: [batch, heads, seq, head_dim]
        Need:       [batch, seq, d_model]
        Transpose swaps heads and sequence:
        [batch, seq, heads, head_dim]
        Reshape flattens heads x head_dim:
        [batch, seq, d_model]
        
        The final linear projection lets information flow between
        heads — each head's discoveries can now influence the
        combined representation.
        """
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output) # mix across heads
        output = self.resid_dropout(output) # regularization
        return output
    
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    WHAT: Create a causal (lower triangular) attention mask.
    WHY:  Prevents tokens from attending to future tokens during training.

    Visual for seq_len=6:
        [[✓, ✗, ✗, ✗, ✗, ✗],     Token 0 (first word)
            [✓, ✓, ✗, ✗, ✗, ✗],     Token 1
            [✓, ✓, ✓, ✗, ✗, ✗],     Token 2
            [✓, ✓, ✓, ✓, ✗, ✗],     Token 3
            [✓, ✓, ✓, ✓, ✓, ✗],     Token 4
            [✓, ✓, ✓, ✓, ✓, ✓]]     Token 5 (last word — sees everything)

    ✓ = position is visible (1.0)
    ✗ = position is masked (0.0, becomes -inf in attention)

    Reshaped to [1, 1, seq_len, seq_len] for broadcasting over:
    - batch dimension (all batches use same mask)
    - head dimension (all heads use same mask — heads CAN'T see future)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)