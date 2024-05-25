import torch
from torch import nn
from torch.nn import functional as F

from config import ModelConfig


class DecoderModel(nn.Module):
    """
    This class is the Decoder model improved based on the Bigram model we have
    before with multihead attention and transformer block.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        n_embed = config.n_embed
        block_size = config.block_size
        attention_blocks = config.attention_blocks
        vocab_size = config.vocab_size
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(attention_blocks)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets: torch.Tensor = None):
        _, T = idx.shape  # C = n_embed, embedding dimensionality
        token_embeddings: torch.Tensor = self.token_embedding_table(idx)  # B, T, C
        pos_embeddings = self.positional_embedding_table(torch.arange(T))  # T, C
        x = token_embeddings + pos_embeddings  # B, T, C
        x = self.blocks(x)  # B, T, C head before the linear layer
        x = self.layer_norm(x)
        logits = self.lm_head(x)  # B, T, Vocab Size)
        if targets is None:
            loss = None
        else:
            # B=batch size, T=sequence length(block size), C(Channel)=embedding dimensionality
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Calculates loss
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size=8):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context
            idx_context = idx[:, -block_size:]
            logits, _ = self(idx_context)  # self(idx) goes to forward function
            # focus on the last time step, -1 means last one in the sequence
            logits = logits[:, -1, :]
            # Apply softmax, softmax function is often used in neural networks to convert logits into probabilities.
            probs = F.softmax(logits, dim=-1)  # B, C
            # sample from the distribution, could be replaced with argmax?
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # B, 1 each batch dimension we have single prediction
            # append sampled index to the running sequence and continue
            idx = torch.cat([idx, idx_next], dim=1)  # B, T + 1 (next in sequence)
        return idx


class Block(nn.Module):
    "Transformer block: communication then computation"

    def __init__(self, config: ModelConfig):
        super().__init__()
        n_embed = config.n_embed
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.heads = MultiHeadAttention(config)
        self.layer_norm2 = nn.LayerNorm(n_embed)
        self.ff = FeedForwardLayer(config)

    def forward(self, x):
        # Residual connection
        x = x + self.heads(self.layer_norm1(x))
        x = x + self.ff(self.layer_norm2(x))
        return x


class FeedForwardLayer(nn.Module):
    """
    Feed forward layer: two linear layers with a ReLU in between
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        n_embed = config.n_embed
        dropout = config.dropout
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class MultiHeadAttention(nn.Module):
    """
    Multihead attention module: divide d_model into n_heads and run them in parallel
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        n_embed = config.n_embed
        n_heads = config.n_heads
        dropout = config.dropout
        
        # List of modules run in parallel
        self.heads = nn.ModuleList(
            [Head(config) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_res = [head(x) for head in self.heads]
        # Concatenate the results
        res = torch.cat(head_res, dim=-1)
        res = self.projection(res)
        return res


class Head(nn.Module):
    "Single head of self attention"

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Do not learn bias
        # d_model = n_embed, d_k, dv = head_size
        n_embed = config.n_embed
        head_size = n_embed // config.n_heads
        block_size = config.block_size

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # This is not a parameter! It is just a buffer to mask. paramters will be optimized
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, idx: torch.Tensor):
        B, T, C = idx.shape
        k = self.key(idx)
        q = self.query(idx)
        # Q * K^T / sqrt(d_k), scaling factor
        weights = q @ k.transpose(-2, -1) / (C**0.5)  # B, T, T
        # masked fill, so the upper values are set to -inf, lower parts are saved
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # softmax over the last dimension
        weights = F.softmax(weights, dim=-1)
        # weighted aggregation
        v = self.value(idx)
        res = weights @ v
        return res
