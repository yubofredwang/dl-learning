import torch
from torch import nn
from torch.nn import functional as F

class DecoderModel(nn.Module):

    """
    This class is the Decoder model improved based on the Bigram model we have
    before with multihead attention and transformer block.
    """
    
    def __init__(self, vocab_size, n_embed=32, block_size=8):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # 32 x 65


    def forward(self, idx, targets: torch.Tensor = None):
        _, T = idx.shape # C = n_embed, embedding dimensionality
        token_embeddings: torch.Tensor = self.token_embedding_table(idx) # B, T, C
        pos_embeddings = self.positional_embedding_table(torch.arange(T))   # T, C
        x = token_embeddings + pos_embeddings # B, T, C
        logits = self.lm_head(x) # B, T, Vocab Size
        if targets is None:
            loss = None
        else:
            # B=batch size, T=sequence length(block size), C(Channel)=embedding dimensionality
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Calculates loss
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, _ = self(idx) #self(idx) goes to forward function
            # focus on the last time step, -1 means last one in the sequence
            logits = logits[:, -1, :]
            # Apply softmax, softmax function is often used in neural networks to convert logits into probabilities.
            probs = F.softmax(logits, dim=-1) # B, C
            # sample from the distribution, could be replaced with argmax?
            idx_next  = torch.multinomial(probs, num_samples=1) # B, 1 each batch dimension we have single prediction
            # append sampled index to the running sequence and continue
            idx = torch.cat([idx, idx_next], dim=1) # B, T + 1 (next in sequence)
        return idx