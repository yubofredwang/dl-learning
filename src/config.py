from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model config for the model."""
    n_embed: int
    n_heads: int
    block_size: int
    vocab_size: int
    dropout: float
    attention_blocks: int
    
    def __post_init__(self):
        assert self.n_embed % self.n_heads == 0, "n_embed should be divisible by n_heads"
        
        
@dataclass
class TrainingConfig:
    """Training config for the model."""
    batch_size: int
    eval_interval: int
    n_iters: int
    max_new_tokens: int
    learning_rate: float