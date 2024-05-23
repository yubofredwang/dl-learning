import torch
from torch import nn
from torch.nn import functional as F

class BigramLangugageModel(nn.Module):

    """
    BigramLangugageModel class to create a NN model that has one layer of embeddings.
    
    Embedding Layer:
    An embedding layer in deep learning, particularly in the context of natural language processing (NLP), 
    is used to convert categorical data, such as words, into dense, continuous-valued vectors. 
    
    Use of Embedding Layers:
    1. Dimensionality Reduction: Embedding layers transform high-dimensional categorical data 
    (e.g., one-hot encoded vectors) into low-dimensional dense vectors. This reduces the computational cost 
    and memory usage.

    2. Semantic Representation: The embeddings capture semantic relationships between categories. For instance, 
    in word embeddings, similar words or words with similar contexts are mapped to vectors that are close to 
    each other in the embedding space.

    3. Input Representation: Embedding layers provide a fixed-size vector representation for each category, 
    which is suitable for input to neural networks.

    How an Embedding Layer Works
    1. Lookup Table: The embedding layer maintains a lookup table, where each unique category 
    (e.g., word) is assigned a unique vector of fixed size.

    2. Training: During training, the embeddings are learned and updated through backpropagation. 
    The objective is to adjust the vectors such that they capture useful semantic information for the task at hand.


    """
    
    def __init__(self, vocab_size):
        super().__init__()
        # num_embeddings (int): size of the dictionary of embeddings
        # embedding_dim (int): the size of each embedding vector
        # Here is basically means a word can be linked to another word.
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets: torch.Tensor = None):
        # idx and targets are both (B, T) tensor of integers
        logits: torch.Tensor = self.embeddings(idx)
        if targets is None:
            loss = None
        else:
            # B=batch size, T=sequence length(block size), C(Channel)=embedding dimensionality
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Cross entropy loss expects the input to be of shape (minibatch, C)
            # it calculates loss between input logits and target.
            # It is useful when training a classification problem with C classes. 
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