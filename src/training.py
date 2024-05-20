import torch
torch.manual_seed(42)

from data_util import DataLoader
from bigram_model import BigramLangugageModel

def train():
    print("Training the model...")
    data_loader = DataLoader("shakespeare_char")
    batch_input, batch_target = data_loader.get_train_data(batch_size=32, block_size=8)
    model = BigramLangugageModel(data_loader.vocab_size)
    logits, loss = model(idx=batch_input, targets=batch_target)
    print("Loss: ", loss.item())





if __name__ == "__main__":
    train()

