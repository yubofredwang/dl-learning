import torch
torch.manual_seed(42)

from data import DataLoader
from bigram_model import BigramLangugageModel
from decoder_model import DecoderModel

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataset: DataLoader, eval_iters=200):
    out = {}
    # Change to eval mode
    model.eval()
    def calculate_average_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, eval_iters=200):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # Forward pass
            _, loss = model(x, y)
            losses[k] = loss.item()
        return losses.mean()
    train_x, train_y = dataset.get_train_data(batch_size=32, block_size=8)
    out['train'] = calculate_average_loss(model, train_x, train_y)
    eval_x, eval_y = dataset.get_eval_data(batch_size=32, block_size=8)
    out['eval'] = calculate_average_loss(model, eval_x, eval_y)
    model.train()
    return out


def train():
    # Create data loader
    data_loader = DataLoader("shakespeare_char")
    # Initialize the model
    model = DecoderModel(data_loader.vocab_size)
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eval_interval = 1000
    # Start training loop
    print("Training the model...")
    for iter in range(10000):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, data_loader)
            print(f"Iter: {iter}, Train loss: {losses['train']}, Eval loss: {losses['eval']}")
        
        batch_input, batch_target = data_loader.get_train_data(batch_size=32, block_size=8)
        logits, loss = model(idx=batch_input, targets=batch_target)
        # Backward, set models.embeddings.weight.grad
        loss.backward()
        # Optimizer update the weights
        optimizer.step()
        # Clear the gradients for next calculation
        optimizer.zero_grad(set_to_none=True)

    print("Loss: ", loss.item())
    # Generate text
    # First character is the new line B=1, T=1, 0 is new line character
    idx = torch.zeros((1, 1), dtype=torch.int64)
    # 0 to unblock the batch dimension, use the first one in the batch
    print(data_loader.decode(model.generate(idx, max_new_tokens=100)[0].tolist()))


if __name__ == "__main__":
    train()

