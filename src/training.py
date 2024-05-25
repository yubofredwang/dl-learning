import torch
torch.manual_seed(42)

from data_utils import DataLoader
from config import ModelConfig, TrainingConfig
from decoder_model import DecoderModel


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataset: DataLoader, config: ModelConfig):
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
    train_x, train_y = dataset.get_train_data(batch_size=32, block_size=config.block_size)
    out['train'] = calculate_average_loss(model, train_x, train_y)
    eval_x, eval_y = dataset.get_eval_data(batch_size=32, block_size=config.block_size)
    out['eval'] = calculate_average_loss(model, eval_x, eval_y)
    model.train()
    return out


def train(data: DataLoader, model_config: ModelConfig, training_config: TrainingConfig) -> torch.nn.Module:    
    # Initialize the model
    model = DecoderModel(model_config)
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    # Start training loop
    print("Training the model...")
    for iter in range(training_config.n_iters):
        if iter % training_config.eval_interval == 0:
            losses = estimate_loss(model, data, model_config)
            print(f"Iter: {iter}, Train loss: {losses['train']}, Eval loss: {losses['eval']}")
        
        batch_input, batch_target = data.get_train_data(training_config.batch_size, model_config.block_size)
        _, loss = model(idx=batch_input, targets=batch_target)
        # Backward, set models.embeddings.weight.grad
        loss.backward()
        # Optimizer update the weights
        optimizer.step()
        # Clear the gradients for next calculation
        optimizer.zero_grad(set_to_none=True)

    print("Final Loss: ", loss.item())
    return model


if __name__ == "__main__":
    # Get data loader
    data_loader = DataLoader("./shakespeare_char")
    # TODO: Should load a config from a file
    # Model config
    model_config=ModelConfig(
        n_embed=32,
        n_heads=4,
        block_size=16,
        vocab_size=data_loader.vocab_size,
        dropout=0.0,
        attention_blocks=3,
    )
    # Traning Config
    training_config=TrainingConfig(
        batch_size=32,
        eval_interval=1000,
        n_iters=10000,
        max_new_tokens=100,
        learning_rate=1e-3,
    )
    # Start training!
    model = train(data_loader, model_config, training_config)
    # Generate text
    # First character is the new line B=1, T=1, 0 is new line character
    idx = torch.zeros((1, 1), dtype=torch.int64)
    # 0 to unblock the batch dimension, use the first one in the batch
    print(data_loader.decode(model.generate(idx, max_new_tokens=100)[0].tolist()))


