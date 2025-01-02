import numpy as np
import torch
import os
import pickle

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataLoader:

    """
    DataLoader class to load the data from the bin files.
    """

    def __init__(self, data_dir: str) -> None:
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        self.train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        self.eval_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        # Loads the meta data
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            self.stoi = meta['stoi']
            self.itos = meta['itos']
            self.vocab_size = meta['vocab_size']

    def get_train_data(self, batch_size=1, block_size=1):
        ix = torch.randint(len(self.train_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((self.train_data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.train_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return x, y
    
    def get_eval_data(self, batch_size=1, block_size=1):
        ix = torch.randint(len(self.eval_data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((self.eval_data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.eval_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        return x, y

    def encode(self, s):
        return [self.stoi[c] for c in s] # encoder: take a string, output a list of integers

    def decode(self, l):
        return ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string