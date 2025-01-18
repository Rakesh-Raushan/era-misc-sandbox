# script to train the transformer model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import numpy as np
import os, time
import argparse
from tqdm import tqdm
from transformer import GPT, GPTConfig
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# STOP
num_return_sequences = 5
max_length = 30

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('./input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


model = GPT(GPTConfig())
model.to(device)

def train(config, model, optimizer, scheduler, train_loader, val_loader):
    # Training Loop
    steps_per_epoch = len(train_loader.tokens) // (batches * no_of_tokens)
    print(steps_per_epoch)
    EPOCHS = 50

    # Use summary writer for tensorboard
    writer = SummaryWriter(log_dir=os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S")))

    for epoch in range(EPOCHS):
        loss_list = []
        train_loader_temp = train_loader
        start_time = time.time()

        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="step") as pbar:
            for step in range(steps_per_epoch):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                writer.add_scalar("Loss/train", loss.item(), global_step=epoch * steps_per_epoch + step)
                writer.add_scalar("Perplexity/train", np.exp(loss.item()), global_step=epoch * steps_per_epoch + step)


        scheduler.step()
        epoch_loss = sum(loss_list) / len(loss_list)
        writer.add_scalar("Loss/train_epoch", epoch_loss, global_step=epoch)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.2f}s")
        if(epoch_loss < 0.099999):
            break
    
    writer.close()



batches, no_of_tokens = 16, 256
train_loader = DataLoaderLite(B=batches, T=no_of_tokens)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

if __name__ == "__main__":
    train(GPTConfig(), model, optimizer, scheduler, train_loader, None)
    torch.save(model.state_dict(), "model.pth")