import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import wandb
from model import create_model_from_config
import json
import time

# Only enable tensor cores if CUDA is available
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

def get_device():
    """Get the appropriate device (CUDA or CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_checkpoint(model, optimizer, scaler, step, loss, checkpoint_dir, config):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'step': step,
        'loss': loss,
        'config': config  # Save the config used during training
    }
    path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pt')
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, scaler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)  # Add weights_only=True to avoid the warning
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['step'], checkpoint['loss'], checkpoint.get('config', None)

def create_tokenizer():
    """Create tokenizer with SmolLM2 specific configuration"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add special tokens from SmolLM2 config
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<repo_name>",
            "<reponame>",
            "<file_sep>",
            "<filename>",
            "<gh_stars>",
            "<issue_start>",
            "<issue_comment>",
            "<issue_closed>",
            "<jupyter_start>",
            "<jupyter_text>",
            "<jupyter_code>",
            "<jupyter_output>",
            "<jupyter_script>",
            "<empty_output>"
        ]
    }
    
    # First add the special tokens
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added_tokens} special tokens")
    print(f"Vocabulary size after adding special tokens: {len(tokenizer)}")
    
    # Explicitly set pad_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens with their specific attributes
    added_tokens_decoder = {
        "0": {"content": "<|endoftext|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "1": {"content": "<|im_start|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "2": {"content": "<|im_end|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "3": {"content": "<repo_name>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "4": {"content": "<reponame>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "5": {"content": "<file_sep>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "6": {"content": "<filename>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "7": {"content": "<gh_stars>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "8": {"content": "<issue_start>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "9": {"content": "<issue_comment>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "10": {"content": "<issue_closed>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "11": {"content": "<jupyter_start>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "12": {"content": "<jupyter_text>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "13": {"content": "<jupyter_code>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "14": {"content": "<jupyter_output>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "15": {"content": "<jupyter_script>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "16": {"content": "<empty_output>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True}
    }
    
    # Add tokens with their specific configurations
    for _, token_info in added_tokens_decoder.items():
        tokenizer.add_special_tokens({'additional_special_tokens': [token_info['content']]})
        # Get the token id and set its properties
        token_id = tokenizer.convert_tokens_to_ids(token_info['content'])
        if hasattr(tokenizer, 'special_tokens_map_extended'):
            tokenizer.special_tokens_map_extended[token_id] = token_info
    
    # Set other tokenizer attributes
    tokenizer.model_max_length = 8192
    tokenizer.clean_up_tokenization_spaces = False
    tokenizer.add_prefix_space = False
    tokenizer.padding_side = "right"
    
    return tokenizer

def train(config_path, total_steps=5000, checkpoint_every=500, checkpoint_dir='checkpoints', start_step=0, project_name = "smollm2-training-5000-epochs"):
    # Initialize wandb
    wandb.init(project=project_name)
    
    # Load config and create model
    config = load_config(config_path)
    device = get_device()
    
    # Initialize tokenizer first to get the final vocab size
    tokenizer = create_tokenizer()
    
    # Update config with new vocab size
    config['vocab_size'] = len(tokenizer)
    print(f"Updated model vocabulary size to match tokenizer: {config['vocab_size']}")
    
    # Create model with updated config
    model = create_model_from_config(config)
    model = model.to(device)
    
    # Enable torch.compile only if available
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        model = torch.compile(model)
    
    # Print vocabulary sizes for debugging
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Model vocabulary size: {config['vocab_size']}")
    
    # Load dataset
    dataset = load_dataset("semran1/cosmopedia-v2-subset", streaming=True)
    train_data = dataset['train'].shuffle(seed=42)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    step = start_step
    total_tokens = 0
    start_time = time.time()
    batch_start_time = time.time()
    
    for batch in tqdm(train_data, total=total_steps-start_step):
        if step >= total_steps:
            break
            
        # Tokenize batch
        inputs = tokenizer(batch['text'], truncation=True, max_length=512, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        # Clamp input IDs to be within vocab size
        input_ids = torch.clamp(input_ids, max=config['vocab_size']-1)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Debug information for first batch
        if step == start_step:
            print(f"Input IDs max value: {input_ids.max().item()}")
            print(f"Input IDs min value: {input_ids.min().item()}")
            print(f"Input IDs shape: {input_ids.shape}")
        
        # Count tokens in this batch (excluding padding)
        num_tokens = attention_mask.sum().item()
        total_tokens += num_tokens
        
        # Forward pass with autocast only when using CUDA
        if torch.cuda.is_available():
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, config['vocab_size']),
                    shift_labels.view(-1)
                )
        else:
            outputs = model(input_ids, attention_mask=attention_mask)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, config['vocab_size']),
                shift_labels.view(-1)
            )
        
        # Backward pass with gradient scaling if CUDA is available
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Synchronize CUDA for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Calculate timing metrics
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        total_time = batch_end_time - start_time
        
        # Calculate tokens per second
        tokens_per_sec = num_tokens / batch_time
        avg_tokens_per_sec = total_tokens / total_time
        
        # Log metrics
        wandb.log({
            "loss": loss.item(),
            "step": step,
            "batch_time": batch_time,
            "tokens_per_sec": tokens_per_sec,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "total_tokens": total_tokens
        })
        
        # Print timing info
        print(f"\rStep {step}: {batch_time:.2f}s/batch, {tokens_per_sec:.2f} tokens/s (avg: {avg_tokens_per_sec:.2f})", end="")
        
        # Save checkpoint and generate sample text
        if (step + 1) % checkpoint_every == 0:
            save_checkpoint(model, optimizer, scaler, step, loss.item(), checkpoint_dir, config)
            
            # Generate sample text
            with torch.no_grad():
                sample_input = tokenizer.encode("Once upon a time", return_tensors="pt").to(device)
                sample_output = model(sample_input)
                sample_tokens = torch.argmax(sample_output[0], dim=-1)
                sample_text = tokenizer.decode(sample_tokens)
                print(f"\nSample generation at step {step}:")
                print(sample_text)
        
        step += 1
        batch_start_time = time.time()  # Reset batch timer
    
    # Final checkpoint
    save_checkpoint(model, optimizer, scaler, step, loss.item(), checkpoint_dir, config)
    
    # Print final statistics
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s")
    print(f"Average tokens/s: {total_tokens/total_time:.2f}")
    print(f"Total tokens processed: {total_tokens}")
    
    wandb.finish()

def continue_training(checkpoint_path, config_path, additional_steps=50):
    # Load checkpoint first to get the config
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    saved_config = checkpoint.get('config', load_config(config_path))
    
    device = get_device()
    
    # Initialize tokenizer
    tokenizer = create_tokenizer()
    
    # Update saved_config with tokenizer's vocab size
    saved_config['vocab_size'] = len(tokenizer)
    print(f"Updated saved config vocabulary size to match tokenizer: {saved_config['vocab_size']}")
    
    # Create model with the updated saved config
    model = create_model_from_config(saved_config)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Now load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_step = checkpoint['step']
    last_loss = checkpoint['loss']
    
    # Enable torch.compile only if available
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        model = torch.compile(model)
    
    # Continue training with the saved config
    train(config_path, total_steps=start_step + additional_steps, checkpoint_every=10, start_step=start_step, project_name = "smollm2-training-50-more-epochs-from-checkpoint")

if __name__ == "__main__":
    train("config.json")
    # To continue training:
    print("sleeping for 20 secs . . .After I wake up I'll resume training from the last checkpoint ")
    time.sleep(20)
    continue_training("checkpoints/checkpoint_5000.pt", "config.json") 