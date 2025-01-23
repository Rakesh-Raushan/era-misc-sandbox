import gradio as gr
import torch
from model import create_model_from_config
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import json
import os

def load_model_and_tokenizer():
    # Load config
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": [
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<repo_name>", "<reponame>", "<file_sep>", "<filename>",
            "<gh_stars>", "<issue_start>", "<issue_comment>",
            "<issue_closed>", "<jupyter_start>", "<jupyter_text>",
            "<jupyter_code>", "<jupyter_output>", "<jupyter_script>",
            "<empty_output>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Update config vocab size to match tokenizer
    config['vocab_size'] = len(tokenizer)
    
    # Create model
    model = create_model_from_config(config)
    
    # Download checkpoint from HuggingFace Hub
    checkpoint_path = hf_hub_download(
        repo_id="Perpetualquest/smolLm2checkpoint",
        filename="checkpoint_5050.pt",
        repo_type="model"
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer

# Load model and tokenizer once at startup
model, tokenizer = load_model_and_tokenizer()

def generate_text(prompt, max_length=100, temperature=0.7):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    # Generate text token by token
    for _ in range(max_length - input_ids.shape[1]):
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
        # Append next token
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stop if we generate EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=500, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Implementation Demo",
    description="Enter a prompt to generate text using the SmolLM2-inspired model."
)

if __name__ == "__main__":
    iface.launch() 