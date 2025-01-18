import gradio as gr
import torch
import tiktoken
from transformer import GPT, GPTConfig

# Load the model
def load_model():
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    return model

# Text generation function
def generate_text(prompt, max_tokens=500, temperature=0.8, top_k=40):
    print(f"inputs: {prompt=}, {max_tokens=}, {temperature=}, {top_k=}")
    # Encode the prompt
    enc = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    
    # Generate
    model = load_model()
    with torch.no_grad():
        output_sequence = []
        for _ in range(max_tokens):
            # Get predictions
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to output
            output_sequence.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we generate a newline
            if next_token.item() == enc.encode('\n')[0]:
                break
    
    # Decode and return the generated text
    generated_text = enc.decode(output_sequence)
    return prompt + generated_text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=1, maximum=5000, value=1000, step=1, label="Max-Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=1000, value=40, step=1, label="Top-K")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="GPT Text Generator",
    description="Enter a prompt and generate text using a custom-trained GPT model."
)

if __name__ == "__main__":
    iface.launch() 