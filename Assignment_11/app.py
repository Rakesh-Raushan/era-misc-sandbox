import gradio as gr
from bpe_tokenizer import SanskritBPETokenizer
import json

# Load the trained tokenizer
with open('tokenizer_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

tokenizer = SanskritBPETokenizer(vocab_size=4999)
tokenizer.vocab = config['vocab']
# Convert string keys back to tuples
tokenizer.merges = {tuple(k.split('|')): v for k, v in config['merges'].items()}
tokenizer.inverse_vocab = config['inverse_vocab']

# Load example texts
with open('examples/example1.txt', 'r', encoding='utf-8') as f:
    example1 = f.read().strip()
    
with open('examples/example2.txt', 'r', encoding='utf-8') as f:
    example2 = f.read().strip()

def process_text(text):
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    return f"Encoded tokens: {tokens}\nDecoded text: {decoded}"

# Create Gradio interface
iface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(label="Enter Sanskrit Text"),
    outputs=gr.Textbox(label="Tokenization Result"),
    title="Sanskrit BPE Tokenizer",
    description="Enter Sanskrit text to see its tokenization",
    examples=[
        [example1],
        [example2]
    ]
)

iface.launch() 