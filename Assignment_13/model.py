import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Create causal mask
            causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=hidden_states.device))
            # Combine with attention mask
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            causal_mask = causal_mask.view(1, 1, seq_length, seq_length)
            mask = attention_mask * causal_mask
            # Convert to additive mask
            mask = (1.0 - mask) * torch.finfo(hidden_states.dtype).min
            mask = mask.expand(batch_size, self.num_attention_heads, seq_length, seq_length)
        
        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        key_states = key_states.transpose(1, 2)      # (batch, n_kv_heads, seq_len, head_dim)
        value_states = value_states.transpose(1, 2)  # (batch, n_kv_heads, seq_len, head_dim)
        
        # Repeat k/v heads for multi-query attention
        key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_key_value_groups)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_key_value_groups)
        
        # Apply attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, config["rms_norm_eps"])
        self.post_attention_layernorm = LlamaRMSNorm(self.hidden_size, config["rms_norm_eps"])

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], self.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config["num_hidden_layers"])])
        self.norm = LlamaRMSNorm(self.hidden_size, config["rms_norm_eps"])
        self.rotary_emb = LlamaRotaryEmbedding(
            self.hidden_size // config["num_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"]
        )

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaConfig(PretrainedConfig):
    model_type = "llama"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class LlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Tie weights if specified in config
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

# Example usage:
def create_model_from_config(config_dict):
    """Create model from config dictionary"""
    config = LlamaConfig(**config_dict)
    model = LlamaForCausalLM(config)
    return model