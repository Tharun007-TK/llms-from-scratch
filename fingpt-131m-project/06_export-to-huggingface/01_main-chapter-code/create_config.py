from transformers import AutoConfig
import json
import os

config = {
    "model_type": "fingpt",
    "architectures": ["FinGPTForCausalLM"],

    "vocab_size": 50257,
    "context_length": 1024,

    "d_model": 768,
    "n_layers": 13,
    "n_heads": 12,

    "dropout": 0.2,
    "qkv_bias": False,
    "tie_word_embeddings": False,

    "max_position_embeddings": 1024,
    "hidden_size": 768,
    "num_hidden_layers": 13,
    "num_attention_heads": 12,

    "torch_dtype": "float32"
}

os.makedirs("fingpt-131m", exist_ok=True)

with open(
    "fingpt-131m/config.json",
    "w"
) as f:
    json.dump(
        config,
        f,
        indent=2
    )

print("config.json created successfully in fingpt-131m/")
