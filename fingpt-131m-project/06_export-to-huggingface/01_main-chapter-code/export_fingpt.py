import torch
import json
import os
from configuration_fingpt import FinGPTConfig
from modeling_fingpt import FinGPTForCausalLM
from transformers import AutoTokenizer

# Paths relative to this script's location
_HERE = os.path.dirname(os.path.abspath(__file__))
_BEST_PT  = os.path.join(_HERE, "..", "..", "..", "05_training", "01_main-chapter-code", "best.pt")
_MODEL_OUT = os.path.join(_HERE, "fingpt-131m")

print("1. Creating FinGPTConfig...")
config = FinGPTConfig(
    vocab_size=50257,
    context_length=1024,
    d_model=768,
    n_layers=13,
    n_heads=12,
    dropout=0.2,
    qkv_bias=False,
    tie_word_embeddings=False,
)

os.makedirs(_MODEL_OUT, exist_ok=True)
config.save_pretrained(_MODEL_OUT)

print("2. Loading state_dict from best.pt...")
ckpt = torch.load(_BEST_PT, map_location="cpu", weights_only=False)
model_state_dict = ckpt.get("model_state_dict", ckpt)

print("3. Remapping state_dict keys to match HF model...")
new_state_dict = {}
for k, v in model_state_dict.items():
    # Remap FeedForward keys to FinGPTMLP keys
    k = k.replace("ffn.net.0", "mlp.fc1")
    k = k.replace("ffn.net.2", "mlp.fc2")
    # Also handle the renaming of ffn -> mlp if there are other parameters like biases
    k = k.replace("ffn.net", "mlp")
    
    new_state_dict[k] = v

print("4. Initializing HF Model and loading weights...")
hf_model = FinGPTForCausalLM(config)
hf_model.load_state_dict(new_state_dict, strict=True)
print("[OK] State dict loaded successfully")

print("5. Saving Pretrained HF model to", _MODEL_OUT)
# Copy embedding values into head (untied copy) to preserve weight values
hf_model.head.weight.data.copy_(hf_model.token_emb.weight.data)
hf_model.save_pretrained(_MODEL_OUT, safe_serialization=True)

print("6. Testing the uploaded-format model...")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
text = "What is GST?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = hf_model(input_ids=inputs["input_ids"])

print(f"Logits shape: {outputs['logits'].shape}")

print("7. Generating Model Card (README.md)...")
model_card = """# FinGPT-131M

FinGPT-131M is a 131M-parameter decoder-only language model
trained from scratch on a GST/tax regulatory corpus.

## Architecture

- Parameters: ~131.3M
- Layers: 13
- Hidden size: 768
- Attention heads: 12
- Context length: 1024
- Vocabulary: 50,257
- Tokenizer: GPT-2 BPE
- Activation: GELU
- Attention: causal scaled dot-product attention
- Weight tying: Yes

## Training

Best checkpoint:
- Step: 11,250
- Validation loss: 2.7641

The model was trained from scratch; it is not fine-tuned
from GPT-2.

## Intended use

The model is intended for experimentation and research
on language modeling over GST and tax-related text.

## Limitations

This model should not be treated as a source of authoritative
legal or tax advice. Tax regulations can change, and model
outputs may contain hallucinations or outdated information.
"""

with open(os.path.join(_MODEL_OUT, "README.md"), "w", encoding="utf-8") as f:
    f.write(model_card)

print("Export process complete!")
