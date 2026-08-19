from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

tokenizer.save_pretrained("./fingpt-131m")