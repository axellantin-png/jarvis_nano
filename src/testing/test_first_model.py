from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch

tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_hf")
model = AutoModelForCausalLM.from_pretrained("models/jarvis_first")

# Assure-toi que pad_token existe
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

prompt = "<|user|>Ouvre Spotify\n<|assistant|>"

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
outputs = model.generate(
    **inputs,
    max_length=80,
    temperature=0.7,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=False))

