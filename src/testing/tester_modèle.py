import json
import torch
from transformers import PreTrainedTokenizerFast
from model import MiniGPT
"""
def load_model_and_tokenizer(...):
    ...
"""

def evaluate():
    model, tokenizer, device = load_model_and_tokenizer(...)
    tests = []
    with open("tests/jarvis_eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            tests.append(json.loads(line))

    success = 0

    for t in tests:
        prompt = f"<|user|> {t['input']}\n<|assistant|>"
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=64,
                temperature=0.7,
                top_k=50,
            )

        text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
        ok = t["expected_contains"].lower() in text.lower()
        print("Input:", t["input"])
        print("Model:", text)
        print("OK:", ok)
        print("---")
        success += int(ok)

    print(f"Score: {success}/{len(tests)} réussis")

if __name__ == "__main__":
    evaluate()
