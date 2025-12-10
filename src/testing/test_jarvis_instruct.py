import torch
from transformers import PreTrainedTokenizerFast
from model import MiniGPT

TOKENIZER_DIR = "tokenizer_hf"
MODEL_PATH = "models/jarvis_instruct/model_epoch2.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    vocab_size = tokenizer.vocab_size

    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.0,
    ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Exemple de requête utilisateur
    user_instruction = "Ouvre Spotify"

    prompt = f"<|user|> {user_instruction}\n<|assistant|>"

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
    print("\n=== Réponse brute du modèle ===")
    print(text)


if __name__ == "__main__":
    main()
