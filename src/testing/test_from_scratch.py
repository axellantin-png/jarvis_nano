import torch
from transformers import PreTrainedTokenizerFast
import argparse
from model import MiniGPT

TOKENIZER_DIR = "tokenizer_hf"
MODEL_PATH = "models/jarvis_from_scratch/model_epoch3.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
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

    prompt = "bonjour, aujourd'hui nous sommes "
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.9,
            top_k=50
        )

    text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    print("Généré :")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MiniGPT model with a prompt.")
    parser.add_argument("model_path", type=str, help="Path to the trained model (.pt file)")
    parser.add_argument("prompt", type=str, help="Prompt to complete")

    args = parser.parse_args()

    # Exemple :
    # args.model_path -> "models/jarvis_from_scratch/model_epoch2.pt"
    # args.prompt -> "Le projet de la réunion est"

    model_path = args.model_path
    prompt = args.prompt

    # Charger le tokenizer
    from tokenizer_hf import load_tokenizer  # à adapter selon où tu l'importes réellement
    tokenizer = load_tokenizer()

    # Charger ton modèle
    model = MiniGPT(...)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    # Encoder le prompt
    enc = tokenizer(prompt, return_tensors="pt")

    # Génération
    with torch.no_grad():
        out = model.generate(enc["input_ids"], max_new_tokens=50)

    # Décoder
    print("Prompt :", prompt)
    print("Réponse :", tokenizer.decode(out[0], skip_special_tokens=True))

