# src/train_jarvis_instruct.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from model import MiniGPT
from instruction_dataset import InstructionDataset
from tqdm import tqdm


DATA_FILE = "data/jarvis_instructions_clean.jsonl"
TOKENIZER_DIR = "tokenizer_hf"
PRETRAINED_MODEL_PATH = "models/jarvis_from_scratch/model_epoch2.pt" 
OUTPUT_DIR = "models/jarvis_instruct"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) Charger tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    # 2) Créer dataset d'instructions
    max_length = 128
    dataset = InstructionDataset(DATA_FILE, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("Nombre d'exemples d'instruction:", len(dataset))

    # 3) Charger modèle pré-entraîné (langage) comme init
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=max_length,
        dropout=0.1,
    ).to(device)

    if os.path.exists(PRETRAINED_MODEL_PATH):
        print("Chargement du modèle pré-entraîné depuis", PRETRAINED_MODEL_PATH)
        state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("⚠️ Modèle pré-entraîné introuvable, on entraîne uniquement sur instructions (ça marchera mais sera plus faible).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    num_epochs = 50

    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # attention_mask pas encore utilisé dans MiniGPT, mais dispo si tu veux l'ajouter
            # attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 20 == 0:
                avg = total_loss / 20
                print(f"Epoch {epoch+1}, step {step}, loss = {avg:.4f}")
                total_loss = 0.0

        # sauvegarde par epoch
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pt"))
    print("✔ Instruction-tuning terminé, modèle sauvegardé dans", OUTPUT_DIR)


if __name__ == "__main__":
    main()
