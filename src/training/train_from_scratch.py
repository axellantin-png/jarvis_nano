import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm


from src.models.model import MiniGPT
from dataset import TextDataset
from tensorboard_llm_logger import LLMTensorboardLogger

import time
import re

path = "models/jarvis_from_scratch/model_epoch4.pt"



# pour lancer tensor board : tensorboard --logdir runs

# informations générales 
DATA_FILE = "data/corpus_fr_fin.txt"
TOKENIZER_DIR = "tokenizer_hf"
OUTPUT_DIR = "models/jarvis_from_scratch_384_6_6"


def extract_epoch(model_path: str) -> int | None:
    match = re.search(r"epoch(\d+)", model_path)
    if match:
        return int(match.group(1))
    return None


# si on veut reprendre l'entrainement 
REPRENDRE = True
MODEL_A_REPRENDRE = "models/jarvis_from_scratch_384_6_6/model_epoch6.pt"
if REPRENDRE:
    epoch_number_to_add = epoch = extract_epoch(MODEL_A_REPRENDRE)
    print(epoch_number_to_add)
else : 
    epoch_number_to_add = 0



os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # on concatène tout en un seul texte
    text = "\n".join(line.strip() for line in lines if line.strip())
    return text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) Charger tokenizer HF
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    # 2) Charger corpus brut
    text = load_corpus(DATA_FILE)
    print("Longueur texte (caractères):", len(text))

    # 3) Tokeniser tout le corpus
    print("Tokenisation du corpus...")

    # Découpe le texte en lignes non vides
    lines = [l for l in text.splitlines() if l.strip()]

    all_ids = []

    for line in tqdm(lines, desc="Tokenisation", total=len(lines)):
        enc = tokenizer(
            line,
            return_tensors=None,
            add_special_tokens=True
        )
        all_ids.extend(enc["input_ids"])

    input_ids = all_ids
    print("Nombre total de tokens:", len(input_ids))


    # 4) Créer dataset
    block_size = 64
    dataset = TextDataset(input_ids, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 5) Créer le modèle
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1024,
        max_seq_len=block_size,
        dropout=0.1,
    ).to(device)

    # si on veut reprendre l'entrainement 
    if os.path.exists(MODEL_A_REPRENDRE) and REPRENDRE :
        print("Chargement du modèle pré-entraîné depuis", MODEL_A_REPRENDRE)
        state_dict = torch.load(MODEL_A_REPRENDRE, map_location=device)
        model.load_state_dict(state_dict)


    # Important: si on a ajouté des tokens spéciaux, ajuster les embeddings
    if model.vocab_size != vocab_size:
        model.vocab_size = vocab_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    


    # 6) Boucle d'entraînement simple
    num_epochs = 30  # pour tester
    global_step = 0

    run_name = f"pretrain_{int(time.time())}"
    log_dir_var = os.path.join("runs", run_name)

    logger = LLMTensorboardLogger(log_dir=log_dir_var, port=6006)

    model.train()

    for epoch in range(num_epochs):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader)*num_epochs)
        total_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, labels=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0 : 

                # Loss + perplexité
                logger.log_train_loss(loss.item(), global_step)

                # Learning rate
                logger.log_lr(optimizer, global_step)

                # Normes des gradients
                logger.log_grad_norm(model, global_step)

                # Normes des poids
                logger.log_weight_norm(model, global_step)

            # Exemple de texte généré toutes les n étapes
            if global_step % 5000 == 0:
                logger.log_generated_text(tokenizer, model, global_step)

            # Histogram a changer si trop lourd 
            if global_step % 50000 == 0:
                logger.log_histograms(model, global_step)

            
            # si les epochs sont trop longues 
            if global_step % 100000 == 0 : 
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch{epoch+epoch_number_to_add}.pt"))




        # sauvegarde par epoch
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model_epoch{epoch+1+epoch_number_to_add}.pt"))

    logger.close()
    print("Entraînement terminé.")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_final.pt"))


if __name__ == "__main__":
    main()
