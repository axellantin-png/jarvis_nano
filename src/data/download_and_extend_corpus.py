import os
import re
import sys
from datasets import load_dataset
from tqdm import tqdm
import datasets

# =========================================================
# CONFIG
# =========================================================

PERCENT = 3  # pourcentage du dataset à télécharger
OUTPUT_FILE = "data/corpus_fr_fin.txt"

MODE = "sub" # sub wiki ou both

# Wikipedia FR moderne
WIKI_DATASET = ("wikimedia/wikipedia", "20231101.fr")

# Dialogues FR modernes (multi30k)
DIALOGUE_DATASET = "bentrevett/multi30k"

datasets.disable_progress_bar()
os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# =========================================================
# CLEANING
# =========================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def append_to_corpus(texts):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for t in tqdm(texts, desc="➕ Ajout au corpus"):
            cleaned = clean_text(t)
            if len(cleaned) > 15:
                f.write(cleaned + "\n\n")


# =========================================================
# WIKIPEDIA
# =========================================================

def download_wikipedia(percent):
    print(f"📘 Téléchargement Wikipedia FR ({percent}%)...")

    split = f"train[:{percent}%]"
    ds = load_dataset(WIKI_DATASET[0], WIKI_DATASET[1], split=split)

    texts = []
    for row in tqdm(ds, desc="🧹 Extraction Wikipedia"):
        texts.append(row["text"])

    print(f"✔ {len(texts)} articles extraits.")
    return texts


# =========================================================
# DIALOGUES (MULTI30K)
# =========================================================

def download_dialogues(percent):
    print(f"🎭 Téléchargement dialogues Multi30k FR ({percent}%)...")

    split = f"train[:{percent}%]"
    ds = load_dataset(DIALOGUE_DATASET, split=split)

    texts = []
    for row in tqdm(ds, desc="🧹 Extraction dialogues"):
        txt = row.get("fr", "")
        if txt:
            texts.append(txt)

    print(f"✔ {len(texts)} lignes de dialogue extraites.")
    return texts


# =========================================================
# MAIN
# =========================================================

def main():
    mode = MODE

    print("🚀 Téléchargement du corpus...")

    if mode in ("wiki", "both"):
        wiki_texts = download_wikipedia(PERCENT)
        append_to_corpus(wiki_texts)
        print("✔ Ajout Wikipedia terminé.\n")

    if mode in ("sub", "both"):
        dialogue_texts = download_dialogues(PERCENT)
        append_to_corpus(dialogue_texts)
        print("✔ Ajout dialogues terminé.\n")

    print(f"🎉 Corpus final mis à jour → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
