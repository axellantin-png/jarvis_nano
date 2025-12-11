import os
import re

INPUT_FILE = "data/raw_corpus2.txt"
OUTPUT_FILE = "data/clean_corpus.txt"

def remove_inline_page_numbers(text):
    """
    Supprime les numéros de pages au milieu du texte :
    – 12 –, - 45 -, — 78 —, –12– etc.
    """
    # remplace les numéros encadrés par tirets
    text = re.sub(r"[–—-]\s*\d+\s*[–—-]", " ", text)
    return text


def remove_page_numbers(text):
    # Numéros seuls sur une ligne
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Numéros isolés dans le flux
    text = re.sub(r"\s+\d+\s+", " ", text)
    return text


def fix_hyphenated_words(text):
    # Recolle les mots coupés à la ligne
    return re.sub(r"-\s*\n\s*", "", text)


def remove_repeated_work_titles(text):
    patterns = [
        r"^\s*L['’]Homme Qui Rit\s*$",
        r"^\s*LES\s+MISÉRABLES\s*$",
        r"^\s*NOTRE[- ]DAME\s+DE\s+PARIS\s*$",
        r"^\s*J\.-?J\.?\s*Rousseau\s*$",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.MULTILINE)
    return text


def remove_chapter_titles(text):
    """
    Supprime :
      - TITRES EN MAJUSCULES
      - 'X. TITRE DU CHAPITRE'
      - 'IV. TITRE ... 175'
      - 'Chapitre X', 'CHAPITRE IV'
      - Variantes avec chiffres romains ou arabes
    """

    # 1. Lignes full majuscules (souvent des titres)
    text = re.sub(
        r"^[A-ZÉÈÀÂÊÎÔÛÙÇ0-9 ,.'’\-:;!?]{8,}$",
        "",
        text,
        flags=re.MULTILINE
    )

    # 2. Titres du type : "X. TITRE …" ou "IV. La suite …"
    text = re.sub(
        r"^[IVXLC0-9]+\.\s+[A-ZÉÈÀÂÊÎÔÛÙÇ][^\n]+$",
        "",
        text,
        flags=re.MULTILINE
    )

    # 3. Titres du type : "Chapitre X", "CHAPITRE XX"
    text = re.sub(
        r"^\s*(Chapitre|CHAPITRE)\s+[IVXLC0-9]+\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # 4. Cas particulier : titre + numéro de page collé
    text = re.sub(
        r"^[A-ZÉÈÀÂÊÎÔÛÙÇ ]+\s+\d{1,4}$",
        "",
        text,
        flags=re.MULTILINE
    )

    return text


def normalize_spacing(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])([A-Za-zÀ-ÿ])", r"\1 \2", text)
    return text.strip()


def clean_text(text):
    text = fix_hyphenated_words(text)
    text = remove_page_numbers(text)
    text = remove_chapter_titles(text)
    text = remove_repeated_work_titles(text)
    text = normalize_spacing(text)
    text = remove_inline_page_numbers(text)

    return text


def merge_and_clean():
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    print(f"Taille avant nettoyage : {len(raw)}")

    cleaned = clean_text(raw)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write(cleaned)

    print(f"Taille après nettoyage : {len(cleaned)}")



if __name__ == "__main__":
    merge_and_clean()
