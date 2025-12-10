import re
from tqdm import tqdm

# --------------------------
# 0) Nom des fichiers
# --------------------------
INPUT_FILE = "data/corpus_fr.txt"      # ton fichier brut (livres + wiki + dialogues)
OUTPUT_FILE = "data/corpus_fr_fin.txt"      # fichier nettoyé final


# --------------------------
# 1) Nettoyage global brut
# --------------------------

def remove_speaker_tags(text):
    # Enlever les formes "spk1:", "spk2 :", "SPK12 :", etc.
    text = re.sub(r"\bspk\d+\s*:\s*", "", text, flags=re.IGNORECASE)

    # Enlever les formes "speaker1:", "locuteur3:", "intervenant2:"
    text = re.sub(r"\b(speaker|locuteur|intervenant)\d+\s*:\s*", "", text, flags=re.IGNORECASE)

    # Enlever les marqueurs style "L1:", "L2:", etc.
    text = re.sub(r"\bL\d+\s*:\s*", "", text)

    # Nettoyage optionnel des préfixes mal formés
    text = re.sub(r"^\s*:\s*", "", text)

    return text

def remove_noise_tags(text):
    return re.sub(r"\[NOISE\]\s*", "", text, flags=re.IGNORECASE)

def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def remove_page_numbers(text):
    # supprime nombres isolés (souvent numéros de pages)
    return re.sub(r"\b\d{1,4}\b", " ", text)

def basic_cleanup(text):
    # garde ponctuation FR et lettres accentuées
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9 ,.'’?!:;\"()\-\n]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------
# 2) Reconstruction des paragraphes PDF
# ------------------------------------

def reconstruct_pdf_lines(lines):
    """
    Fusionne les lignes qui ne se terminent pas par ., ?, ! ou :
    pour reconstruire des phrases coupées par les PDF.
    """
    merged = []
    buffer = ""

    end_mark = tuple(".?!")  # fins de phrase

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # si le buffer est vide, on commence une nouvelle phrase
        if not buffer:
            buffer = line
        else:
            # vérifier si la ligne précédente semblait finie
            if buffer.endswith(end_mark):
                # on push, et on démarre une nouvelle
                merged.append(buffer)
                buffer = line
            else:
                # sinon, c'était une phrase coupée → concaténer
                buffer += " " + line

    # push final
    if buffer:
        merged.append(buffer)

    return merged


# --------------------------
# 3) Split en phrases
# --------------------------

def split_into_sentences(text):
    text = re.sub(r"([.!?])", r"\1 ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def clean_sentence(sentence):
    sentence = sentence.strip()
    if not sentence:
        return ""

    # pas de caractères bizarres
    sentence = re.sub(r"[^a-zA-ZÀ-ÿ0-9 ,.'’?!:;\"()\-\n]", " ", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"\s+([,.?!:;])", r"\1", sentence)

    if len(sentence) < 10:
        return ""

    return sentence.strip()

def clean_dialogue_artifacts(sentence):
    # enlever les répétitions genre "euh", "ben", "hein" quand triplement répétés
    sentence = re.sub(r"\b(euh|ben|hein|bah)(\s+\1){1,3}\b", r"\1", sentence, flags=re.IGNORECASE)

    # enlever les vestiges de pauses "//", "///"
    sentence = re.sub(r"/{1,5}", " ", sentence)

    # enlever les "+", " - ", " + "
    sentence = re.sub(r"\s*[+]+\s*", " ", sentence)

    # espaces propres
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()


# --------------------------
# MAIN PIPELINE
# --------------------------

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    print("Nettoyage global du texte brut…")

    raw_text = "\n".join(raw_lines)
    raw_text = remove_speaker_tags(raw_text)
    raw_text = remove_noise_tags(raw_text)
    raw_text = remove_urls(raw_text)
    raw_text = remove_page_numbers(raw_text)
    raw_text = basic_cleanup(raw_text)

    # Séparer par lignes pour reconstruction PDF
    lines = raw_text.split("\n")

    # reconstruction PDF : fusion des lignes brisées
    print("Reconstruction des lignes PDF…")
    merged_lines = reconstruct_pdf_lines(lines)

    # split en phrases
    print("Découpage en phrases…")
    all_sentences = []
    for line in merged_lines:
        all_sentences.extend(split_into_sentences(line))

    # nettoyage final phrase par phrase
    print("Nettoyage phrase par phrase…")
    cleaned = []
    for s in tqdm(all_sentences, desc="Nettoyage"):
        cs = clean_sentence(s)
        cs = clean_dialogue_artifacts(cs)
        if cs:
            cleaned.append(cs)

    # option : enlever doublons
    cleaned = list(dict.fromkeys(cleaned))

    # écrire fichier final propre
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in cleaned:
            f.write(s + "\n")

    print(f"Fichier propre généré : {OUTPUT_FILE}")
    print(f"Nombre final de phrases : {len(cleaned)}")


if __name__ == "__main__":
    main()