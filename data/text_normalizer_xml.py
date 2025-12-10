import zipfile
import re
import xml.etree.ElementTree as ET

DOCX_FILE = "data/dialogues2.docx"
OUTPUT_FILE = "data/intermediate_dialogues.txt"

# ----------------------------------------------------
# 1) Extraire le XML Trans à partir du .docx Word
# ----------------------------------------------------
def extract_xml_from_docx(docx_path):
    # On lit word/document.xml
    with zipfile.ZipFile(docx_path, "r") as z:
        doc_xml = z.read("word/document.xml").decode("utf-8")

    # On parse le WordprocessingML
    root = ET.fromstring(doc_xml)

    # On récupère tout le texte contenu dans les balises <w:t>
    # (c'est là que ton XML Trans est stocké comme du texte)
    texts = [
        t.text
        for t in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        if t.text
    ]

    # On concatène tout : ça redonne ton XML Trans complet
    xml_trans = "".join(texts)
    return xml_trans


# ----------------------------------------------------
# 2) Nettoyer les balises inutiles (Sync, Event, Comment)
# ----------------------------------------------------
def clean_transcription_text(text):
    # balises autofermantes
    text = re.sub(r"<Sync[^>]*/>", " ", text)
    text = re.sub(r"<Event[^>]*/>", " ", text)

    # balises ouvrantes/fermantes
    text = re.sub(r"<Event[^>]*>.*?</Event>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<Comment[^>]*>.*?</Comment>", " ", text, flags=re.DOTALL)

    # entités XML éventuelles
    text = text.replace("&lt;", "<").replace("&gt;", ">")

    # espaces propres
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------------------------------
# 3) Extraire les tours de parole <Turn speaker="X">...</Turn>
# ----------------------------------------------------
def extract_turns(xml_string):
    # Match les blocs <Turn ...>...</Turn>
    turn_blocks = re.findall(
        r"<Turn([^>]*)>(.*?)</Turn>",
        xml_string,
        flags=re.DOTALL
    )

    results = []

    for attrs, content in turn_blocks:

        # -----------------------
        # 1) Récupération du speaker
        # -----------------------
        speaker_match = re.search(r'speaker="([^"]+)"', attrs)
        speaker = speaker_match.group(1) if speaker_match else "unknown"

        # -----------------------
        # 2) Nettoyage initial
        # -----------------------
        cleaned = clean_transcription_text(content)

        # -----------------------
        # 3) Nettoyages spécifiques aux dialogues
        # -----------------------
        # enlever les + et ///
        cleaned = re.sub(r"\+ ?", " ", cleaned)
        cleaned = re.sub(r"/{1,3}", " ", cleaned)

        # enlever les hésitations style < xxx > ou < />
        cleaned = re.sub(r"<[^>]*>", " ", cleaned)

        # enlever les répétitions incomplètes (ex: euh euh, on on)
        cleaned = re.sub(r"\b(\w+)\s+\1\b", r"\1", cleaned)

        # espaces propres
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if cleaned:
            # Ligne finale
            results.append(f"{speaker}: {cleaned}")

    return results


# ----------------------------------------------------
# 4) Pipeline complet TEXT -> TEXT structuré
# ----------------------------------------------------
def main():
    print("Extraction du XML Trans depuis le DOCX…")
    xml_str = extract_xml_from_docx(DOCX_FILE)

    print("Extraction des tours de parole…")
    turns = extract_turns(xml_str)

    print(f"{len(turns)} tours extraits.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(t + "\n")

    print(f"Fichier généré : {OUTPUT_FILE}")
    print("➡ Tu peux maintenant passer ce fichier dans text_normalizer.py sans rien changer.")


if __name__ == "__main__":
    main()
