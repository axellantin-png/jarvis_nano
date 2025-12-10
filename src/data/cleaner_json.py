import json
import re
import sys
from pathlib import Path

def clean_response(text: str) -> str:
    """
    Supprime les séquences <CALL:...> dans les réponses.
    Nettoie également les espaces inutiles.
    """
    if not isinstance(text, str):
        return text

    # enlever <CALL:...>
    text = re.sub(r"<CALL:[^>]*>", "", text)

    # nettoyer espaces en début/fin
    text = text.strip()

    return text


def clean_jsonl(input_path: str, output_path: str):
    """
    Lit un fichier JSONL, nettoie chaque réponse et écrit un fichier propre.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"❌ Fichier introuvable : {input_path}")
        sys.exit(1)

    clean_lines = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)
            if "response" in obj:
                obj["response"] = clean_response(obj["response"])

            clean_lines.append(obj)

    with output_path.open("w", encoding="utf-8") as fout:
        for item in clean_lines:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f" Nettoyage terminé !")
    print(f" Fichier propre écrit ici : {output_path}")
    print(f" Nombre d'exemples : {len(clean_lines)}")


if __name__ == "__main__":
    file_to_clean = "data/jarvis_instructions.jsonl"
    clean_file = "data/jarvis_instructions_clean.jsonl"

    clean_jsonl(file_to_clean, clean_file)
