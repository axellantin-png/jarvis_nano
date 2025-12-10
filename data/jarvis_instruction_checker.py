import json

path = "data/jarvis_instructions.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Ligne {i} invalide :", e)
            break
    else:
        print(" Toutes les lignes sont valides.")
