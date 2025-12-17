
##############################################################################################################################################################################################
# les modules, classes et fonctions importées 
##############################################################################################################################################################################################

import torch
import torch.nn.functional as F
from collections import Counter
from transformers import PreTrainedTokenizerFast
from src.models.model import MiniGPT
from src.models.model_upgrades import postprocess_french_text

##############################################################################################################################################################################################
# les paramètres 
##############################################################################################################################################################################################
TOKENIZER_DIR = "tokenizer_hf"
MODEL_A_REPRENDRE = "models/jarvis_from_scratch_384_6_6/model_epoch6.pt"


##############################################################################################################################################################################################
# les fonctions 
##############################################################################################################################################################################################

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.0,
    device="cuda"
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model(generated)
        logits = outputs.logits[:, -1, :]

        # repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                logits[:, token_id] /= repetition_penalty

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # top-p
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(sorted_probs, 1)
        next_token = sorted_indices.gather(-1, next_token)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

def generate_cot_structured(model, tokenizer, user_input, device="cuda"):
    prompt = f"""
Tu es un assistant logique et structuré.

Analyse :
- Sujet principal :
- Intention de la réponse :
- Informations pertinentes :
- Plan de réponse :

Brouillon :
Réponds clairement et sans répétitions.

Question :
{user_input}
"""
    return generate(
        model,
        tokenizer,
        prompt,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        device=device
    )


def generate_reflection(model, tokenizer, draft_output, device="cuda"):
    prompt = f"""
Tu vas améliorer la réponse suivante.

Tâches :
- Corriger les incohérences
- Supprimer les répétitions
- Clarifier les phrases

Texte à corriger :
{draft_output}

Réponse finale :
"""
    return generate(
        model,
        tokenizer,
        prompt,
        temperature=0.55,
        top_p=0.9,
        repetition_penalty=1.1,
        device=device
    )

def generate_cot_with_reflection(model, tokenizer, user_input, device="cuda"):
    draft = generate_cot_structured(model, tokenizer, user_input, device)
    final = generate_reflection(model, tokenizer, draft, device)
    return final


def repetition_score(text, n=3):
    tokens = text.split()
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c-1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)

@torch.no_grad()
def perplexity_proxy(model, tokenizer, text, device="cuda"):
    model.eval()
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean"
    )
    return torch.exp(loss).item()

def compare_generation(model, tokenizer, prompts, device="cuda"):
    results = []

    for prompt in prompts:
        base = generate(
            model,
            tokenizer,
            prompt,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.0,
            device=device
        )

        cot = generate_cot_with_reflection(
            model,
            tokenizer,
            prompt,
            device=device
        )

        results.append({
            "prompt": prompt,
            "baseline": {
                "text": postprocess_french_text(base),
                "length": len(base.split()),
                "repetition": repetition_score(base),
                "ppl_proxy": perplexity_proxy(model, tokenizer, base, device),
            },
            "cot_reflection": {
                "text": postprocess_french_text(cot),
                "length": len(cot.split()),
                "repetition": repetition_score(cot),
                "ppl_proxy": perplexity_proxy(model, tokenizer, cot, device),
            }
        })

    return results


##############################################################################################################################################################################################
# le main  
##############################################################################################################################################################################################


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    vocab_size = tokenizer.vocab_size

    model =  MiniGPT(
            vocab_size=vocab_size,
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_ff=1024,
            max_seq_len=64,
            dropout=0.1,
        ).to(device)

    state_dict = torch.load(MODEL_A_REPRENDRE, map_location=device)
    model.load_state_dict(state_dict)

    test_prompts = [
        "Explique pourquoi les véhicules électriques sont importants.",
        "Résume l'impact culturel de Star Wars.",
        "Explique ce qu'est un pays fondateur."
    ]

    results = compare_generation(model, tokenizer, test_prompts)

    for r in results:
        print("="*80)
        print("PROMPT:", r["prompt"])
        print("\n--- BASELINE ---")
        print(r["baseline"]["text"])
        print("repetition:", r["baseline"]["repetition"])
        print("ppl:", r["baseline"]["ppl_proxy"])

        print("\n--- CoT + REFLECTION ---")
        print(r["cot_reflection"]["text"])
        print("repetition:", r["cot_reflection"]["repetition"])
        print("ppl:", r["cot_reflection"]["ppl_proxy"])



if __name__ == "__main__":
    main()
