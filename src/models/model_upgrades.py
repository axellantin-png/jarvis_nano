import re
from src.models.memory import MemoryController
from src.models.memory import MemoryItem
from typing import List, Dict, Any, Optional


##############################################################################################################################################################################################
# correction de la plus part des erreurs du tokenizer 
##############################################################################################################################################################################################


def postprocess_french_text(text: str) -> str:
    # 1) Recolle les nombres éclatés : "1 9 9 9" -> "1999"
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)

    # 2) Recolle les lettres isolées dans un mot
    # ex: "plan ning" -> "planning"
    text = re.sub(
        r'(?<=[A-Za-zÀ-ÖØ-öø-ÿ])\s+(?=[A-Za-zÀ-ÖØ-öø-ÿ])',
        '',
        text
    )

    # 3) Nettoyage des espaces avant ponctuation
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)

    # 4) Nettoyage espaces après parenthèses ouvrantes
    text = re.sub(r'\(\s+', '(', text)

    # 5) Nettoyage espaces avant parenthèses fermantes
    text = re.sub(r'\s+\)', ')', text)

    # 6) Nettoyage espaces multiples
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()

##############################################################################################################################################################################################
# amelioration de la mémoire pour le modèle
##############################################################################################################################################################################################

def build_prompt_with_memory(retrieved: List[MemoryItem], user_message: str) -> str:
    mem_block = ""
    if retrieved:
        mem_lines = [f"- {m.text}" for m in retrieved]
        mem_block = "Mémoire pertinente (faits connus sur l'utilisateur / le contexte):\n" + "\n".join(mem_lines)
    else:
        mem_block = "Mémoire pertinente : (aucune mémoire pertinente trouvée)"

    prompt = f"""
Tu es Jarvis, un assistant qui dispose d'une mémoire externe.

{mem_block}

Message actuel de l'utilisateur :
{user_message}

En t'appuyant sur la mémoire si elle est utile, réponds de manière cohérente et utile.
"""
    return prompt


class JarvisAgent:
    def __init__(self, llm, memory_controller: MemoryController):
        self.llm = llm
        self.memory_controller = memory_controller

    def handle_message(self, user_message: str, user_id: str) -> str:
        # 1. retrieve mémoire
        mem_items = self.memory_controller.retrieve_memory(user_message, top_k=5)

        # 2. construire prompt
        prompt = build_prompt_with_memory(mem_items, user_message)

        # 3. appel LLM
        assistant_reply = self.llm.generate(prompt)

        # 4. décider ce qu'on mémorise
        self.memory_controller.decide_and_write(
            user_message=user_message,
            assistant_message=assistant_reply,
            user_id=user_id,
        )

        return assistant_reply
    

##############################################################################################################################################################################################
# amelioration de la reflection du modèle 
##############################################################################################################################################################################################
    
class StructuredCoTWithReflection:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, user_input: str) -> str:
        # -------- PASS 1 : raisonnement structuré --------
        prompt_1 = f"""
Tu es un assistant logique et structuré.

Règles IMPORTANTES :
- Suis STRICTEMENT le format demandé.
- N'invente pas d'informations.
- Écris clairement et simplement.
- Ne répète pas inutilement les mots.

Format obligatoire :

<analysis>
- Sujet principal :
- Intention de la réponse :
- Informations pertinentes :
- Plan de réponse :
</analysis>

<draft>
</draft>

Question :
{user_input}
"""

        draft_output = self.llm.generate(
            prompt_1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
        )

        # -------- PASS 2 : self-reflection --------
        prompt_2 = f"""
Tu vas relire la réponse suivante et l'améliorer.

Objectifs :
- Corriger les incohérences logiques
- Supprimer les répétitions
- Clarifier les phrases confuses
- Améliorer la structure globale

Voici la réponse à analyser :

{draft_output}

Instructions :
- Ne montre PAS ton raisonnement.
- Donne UNIQUEMENT la version finale améliorée.
"""

        final_output = self.llm.generate(
            prompt_2,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        return final_output.strip()

