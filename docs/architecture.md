# 🧠 Jarvis-nano — Architecture Technique

## 1. Objectif du projet
Créer un assistant personnel local (Jarvis) basé sur un nano-LLM (< 100M de paramètres) 
capable de comprendre des instructions en français et d’exécuter quelques actions locales.

---

## 2. Vue d'ensemble du système

**Composants principaux :**
1. **LLM (nano modèle Transformer)** — cœur du système, comprend et génère du texte.  
2. **Orchestrateur Python** — lit les réponses du modèle et exécute les fonctions demandées (`<CALL:...>`).  
3. **Outils (Tools)** — fonctions Python locales (ouvrir app, ajouter tâche, etc.).  
4. **Mémoire / RAG** — pour retrouver des infos personnelles.  
5. **Interface vocale** — micro (STT) + voix (TTS).  

**Schéma simplifié :**



---

## 3. Architecture du modèle (LLM)

| Paramètre | Valeur prévue | Commentaire |
|------------|---------------|--------------|
| Type | Decoder-only Transformer | similaire à GPT |
| n_layers | 8 | nombre de blocs empilés |
| n_heads | 8 | têtes d’attention |
| d_model | 512 | dimension interne |
| vocab_size | 10 000 | taille du tokenizer |
| context_length | 512 | taille du contexte max |
| activation | SwiGLU | plus stable que ReLU |
| normalization | RMSNorm | légère et rapide |

---

## 4. Flux de données

1. L’utilisateur parle → STT transforme en texte.  
2. Texte → tokenisation → entrée du modèle.  
3. Le modèle génère une sortie textuelle.  
4. Si la sortie contient `<CALL:...>` → l’orchestrateur exécute la commande Python.  
5. Le résultat est repassé dans le modèle pour formuler la réponse finale.  

---

## 5. Modules logiciels

| Module | Fichier | Description |
|---------|----------|-------------|
| Entraînement | `src/train.py` | fine-tuning du modèle sur données Jarvis |
| Inference | `src/infer.py` | génération de texte / actions |
| Tools | `src/tools.py` | définitions des fonctions locales |
| Utils | `src/utils.py` | chargement modèle, logs, etc. |
| Interface | `cli.py` | interface console ou vocale |

---

## 6. Données utilisées

- **Corpus de base** : échantillons FR de textes généraux (Wikipedia, dialogues publics).  
- **Données d’instruction** : paires personnalisées ("User" → "Jarvis") avec actions.  

---

## 7. Roadmap du projet (v0 → v3)

| Version | Fonctionnalités principales |
|----------|-----------------------------|
| v0 | LLM répond à du texte simple en local |
| v1 | Outils locaux (`open_app`, `add_todo`) |
| v2 | Mémoire vectorielle (notes personnelles) |
| v3 | Voix + planificateur multi-actions |

---

## 8. Environnement de dev

- **Langage** : Python 3.10  
- **Frameworks** : PyTorch, Hugging Face Transformers, SentencePiece  
- **Outils** : FAISS, Whisper, Piper  
- **OS cible** : Linux ou Windows

---

## 9. Sécurité et contraintes
- Le modèle ne peut exécuter que des outils explicitement autorisés.
- Confirmation obligatoire avant action critique.
- Toutes les interactions sont loggées localement.

---

## 10. Notes de conception
- Modularité → chaque composant indépendant.
- Priorité aux modèles < 100 Mo pour tourner sur CPU.
- Possibilité de quantisation int8/int4.

