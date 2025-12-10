# Description 

Jarvis_nano est un modèle de langage Transformer que je développe, dans le cadre d’un projet personnel dont l’objectif est de comprendre le fonctionnement d’un LLM.
Ce projet est inspiré de J.A.R.V.I.S des filmes iron man et est un terrain d'experimentation ou je peux : 
- comprendre comment se construit un dataset pour un LLM
- apprendre le fonctionnement de la tokenisation
- implémenter un Transformer (attention, feed-forward, embeddings, normalisation…)
- entraîner un modèle depuis zéro
- Comprendre comment suivre un entrainement avec les outils mis a disposition (tqdm, TensorBoard, …)
- observer et analyser toutes les dynamiques internes de l’entraînement
- expérimenter avec les hyperparamètres, architectures et techniques de génération
- développer une intuition solide sur le fonctionnement des modèles  de llms

# Objectifs 

- Comprendre chaque composant interne d’un LLM.
- Implémenter manuellement les blocs clés du Transformer.
- Préparer un corpus et entraîner un tokenizer personnalisé.
- Entraîner un modèle de langage depuis zéro.
- Visualiser la progression grâce à TensorBoard (loss, gradients, poids, génération…).
- Expérimenter librement pour voir comment le modèle réagit.
- Développer une intuition pratique de l’apprentissage profond appliqué aux LLM.

# Structure du projet 

```
jarvis_nano/
├── data
│   ├── corpus_fr_fin.txt
│   ├── corpus_fr.txt
│   ├── file.py
│   ├── jarvis_instructions_clean.jsonl
│   └── jarvis_instructions.jsonl
├── docs
│   └── architecture.md
├── models
│   ├── jarvis_from_scratch
│   └── jarvis_instruct
├── README.md
├── src
│   ├── data
│   │   ├── jarvis_instruction_checker.py
│   │   ├── text_normalizer_xml.py
│   │   └── text_normalizer.py
│   ├── docs
│   │   └── architecture.md
│   ├── model
│   │   └── model.py
│   ├── testing
│   │   ├── test_from_scratch.py
│   │   ├── test_jarvis_instruct.py
│   │   ├── test_tokenizer.py
│   │   └── tester_modèle.py
│   └── training
│       ├── dataset.py
│       ├── instruction_dataset.py
│       ├── tensorboard_llm_logger.py
│       ├── train_from_scratch.py
│       ├── train_jarvis_instruct.py
│       └── train_tokenizer.py
└── tokenizer_hf
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── tokenizer.json
```



# Entrainer le modèle 

Lancement :
python src/training/train_from_scratch.py

Le script gère :
•	les checkpoints
•	la reprise d’entraînement (il faut modifier le paramètre REPRENDRE en True et indiquer le nom du modèle a reprendre)
•	l’enregistrement automatique dans TensorBoard
•	la génération régulière d’exemples

# Suivre l'entrainement avec TensorBoard 

Le projet utilise TensorBoard pour suivre :
- la loss d’entraînement et de validation
- la perplexité
- le learning rate
- les gradients (normes, stabilité)
- les histogrammes des poids
- des exemples de texte généré pendant l’entraînement

TensorBoard est lancé automatiquement dans le http://localhost:6006/

# Roadmap

- Ajouter des métriques d’évaluation (perplexité, BLEU, etc.)
- Améliorer et enrichir le corpus
- Intégrer des techniques de sampling avancées (top-k, top-p, température)
- Tester Rotary Embeddings (RoPE)
- Support multi-GPU (long terme)
- Entraîner une version plus grande du modèle
- Explorer LoRA et le fine-tuning sur LLaMA

# contibution 

C’est un projet personnel d'apprentissage, mais toute suggestion ou discussion est la bienvenue.
N’hésitez pas à ouvrir une issue pour partager des idées ou poser des questions.

# licence 

MIT License — libre d’utilisation, de modification et d’apprentissage.


