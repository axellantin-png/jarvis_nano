from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid
import json
from datetime import datetime
import chromadb
from chromadb.config import Settings
from datetime import datetime


BASE_IMPORTANCE = {
    "preference": 0.9,
    "fact": 0.6,
    "episode": 0.3,
}


@dataclass
class MemoryItem:
    id: str
    text: str
    metadata: Dict[str, Any]

class MemoryStore:
    """Interface abstraite pour la mémoire vectorielle."""
    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryItem]:
        raise NotImplementedError



class ChromaMemoryStore(MemoryStore):
    def __init__(self, collection_name: str, embed_fn):
        """
        embed_fn: fonction Python text -> vector[list[float]]
        """
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.embed_fn = embed_fn

    def add(self, text: str, metadata: Dict[str, Any]) -> str:
        mem_id = str(uuid.uuid4())
        embedding = self.embed_fn(text)
        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return mem_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryItem]:
        q_emb = self.embed_fn(query)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
        )
        items = []
        for i in range(len(res["ids"][0])):
            items.append(
                MemoryItem(
                    id=res["ids"][0][i],
                    text=res["documents"][0][i],
                    metadata=res["metadatas"][0][i],
                )
            )
        return items

class MemoryController:
    def __init__(self, store: MemoryStore, llm):
        self.store = store
        self.llm = llm  # ton modèle (Jarvis)

    def build_retrieval_query(self, user_message: str) -> Dict[str, Any]:
        """
        Version simple : on dérive un 'topic' à partir du message.
        Version avancée : on utilise le LLM pour faire ce routing.
        """
        # Ex très simple, à raffiner :
        topic = "general"
        if "musique" in user_message.lower():
            topic = "music"
        if "rappel" in user_message.lower() or "rappelle" in user_message.lower():
            topic = "reminder"

        return {
            "query": user_message,
            "where": {"topic": topic}
        }

    def retrieve_memory(self, user_message: str, top_k: int = 5) -> List[MemoryItem]:
        plan = self.build_retrieval_query(user_message)
        return self.store.search(
            query=plan["query"],
            top_k=top_k,
            where=plan.get("where")
        )


MEMORY_DECISION_PROMPT = """
Tu joues le rôle d'un module de mémoire pour un assistant.

On te donne :
- le message de l'utilisateur
- la réponse de l'assistant

Ta tâche :
1. Décider si quelque chose doit être mémorisé pour aider dans de futures conversations.
2. Si oui, extraire cette information en UNE phrase factuelle, stable et utile.
3. Classer la mémoire.

Renvoie STRICTEMENT un JSON avec les champs :
- "should_write": bool
- "memory_text": str (ou "" si rien)
- "memory_type": "preference" | "fact" | "episode"
- "topic": str

Message utilisateur :
{user_message}

Réponse assistant :
{assistant_message}
"""

class MemoryController(MemoryController):  # on étend la classe plus haut
    def decide_and_write(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
    ) -> Optional[str]:
        prompt = MEMORY_DECISION_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message
        )
        raw = self.llm.generate_json(prompt)  # à adapter à ton API
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not data.get("should_write", False):
            return None

        mem_text = data.get("memory_text", "").strip()
        if not mem_text:
            return None

        metadata = {
            "type": data.get("memory_type", "episode"),
            "topic": data.get("topic", "general"),
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        return self.store.add(mem_text, metadata)
    

def time_decay_factor(last_accessed_at: str, half_life_days: float = 30.0) -> float:
    """
    Retourne un facteur entre ~0 et 1.
    half_life_days = le temps après lequel la valeur est divisée par 2.
    """
    last = datetime.fromisoformat(last_accessed_at)
    now = datetime.utcnow()
    age_days = (now - last).total_seconds() / 86400.0
    # décroissance exponentielle
    import math
    return 0.5 ** (age_days / half_life_days)

def compute_memory_score(meta: dict) -> float:
    base_imp = BASE_IMPORTANCE.get(meta.get("type", "episode"), 0.3)
    imp = meta.get("importance", base_imp)

    decay = time_decay_factor(
        meta.get("last_accessed_at", meta.get("created_at"))
    )

    # bonus si très souvent utilisée
    access_bonus = min(meta.get("access_count", 0) / 10.0, 1.0)  # max +1

    return imp * decay + 0.1 * access_bonus

def consolidate_memories(memories: list, llm, user_id: str, topic: str, store): # j'ai ajouter le store ici
    texts = [m.text for m in memories]
    joined = "\n".join(f"- {t}" for t in texts)

    prompt = f"""
    Tu es un module de mémoire.
    Voici une liste de souvenirs épisodiques de l'utilisateur sur le thème '{topic}' :

    {joined}

    Fais un résumé compact en 3 à 5 phrases, en ne gardant que
    les informations qui peuvent être utiles à long terme.
    """

    summary = llm.generate(prompt).strip()

    new_metadata = {
        "type": "episode",
        "topic": topic,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "last_accessed_at": datetime.utcnow().isoformat(),
        "access_count": 0,
        "importance": 0.5,  # plus élevé que chaque petit fragment
    }

    new_id = store.add(summary, new_metadata)

    # supprimer les anciennes
    old_ids = [m.id for m in memories]
    store.delete_many(old_ids)

    return new_id

def cleanup_memory(store, llm, user_id: str):
    # 1. Récupérer toutes les mémoires de ce user
    all_memories = store.list_all(user_id=user_id)  # à implémenter dans ton store

    # 2. Calculer le score de chacune
    scored = []
    for m in all_memories:
        s = compute_memory_score(m.metadata)
        scored.append((m, s))

    # 3. Suppressions simples
    to_delete = []
    for m, score in scored:
        t = m.metadata.get("type", "episode")
        if t == "episode" and score < 0.15:
            to_delete.append(m.id)
        elif t == "fact" and score < 0.1:
            to_delete.append(m.id)
        # preferences : on ne supprime pas ici

    if to_delete:
        store.delete_many(to_delete)

    # 4. Consolidation par topic pour les vieux épisodes
    from collections import defaultdict
    by_topic = defaultdict(list)

    for m, score in scored:
        if m.id in to_delete:
            continue
        if m.metadata.get("type") != "episode":
            continue

        created = datetime.fromisoformat(m.metadata["created_at"])
        age_days = (datetime.utcnow() - created).total_seconds() / 86400.0
        if age_days > 30:  # plus vieux qu'un mois
            topic = m.metadata.get("topic", "general")
            by_topic[topic].append(m)

    # consolider les topics avec beaucoup de mémoires
    for topic, mems in by_topic.items():
        if len(mems) < 5:
            continue  # pas besoin de consolider si peu
        consolidate_memories(mems, llm, user_id=user_id, topic=topic)


def update_preference(store, llm, new_memory_text: str, topic: str, user_id: str):
    old_prefs = store.search(
        query=topic,
        top_k=5,
        where={"user_id": user_id, "type": "preference", "topic": topic},
    )
    if not old_prefs:
        return store.add(new_memory_text, { ... })

    texts = [m.text for m in old_prefs] + [new_memory_text]
    joined = "\n".join(f"- {t}" for t in texts)

    prompt = f"""
On a plusieurs informations sur les préférences de l'utilisateur pour le sujet '{topic}' :

{joined}

Résume et donne la version la plus à jour en UNE seule phrase.
"""
    unified = llm.generate(prompt).strip()

    # supprimer anciennes pref
    store.delete_many([m.id for m in old_prefs])

    # ajouter la version unifiée
    meta = {
        "type": "preference",
        "topic": topic,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "last_accessed_at": datetime.utcnow().isoformat(),
        "access_count": 0,
        "importance": 0.95,
    }
    return store.add(unified, meta)
