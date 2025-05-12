from .base_agent import Agent
from typing import List, Dict, Any
import sys
import os
import chromadb
from datetime import datetime
import json
import requests
import uuid

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MemoryAgent(Agent):
    """
    Agent qui gère la mémoire des messages avec le chat.
    """
    
    def __init__(self, name: str = "memory"):
        super().__init__(name)
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        self._messages = []
        self._current_slots = {
            "location": "",
            "food_type": "",
            "budget": "",
            "time": "",
        }
        
        # Créer le chemin absolu pour ChromaDB dans tourism_agent_system
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        chroma_db_path = os.path.join(base_path, "tourism_agent_system", "chroma_db")
        os.makedirs(chroma_db_path, exist_ok=True)
            
        # Initialiser ChromaDB avec le chemin absolu
        self._chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        self._load_messages_from_chromadb()
        
    def _load_messages_from_chromadb(self) -> None:
        """
        Charge les messages depuis ChromaDB.
        """
        try:
            # Récupérer tous les documents
            results = self._collection.get()
            
            if not results or "metadatas" not in results:
                return
            
            # Parcourir les métadonnées pour charger les messages et les slots
            for i, metadata in enumerate(results["metadatas"]):
                try:
                    # Charger les slots si présents
                    if "slots" in metadata and metadata["slots"]:
                        try:
                            slots = json.loads(metadata["slots"])
                            
                            # Mettre à jour les slots actuels avec les nouvelles valeurs non-nulles
                            for key, value in slots.items():
                                if value is not None:
                                    self._current_slots[key] = value
                        except json.JSONDecodeError:
                            continue

                    # Charger le message
                    if "user_message" in metadata and "ai_message" in metadata:
                        message = {
                            "role": "user",
                            "content": metadata["user_message"]
                        }
                        self._messages.append(message)

                        message = {
                            "role": "assistant",
                            "content": metadata["ai_message"]
                        }
                        if "emotion" in metadata:
                            message["emotion"] = metadata["emotion"]
                        if "intent" in metadata:
                            message["intent"] = metadata["intent"]
                        self._messages.append(message)

                except Exception as e:
                    print(f"Erreur lors du chargement du message {i+1}: {e}")
                    continue

        except Exception as e:
            print(f"Erreur lors du chargement des messages depuis ChromaDB: {e}")
            
    def add_message(self, role: str, content: str, emotion: str = None, slots: Dict[str, Any] = None, intent: str = None) -> None:
        """
        Ajoute un message à la mémoire.
        
        Args:
            role (str): Le rôle de l'émetteur du message ('user' ou 'assistant')
            content (str): Le contenu du message
            emotion (str, optional): L'émotion détectée
            slots (Dict[str, Any], optional): Les slots extraits
            intent (str, optional): L'intention détectée
        """
        try:
            message = {
                "role": role,
                "content": content,
                "emotion": emotion or "",
                "intent": intent or ""
            }
            self._messages.append(message)
            
            if role == "user":
                self._current_slots.update({k: v for k, v in slots.items() if v})
            elif role == "assistant":
                pass
            
            if self._current_conversation["user_message"] and self._current_conversation["ai_message"]:
                self._save_conversation()
                
        except Exception as e:
            print(f"Erreur lors de l'ajout du message: {e}")
            
    def _save_conversation(self) -> None:
        """
        Sauvegarde la conversation courante dans ChromaDB.
        """
        try:
            # Vérifier que nous avons les messages nécessaires
            if not self._current_conversation.get("user_message") or not self._current_conversation.get("ai_message"):
                return

            # Préparer les métadonnées
            metadata = {
                "user_message": self._current_conversation["user_message"],
                "ai_message": self._current_conversation["ai_message"],
                "emotion": self._current_conversation.get("emotion", ""),
                "intent": self._current_conversation.get("intent", ""),
                "slots": json.dumps(self._current_conversation["slots"], ensure_ascii=False)
            }

            # Créer le document avec les métadonnées incluses
            document = (
                f"User: {metadata['user_message']}\n"
                f"Assistant: {metadata['ai_message']}\n"
                f"Émotion: {metadata['emotion']}\n"
                f"Intention: {metadata['intent']}\n"
                f"Slots: {metadata['slots']}"
            )

            # Générer un ID unique pour le document
            doc_id = str(uuid.uuid4())
            
            # Ajouter le document à la collection
            try:
                self._collection.add(
                    ids=[doc_id],
                    documents=[document],
                    metadatas=[metadata]
                )
            except Exception as e:
                print(f"Erreur lors de l'ajout du document: {e}")
            
            # Réinitialiser la conversation courante seulement après une sauvegarde réussie
            self._current_conversation = {
                "user_message": None,
                "ai_message": None,
                "emotion": None,
                "intent": None,
                "slots": {
                    "location": None,
                    "food_type": None,
                    "budget": None,
                    "time": None
                }
            }

        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la conversation: {e}")
            
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Récupère tous les messages.
        
        Returns:
            List[Dict[str, Any]]: Liste des messages avec leur rôle et contenu
        """
        return self._messages.copy()
            
    def clear_memory(self) -> None:
        """
        Efface la mémoire et réinitialise ChromaDB.
        """
        try:
            # Supprimer la collection existante
            if self._collection:
                try:
                    # Récupérer tous les IDs de la collection
                    results = self._collection.get()
                    if results and "ids" in results and results["ids"]:
                        # Supprimer tous les documents de la collection
                        self._collection.delete(ids=results["ids"])
                        
                        # Vérifier que la collection est bien vide
                        remaining = self._collection.get()
                        if remaining and "ids" in remaining and remaining["ids"]:
                            print(f"Certains documents n'ont pas été supprimés: {remaining['ids']}")
                            # Essayer de supprimer à nouveau
                            self._collection.delete(ids=remaining["ids"])
                except Exception as e:
                    print(f"Erreur lors de la suppression des documents: {e}")
            
            # Réinitialiser les attributs
            self._messages = []
            self._current_conversation = {
                "user_message": None,
                "ai_message": None,
                "emotion": None,
                "intent": None,
                "slots": {
                    "location": None,
                    "food_type": None,
                    "budget": None,
                    "time": None
                }
            }
            
        except Exception as e:
            print(f"Erreur lors de l'effacement de la mémoire: {e}")
            # En cas d'erreur, essayer de réinitialiser au moins les variables en mémoire
            self._messages = []
            self._current_conversation = {
                "user_message": None,
                "ai_message": None,
                "emotion": None,
                "intent": None,
                "slots": {
                    "location": None,
                    "food_type": None,
                    "budget": None,
                    "time": None
                }
            }
            
    def run(self, prompt: str) -> str:
        """
        Traite un nouveau message.
        
        Args:
            prompt (str): Le message à traiter
            
        Returns:
            str: Message de confirmation
        """
        self.add_message("user", prompt)
        return f"{self._name}: Message enregistré"

    def search_in_conversations(self, query: str) -> Dict[str, Any]:
        """
        Recherche des informations dans les conversations précédentes.
        
        Args:
            query (str): La requête de recherche
            
        Returns:
            Dict[str, Any]: Les informations trouvées
        """
        try:
            if not self._messages:
                return {"found": False, "confidence": "low", "information": ""}
                
            prompt = [
                {"role": "system", "content": """Analysez l'historique des conversations pour trouver des informations pertinentes.
                Répondez au format JSON:
                {
                    "found": true/false,
                    "confidence": "high"/"medium"/"low",
                    "information": "information trouvée"
                }"""},
                {"role": "user", "content": f"Recherchez dans l'historique: {query}\n\nHistorique:\n{json.dumps(self._messages, indent=2)}"}
            ]
            
            response = self._get_llm_response(prompt)
            return json.loads(response)
        except Exception:
            return {"found": False, "confidence": "low", "information": ""}

    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Obtient une réponse de l'API Mistral.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer au LLM
            
        Returns:
            str: La réponse du LLM
        """
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-tiny",
                "messages": prompt,
                "temperature": self._model_config["temperature"],
                "max_tokens": self._model_config["max_tokens"]
            }
            
            response = requests.post(
                f"{self._api_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Erreur API Mistral: {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API Mistral: {e}")
            raise