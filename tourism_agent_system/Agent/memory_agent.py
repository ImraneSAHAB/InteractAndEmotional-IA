from .base_agent import BaseAgent
from typing import List, Dict, Any
import sys
import os
import chromadb
from datetime import datetime
import json
import requests
import uuid
import hashlib
import logging

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MemoryAgent(BaseAgent):
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
            
        # Initialiser ChromaDB avec le chemin absolu et désactiver les logs
        logging.getLogger('chromadb').setLevel(logging.WARNING)
        self._chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialiser les attributs
        self._messages = []
        self._current_conversation = {
            "user_message": None,
            "ai_message": None,
            "emotion": None,
            "intent": None,
            "slots": {}  # Slots dynamiques
        }
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
            
    def _extract_intent_and_slots(self, message: str) -> tuple[str, Dict[str, Any]]:
        """
        Extrait l'intent et les slots d'un message en utilisant le LLM.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            tuple[str, Dict[str, Any]]: L'intent et les slots extraits
        """
        try:
            prompt = [
                {"role": "system", "content": """Analysez le message et extrayez :
                1. L'intention principale (intent)
                2. Les informations pertinentes (slots)
                
                Répondez au format JSON :
                {
                    "intent": "l'intention détectée",
                    "slots": {
                        "nom_du_slot": "valeur",
                        ...
                    }
                }
                
                Les intents possibles incluent mais ne sont pas limités à :
                - recherche_restaurant
                - recherche_activite
                - reservation_hotel
                - salutation
                - presentation
                - remerciement
                - confirmation
                - negation
                - information_generale
                - demande_information
                
                Les slots peuvent être n'importe quelle information pertinente extraite du message.
                """},
                {"role": "user", "content": f"Message à analyser : {message}"}
            ]
            
            response = self._llm.chat(
                model=self._model_config["name"],
                messages=prompt,
                options={"temperature": 0.1, "max_tokens": 200}
            )
            
            result = json.loads(response["message"]["content"])
            return result.get("intent", ""), result.get("slots", {})
            
        except Exception as e:
            return "", {}

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
            # Si c'est un message utilisateur, extraire l'intent et les slots
            if role == "user" and (not intent or not slots):
                extracted_intent, extracted_slots = self._extract_intent_and_slots(content)
                intent = intent or extracted_intent
                if not slots:
                    slots = extracted_slots
                else:
                    slots.update(extracted_slots)

            message = {
                "role": role,
                "content": content,
                "emotion": emotion or "",
                "intent": intent or ""
            }
            self._messages.append(message)
            
            if role == "user":
                self._current_conversation["user_message"] = content
                self._current_conversation["emotion"] = emotion
                self._current_conversation["intent"] = intent
                if slots:
                    # Mettre à jour les slots dynamiquement
                    if not self._current_conversation["slots"]:
                        self._current_conversation["slots"] = {}
                    self._current_conversation["slots"].update({k: v for k, v in slots.items() if v})
            elif role == "assistant":
                self._current_conversation["ai_message"] = content
            
            # Sauvegarder la conversation si nous avons à la fois le message utilisateur et la réponse de l'assistant
            if self._current_conversation["user_message"] and self._current_conversation["ai_message"]:
                self._save_conversation()
                # Réinitialiser la conversation courante après la sauvegarde
                self._current_conversation = {
                    "user_message": None,
                    "ai_message": None,
                    "emotion": None,
                    "intent": None,
                    "slots": {}
                }
                
        except Exception as e:
            raise Exception(f"Erreur lors de l'ajout du message: {str(e)}")
            
    def _save_conversation(self) -> None:
        """
        Sauvegarde la conversation courante dans ChromaDB.
        """
        try:
            # Vérifier que nous avons les messages nécessaires
            if not self._current_conversation.get("user_message") or not self._current_conversation.get("ai_message"):
                return

            # Vérifier et nettoyer l'émotion
            emotion = self._current_conversation.get("emotion", "")
            if not emotion or len(emotion) < 2:  # Éviter les émotions trop courtes
                emotion = "neutre"

            # Préparer les métadonnées
            metadata = {
                "user_message": self._current_conversation["user_message"],
                "ai_message": self._current_conversation["ai_message"],
                "emotion": str(emotion),
                "intent": str(self._current_conversation.get("intent", "")),
                "slots": json.dumps(self._current_conversation.get("slots", {}), ensure_ascii=False),
                "timestamp": datetime.now().isoformat()
            }

            # Créer le document avec les métadonnées incluses
            document = (
                f"User: {metadata['user_message']}\n"
                f"Assistant: {metadata['ai_message']}\n"
                f"Émotion: {metadata['emotion']}\n"
                f"Intention: {metadata['intent']}\n"
                f"Slots: {metadata['slots']}"
            )

            # Générer un ID unique basé sur le timestamp et un UUID
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_id = f"conv_{timestamp}_{uuid.uuid4().hex[:8]}"
            
            # Ajouter le document à la collection
            try:
                self._collection.add(
                    ids=[unique_id],
                    documents=[document],
                    metadatas=[metadata]
                )
            except Exception as e:
                raise Exception(f"Erreur lors de l'ajout du document: {str(e)}")
            
            # Réinitialiser la conversation courante seulement après une sauvegarde réussie
            self._current_conversation = {
                "user_message": None,
                "ai_message": None,
                "emotion": None,
                "intent": None,
                "slots": {}  # Slots dynamiques
            }

        except Exception as e:
            raise Exception(f"Erreur lors de la sauvegarde de la conversation: {str(e)}")
            
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
                            # Essayer de supprimer à nouveau
                            self._collection.delete(ids=remaining["ids"])
                except Exception as e:
                    raise Exception(f"Erreur lors de la suppression des documents: {str(e)}")
            
            # Réinitialiser les attributs
            self._messages = []
            self._current_conversation = {
                "user_message": None,
                "ai_message": None,
                "emotion": None,
                "intent": None,
                "slots": {}  # Slots dynamiques
            }
            
        except Exception as e:
            raise Exception(f"Erreur lors de l'effacement de la mémoire: {str(e)}")
            
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