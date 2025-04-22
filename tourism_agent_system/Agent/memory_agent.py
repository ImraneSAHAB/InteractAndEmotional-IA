from Agent import Agent
from typing import List, Dict, Any
import sys
import os
import chromadb
from datetime import datetime
import json
import ollama
import time
import uuid

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MemoryAgent(Agent):
    """
    Agent qui gère la mémoire des messages avec le chat.
    """
    
    def __init__(self, name: str = "memory"):
        super().__init__(name)
        # Initialiser le LLM avec la configuration
        self._llm = ollama.Client()
        self._model_config = self._config["model"]
        
        # Créer le chemin absolu pour ChromaDB dans tourism_agent_system
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        chroma_db_path = os.path.join(base_path, "tourism_agent_system", "chroma_db")
        if not os.path.exists(chroma_db_path):
            os.makedirs(chroma_db_path)
            
        # Initialiser ChromaDB avec le chemin absolu
        self._chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialiser les attributs
        self._messages = []
        self._current_conversation = {"user": None, "assistant": None, "emotion": None, "slots": None, "intent": None}
        self._current_slots = {
            "location": None,
            "food_type": None,
            "budget": None,
            "time": None
        }
        self._load_messages_from_chromadb()
        
    def _load_messages_from_chromadb(self) -> None:
        """
        Charge les messages depuis ChromaDB.
        """
        try:
            # Initialiser les slots actuels
            self._current_slots = {}

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
        
        # Créer le message
        message = {
            "role": role,
            "content": content
        }

        # Ajouter l'émotion si disponible
        if emotion:
            message["emotion"] = emotion

        # Ajouter l'intent si disponible
        if intent:
            message["intent"] = intent

        # Ajouter le message à la liste
        self._messages.append(message)

        # Mettre à jour la conversation courante
        if role == "user":
            self._current_conversation["user_message"] = content
            self._current_conversation["emotion"] = emotion if emotion else ""
            self._current_conversation["intent"] = intent if intent else ""
            
            # Mettre à jour les slots actuels avec les nouvelles valeurs non-nulles
            if slots is not None:
                for key, value in slots.items():
                    if value is not None:
                        self._current_slots[key] = value
            
            # Sauvegarder les slots mis à jour dans la conversation
            self._current_conversation["slots"] = json.dumps(self._current_slots)
            
        elif role == "assistant":
            self._current_conversation["ai_message"] = content

        # Si nous avons un message utilisateur et un message assistant, sauvegarder la conversation
        if (self._current_conversation.get("user_message") is not None and 
            self._current_conversation.get("ai_message") is not None):
            self._save_conversation()
            
    def _save_conversation(self) -> None:
        """
        Sauvegarde la conversation courante dans ChromaDB.
        """
        try:
            # Vérifier que nous avons les messages nécessaires
            if not self._current_conversation.get("user_message") or not self._current_conversation.get("ai_message"):
                return

            # Préparer les métadonnées en convertissant les None en chaînes vides
            metadata = {
                "user_message": self._current_conversation["user_message"],
                "ai_message": self._current_conversation["ai_message"],
                "emotion": self._current_conversation.get("emotion", ""),
                "intent": self._current_conversation.get("intent", ""),
                "slots": self._current_conversation.get("slots", "{}")
            }

            # S'assurer que les slots sont une chaîne JSON valide
            if metadata["slots"] is None:
                metadata["slots"] = "{}"
            elif not isinstance(metadata["slots"], str):
                metadata["slots"] = json.dumps(metadata["slots"])

            # Créer le document avec les métadonnées incluses
            document = (
                f"User: {metadata['user_message']}\n"
                f"Assistant: {metadata['ai_message']}\n"
                f"Émotion: {metadata['emotion']}\n"
                f"Intention: {metadata['intent']}\n"
                f"Slots: {metadata['slots']}"
            )

            # Supprimer la collection existante
            try:
                self._chroma_client.delete_collection("conversations")
            except:
                pass
            
            # Recréer la collection
            self._collection = self._chroma_client.get_or_create_collection(
                name="conversations",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Ajouter le document à la collection
            try:
                self._collection.add(
                    ids=["0"],
                    documents=[document],
                    metadatas=[metadata]
                )
            except:
                pass
            
            # Réinitialiser la conversation courante seulement après une sauvegarde réussie
            self._current_conversation = {
                "user_message": None,
                "ai_message": None,
                "emotion": None,
                "intent": None,
                "slots": None
            }

        except:
            pass
            
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Récupère tous les messages.
        
        Returns:
            List[Dict[str, Any]]: Liste des messages avec leur rôle et contenu
        """
        return self._messages.copy()
            
    def clear_memory(self) -> None:
        """Efface la mémoire et réinitialise ChromaDB."""
        # Réinitialiser les attributs
        self._messages = []
        self._current_conversation = {"user": None, "assistant": None, "emotion": None, "slots": None, "intent": None}
        self._current_slots = {
            "location": None,
            "food_type": None,
            "budget": None,
            "time": None
        }
        
        # Réinitialiser ChromaDB
        try:
            # Récupérer tous les IDs existants
            results = self._collection.get()
            if results and "ids" in results and results["ids"]:
                # Supprimer tous les documents existants
                self._collection.delete(ids=results["ids"])
            
            # Supprimer la collection
            try:
                self._chroma_client.delete_collection("conversations")
            except Exception as e:
                print(f"Erreur lors de la suppression de la collection: {e}")
            
            # Recréer une collection vide
            self._collection = self._chroma_client.create_collection(
                name="conversations",
                metadata={"hnsw:space": "cosine"}
            )
            
        except Exception as e:
            print(f"Erreur lors de l'effacement de la mémoire: {e}")
            
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
            
            # Construire le prompt pour le LLM
            prompt = [
                {"role": "system", "content": "Vous êtes un assistant qui analyse les conversations précédentes pour trouver des informations spécifiques."},
                {"role": "user", "content": f"""
                Recherchez dans les conversations suivantes des informations concernant: {query}
                
                Conversations:
                {self._format_conversations_for_search()}
                
                Répondez uniquement avec les informations trouvées au format JSON:
                {{
                    "found": true/false,
                    "information": "l'information trouvée",
                    "confidence": "high/medium/low"
                }}
                
                Si aucune information n'est trouvée, répondez avec:
                {{
                    "found": false,
                    "information": "",
                    "confidence": "low"
                }}
                """}
            ]
            
            # Obtenir la réponse du LLM
            response = self._get_llm_response(prompt)
            
            # Parser la réponse
            try:
                # Nettoyer la réponse pour enlever les backticks et autres caractères non-JSON
                cleaned_response = response.strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()
                
                # Vérifier si la réponse est au format JSON
                if cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                    result = json.loads(cleaned_response)
                    return result
                else:
                    # Si ce n'est pas du JSON, créer un résultat par défaut
                    print(f"La réponse n'est pas au format JSON: {cleaned_response}")
                    
                    # Essayer d'extraire des informations utiles de la réponse
                    if "information" in cleaned_response.lower():
                        # Chercher des informations entre guillemets
                        import re
                        info_matches = re.findall(r'"([^"]*)"', cleaned_response)
                        if info_matches:
                            return {
                                "found": True,
                                "information": info_matches[0],
                                "confidence": "low"
                            }
                    
                    return {"found": False, "information": "", "confidence": "low"}
            except json.JSONDecodeError:
                print(f"Erreur lors du décodage de la réponse: {response}")
                return {"found": False, "information": "", "confidence": "low"}
                
        except Exception as e:
            print(f"Erreur lors de la recherche dans les conversations: {e}")
            return {"found": False, "information": "", "confidence": "low"}
            
    def _format_conversations_for_search(self) -> str:
        """
        Formate les conversations pour la recherche.
        
        Returns:
            str: Les conversations formatées
        """
        formatted = ""
        for i, message in enumerate(self._messages):
            role = "Utilisateur" if message["role"] == "user" else "Assistant"
            content = message["content"]
            formatted += f"{role}: {content}\n"
            
            # Ajouter les métadonnées si disponibles
            if "emotion" in message:
                formatted += f"Émotion: {message['emotion']}\n"
            if "intent" in message:
                formatted += f"Intention: {message['intent']}\n"
                
            # Ajouter un séparateur entre les messages
            if i < len(self._messages) - 1:
                formatted += "---\n"
                
        return formatted
        
    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Obtient une réponse du LLM.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer au LLM
            
        Returns:
            str: La réponse du LLM
        """
        try:
            response = self._llm.chat(
                model=self._model_config["name"],
                messages=prompt,
                options={
                    "temperature": self._model_config["temperature"],
                    "max_tokens": self._model_config["max_tokens"]
                }
            )
            
            # Vérifier si la réponse contient le contenu attendu
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                return "Désolé, je n'ai pas pu générer une réponse appropriée. Format de réponse inattendu."
        except Exception as e:
            return "Désolé, je n'ai pas pu générer une réponse appropriée. Veuillez réessayer."

    def _generate_unique_id(self) -> str:
        """
        Génère un ID unique qui n'existe pas déjà dans la collection.
        
        Returns:
            str: Un ID unique
        """
        # Récupérer tous les IDs existants
        try:
            results = self._collection.get()
            existing_ids = set(results.get("ids", [])) if results else set()
            
            # Générer un nouvel ID jusqu'à ce qu'il soit unique
            while True:
                new_id = str(uuid.uuid4())
                if new_id not in existing_ids:
                    return new_id
        except Exception as e:
            # En cas d'erreur, retourner simplement un UUID
            print(f"Erreur lors de la génération d'un ID unique: {e}")
            return str(uuid.uuid4())