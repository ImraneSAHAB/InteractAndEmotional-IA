from Agent import Agent
from typing import List, Dict, Any
import sys
import os
import chromadb
from datetime import datetime

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MemoryAgent(Agent):
    """
    Agent qui gère la mémoire des messages avec le chat.
    """
    
    def __init__(self, name: str = "memory"):
        super().__init__(name)
        # Initialiser ChromaDB
        self._chroma_client = chromadb.PersistentClient(path="chroma_db")
        self._collection = self._chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialiser les attributs
        self._messages = []
        self._current_conversation = {"user": None, "assistant": None, "emotion": None}
        self._load_messages_from_chromadb()
        
    def _load_messages_from_chromadb(self) -> None:
        """Charge les messages depuis ChromaDB."""
        try:
            results = self._collection.get()
            if not results or "metadatas" not in results:
                return
                
            for metadata in results["metadatas"]:
                for role in ["user", "ai"]:
                    msg_key = f"{role}_message"
                    if msg_key in metadata:
                        self._messages.append({
                            "role": "user" if role == "user" else "assistant",
                            "content": metadata[msg_key]
                        })
        except Exception as e:
            print(f"Erreur de chargement ChromaDB: {e}")
            
    def add_message(self, role: str, content: str, emotion: str = None) -> None:
        """
        Ajoute un message à la mémoire et gère la conversation.
        
        Args:
            role (str): Le rôle de l'émetteur (user/assistant)
            content (str): Le contenu du message
            emotion (str, optional): L'émotion détectée dans le message
        """
        try:
            content = str(content).strip()
            self._messages.append({"role": role, "content": content})
            
            if role in ["user", "assistant"]:
                self._current_conversation[role] = content
                if role == "user" and emotion:
                    self._current_conversation["emotion"] = emotion
                if role == "assistant" and self._current_conversation["user"]:
                    self._save_conversation()
                    self._current_conversation = {"user": None, "assistant": None, "emotion": None}
                    
        except Exception as e:
            print(f"Erreur d'ajout de message: {e}")
            
    def _save_conversation(self) -> None:
        """Sauvegarde la conversation complète dans ChromaDB."""
        try:
            conv = self._current_conversation
            if not conv["user"] or not conv["assistant"]:
                return
                
            current_ids = self._collection.get()
            next_id = len(current_ids.get('ids', [])) + 1 if current_ids.get('ids') else 1
            
            # Créer le document avec l'émotion
            document_text = f"Utilisateur: {conv['user']}\nAssistant: {conv['assistant']}\nÉmotion: {conv.get('emotion', 'neutre')}"
            
            self._collection.add(
                ids=[f"conv_{next_id}"],
                documents=[document_text],
                metadatas=[{
                    "type": "conversation",
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": f"conv_{next_id}",
                    "user_message": conv["user"],
                    "ai_message": conv["assistant"],
                    "emotion": conv.get("emotion", "neutre"),
                    "agent_name": self._name,
                    "agent_role": self._role
                }]
            )
        except Exception as e:
            print(f"Erreur de sauvegarde ChromaDB: {e}")
            
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
        self._current_conversation = {"user": None, "assistant": None}
        
        # Réinitialiser ChromaDB
        try:
            self._collection.delete()
            self._collection = self._chroma_client.get_or_create_collection(
                name="conversations",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Erreur de réinitialisation ChromaDB: {e}")
            
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