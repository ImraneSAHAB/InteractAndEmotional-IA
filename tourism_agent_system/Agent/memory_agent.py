from Agent import Agent
from typing import List, Dict, Any
import json
import os

class MemoryAgent(Agent):
    
    """
    Agent qui gère la mémoire des messages avec le chat.
    """
    
    def __init__(self, name: str = "memory"):
        super().__init__(name)
        self._memory_file = "memory/memory_store.json"
        self._messages: List[Dict[str, Any]] = self._load_messages()
        
    def _load_messages(self) -> List[Dict[str, Any]]:
        """
        Charge les messages depuis le fichier de stockage.
        
        Returns:
            List[Dict[str, Any]]: Liste des messages chargés
        """
        try:
            if os.path.exists(self._memory_file):
                with open(self._memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except json.JSONDecodeError:
            print(f"Erreur lors de la lecture du fichier {self._memory_file}")
            return []
            
    def _save_messages(self) -> None:
        """
        Sauvegarde les messages dans le fichier de stockage.
        """
        try:
            with open(self._memory_file, "w", encoding="utf-8") as f:
                json.dump(self._messages, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des messages: {e}")
        
    def add_message(self, role: str, content: str) -> None:
        """
        Ajoute un nouveau message à la mémoire et le sauvegarde.
        
        Args:
            role (str): Le rôle de l'émetteur du message ('user' ou 'assistant')
            content (str): Le contenu du message
        """
        self._messages.append({
            "role": role,
            "content": content
        })
        self._save_messages()
        
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Récupère l'historique complet des messages.
        
        Returns:
            List[Dict[str, Any]]: Liste des messages avec leur rôle et contenu
        """
        return self._messages.copy()
    
    def clear_memory(self) -> None:
        """
        Efface toute la mémoire des messages et le fichier de stockage.
        """
        self._messages = []
        self._save_messages()
        
    def run(self, prompt: str) -> str:
        """
        Méthode principale de l'agent qui traite les messages.
        
        Args:
            prompt (str): Le message à traiter
            
        Returns:
            str: Un message de confirmation
        """
        # Le MemoryAgent n'a pas besoin de traiter le contenu du message
        # Il se contente de le stocker
        self.add_message("user", prompt)
        return f"{self.role}: Message enregistré dans la mémoire"
