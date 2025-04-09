from Agent import Agent
from typing import List, Dict, Any
import json

class MemoryAgent(Agent):
    
    """
    Agent qui gère la mémoire des messages avec le chat.
    """
    
    def __init__(self, name: str = "memory"):
        super().__init__(name)
        self._messages: List[Dict[str, Any]] = []
        
    def add_message(self, role: str, content: str) -> None:
        """
        Ajoute un nouveau message à la mémoire.
        
        Args:
            role (str): Le rôle de l'émetteur du message ('user' ou 'assistant')
            content (str): Le contenu du message
        """
        self._messages.append({
            "role": role,
            "content": content
        })
        
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Récupère l'historique complet des messages.
        
        Returns:
            List[Dict[str, Any]]: Liste des messages avec leur rôle et contenu
        """
        return self._messages.copy()
    
    def clear_memory(self) -> None:
        """
        Efface toute la mémoire des messages.
        """
        self._messages = []
        
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
