import logging
from typing import Optional
from .base_agent import BaseAgent
from .orchestrator import AgentOrchestrator
from typing import Dict, Any
import time

class InteractionalAgent(BaseAgent):
    """
    Agent responsable de la gestion des interactions avec l'utilisateur.
    """

    def __init__(self, name: str = "interaction"):
        super().__init__(name)
        self._interaction_config = self._config.get("interaction", {})
        self._conversation_state = "initial"

    def interact(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Gère l'interaction avec l'utilisateur.
        
        Args:
            message (str): Le message de l'utilisateur
            context (Dict[str, Any], optional): Le contexte de la conversation
            
        Returns:
            Dict[str, Any]: Résultat de l'interaction
        """
        try:
            # Mise à jour du contexte
            if context is None:
                context = {}
            
            # Analyse du message pour déterminer l'état de la conversation
            new_state = self._analyze_conversation_state(message, context)
            self._conversation_state = new_state
            
            # Génération de la réponse appropriée
            response = self._generate_response(message, new_state, context)
            
            return {
                "success": True,
                "state": new_state,
                "response": response,
                "context": context
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "state": self._conversation_state,
                "response": "Désolé, je n'ai pas pu traiter votre message correctement.",
                "context": context or {}
            }

    def _analyze_conversation_state(self, message: str, context: Dict[str, Any]) -> str:
        """
        Analyse le message pour déterminer l'état de la conversation.
        """
        message = message.lower()
        
        # États de conversation possibles
        if "bonjour" in message or "salut" in message:
            return "greeting"
        elif "au revoir" in message or "bye" in message:
            return "farewell"
        elif "merci" in message:
            return "thanks"
        elif "?" in message:
            return "question"
        elif context.get("needs_clarification"):
            return "clarification"
        else:
            return "conversation"

    def _generate_response(self, message: str, state: str, context: Dict[str, Any]) -> str:
        """
        Génère une réponse appropriée en fonction de l'état de la conversation.
        """
        responses = {
            "greeting": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
            "farewell": "Au revoir ! N'hésitez pas à revenir si vous avez d'autres questions.",
            "thanks": "Je vous en prie ! Y a-t-il autre chose que je puisse faire pour vous ?",
            "question": "Je vais chercher cette information pour vous.",
            "clarification": "Pourriez-vous préciser votre demande ?",
            "conversation": "Je comprends. Comment puis-je vous aider davantage ?"
        }
        
        return responses.get(state, "Je suis là pour vous aider.")

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Méthode principale pour exécuter l'agent.
        
        Args:
            prompt (str): Le message de l'utilisateur
            
        Returns:
            Dict[str, Any]: Résultat de l'interaction
        """
        return self.interact(prompt)

    def _print_typing_effect(self, text: str, delay: float = 0.03) -> None:
        """
        Affiche le texte mot par mot avec un effet de frappe.
        
        Args:
            text (str): Le texte à afficher
            delay (float): Le délai entre chaque mot en secondes
        """
        words = text.split()
        for i, word in enumerate(words):
            print(word, end="", flush=True)
            if i < len(words) - 1:
                print(" ", end="", flush=True)
            time.sleep(delay)
        print()  # Nouvelle ligne à la fin
        
    def get_conversation_history(self) -> list:
        """
        Récupère l'historique des conversations depuis l'orchestrateur.
        """
        return self._orchestrator.get_conversation_history()
        
    def clear_memory(self) -> None:
        """
        Efface la mémoire via l'orchestrateur.
        """
        self._orchestrator.clear_memory()

if __name__ == "__main__":
    agent = InteractionalAgent()
    agent.run()
