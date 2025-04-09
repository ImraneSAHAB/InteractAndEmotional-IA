from Agent import Agent
from memory_agent import MemoryAgent
import ollama
from typing import Optional

class InteractionalAgent(Agent):
    """
    Agent qui gère les interactions avec l'utilisateur via un LLM.
    """
    
    def __init__(self, name: str = "interactional"):
        super().__init__(name)
        self._llm = ollama()
        self._memory_agent = MemoryAgent()
        
    def run(self, prompt: str) -> str:
        """
        Traite le message de l'utilisateur et génère une réponse via le LLM.
        
        Args:
            prompt (str): Le message de l'utilisateur
            
        Returns:
            str: La réponse générée par le LLM
        """
        # Stocke le message de l'utilisateur
        self._memory_agent.add_message("user", prompt)
        
        # Récupère l'historique des messages pour le contexte
        context = self._memory_agent.get_messages()
        
        # Génère une réponse avec le LLM
        response = self._llm.respond(prompt, context=context)
        
        # Stocke la réponse du LLM
        self._memory_agent.add_message("assistant", response)
        
        return response