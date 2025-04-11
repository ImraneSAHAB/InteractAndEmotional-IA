from Agent import Agent
from orchestrator import AgentOrchestrator
from typing import Dict, Any
import ollama
from typing import Optional
import sys
import os

class InteractionalAgent(Agent):
    """
    Agent qui gère les interactions avec l'utilisateur
    """
    
    def __init__(self, orchestrator: AgentOrchestrator = None, name: str = "interactional"):
        super().__init__(name)
        self._orchestrator = orchestrator or AgentOrchestrator()
        
    def run(self) -> None:
        """
        Lance la boucle principale d'interaction avec l'utilisateur.
        """
        print("\nBienvenue ! (Pour quitter, appuyez sur 'Entrée')")
        
        while True:
            try:
                user_input = input("\nVous: ").strip()
                
                if not user_input:
                    print("\nAu revoir ! À bientôt !")
                    break
                
                # L'orchestrator gère tout le traitement, y compris la détection d'émotion
                result = self._orchestrator.process_message(user_input)
                
                if result["success"]:
                    print(f"\nAssistant: {result['response']}")
                else:
                    print(f"\nErreur: {result.get('error', 'Erreur inconnue')}")
                    
            except KeyboardInterrupt:
                print("\nAu revoir ! À bientôt !")
                break
            except Exception as e:
                print(f"\nErreur: {e}")
        
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