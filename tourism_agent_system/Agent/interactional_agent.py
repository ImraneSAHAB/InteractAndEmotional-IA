from Agent import Agent
from orchestrator import AgentOrchestrator
from typing import Dict, Any
import time

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
        print("\nBienvenue dans le système d'agent touristique !")
        print("Tapez 'quit' pour quitter ou 'clear' pour effacer l'historique.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nVous: ").strip()
                
                if not user_input or user_input.lower() == 'quit':
                    print("\nAu revoir ! À bientôt !")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_memory()
                    print("\nHistorique effacé.")
                    continue
                
                # L'orchestrator gère tout le traitement
                result = self._orchestrator.process_message(user_input)
                
                if result["success"]:
                    # Afficher la réponse de l'assistant mot par mot
                    print("\nAssistant: ", end="", flush=True)
                    self._print_typing_effect(result['response'])
                else:
                    print(f"\nErreur: {result.get('error', 'Erreur inconnue')}")
                    
            except KeyboardInterrupt:
                print("\nAu revoir ! À bientôt !")
                break
            except Exception as e:
                print(f"\nErreur: {e}")
    
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