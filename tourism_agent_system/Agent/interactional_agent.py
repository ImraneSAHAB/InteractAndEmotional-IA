from Agent import Agent
from orchestrator import AgentOrchestrator
from typing import Dict, Any

class InteractionalAgent(Agent):
    """
    Agent qui gère les interactions avec l'utilisateur
    """
    
    def __init__(self, name: str = "interactional"):
        super().__init__(name)
        # Initialiser l'orchestrateur
        self._orchestrator = AgentOrchestrator()
        
    def run(self) -> None:
        """
        Lance la boucle principale d'interaction avec l'utilisateur.
        """
        print("Bienvenue !")
        print("(Pour quitter, appuyez sur 'Entrée')")
        
        while True:
            try:
                # Obtenir l'entrée utilisateur
                user_input = input("\nVous: ").strip()
                
                # Vérifier si l'utilisateur veut quitter
                if user_input.lower() in [""]:
                    print("\nAu revoir ! À bientôt !")
                    break
                
                # Ignorer les messages vides
                if not user_input:
                    continue
                
                # Traiter le message via l'orchestrateur
                result = self._orchestrator.process_message(user_input)
                
                # Afficher la réponse
                if result["success"]:
                    print(f"\nAssistant: {result['response']}")
                else:
                    print(f"\nDésolé, une erreur est survenue: {result.get('error', 'Erreur inconnue')}")
                    
            except KeyboardInterrupt:
                print("\n\nAu revoir ! À bientôt !")
                break
            except Exception as e:
                print(f"\nUne erreur inattendue est survenue: {e}")
                print("N'hésitez pas à réessayer.")
        
    def get_conversation_history(self) -> list:
        """
        Récupère l'historique des conversations depuis l'orchestrateur.
        
        Returns:
            list: Liste des messages
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