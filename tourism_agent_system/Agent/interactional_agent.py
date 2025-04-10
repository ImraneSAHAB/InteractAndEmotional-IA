from Agent import Agent
from memory_agent import MemoryAgent
from emotion_detection_agent import EmotionDetectionAgent
import ollama
from typing import Optional
import sys
import os
import time

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.fill_db import update_database

class InteractionalAgent(Agent):
    """
    Agent qui gère les interactions avec l'utilisateur via un LLM.
    """
    
    def __init__(self, name: str = "interactional"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._memory_agent = MemoryAgent()
        self._emotion_agent = EmotionDetectionAgent()

        
    def _print_typing_effect(self, text: str, delay: float = 0.02) -> None:
        """
        Affiche le texte caractère par caractère avec un effet de dactylographie.
        
        Args:
            text (str): Le texte à afficher
            delay (float): Le délai entre chaque caractère en secondes
        """
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()  # Nouvelle ligne à la fin
        
    def run(self, prompt: str) -> str:
        """
        Traite le message de l'utilisateur et génère une réponse via le LLM.
        
        Args:
            prompt (str): Le message de l'utilisateur
            
        Returns:
            str: La réponse générée par le LLM
        """
        try:
            # Stocke le message de l'utilisateur
            self._memory_agent.add_message("user", prompt)
            
            # Détection de l’émotion dans le message
            emotion = self._emotion_agent.run(prompt)
            
            # Récupère l'historique des messages pour le contexte
            context = self._memory_agent.get_messages()
            
            # Génère une réponse avec le LLM
            response = self._llm.chat(
                model="gemma3",
                messages=[
                    {"role": "system", "content": "Vous êtes un assistant touristique. Répondez aux questions de manière précise et utile."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Récupère la réponse
            response_text = response["message"]["content"]
            
            # Stocke la réponse
            self._memory_agent.add_message("assistant", response_text)
            
            # Met à jour la base de données après l'interaction complète
            update_database()
            
            # Affiche la réponse avec l'effet de dactylographie
            self._print_typing_effect(response_text)
            
            return response_text
        except Exception as e:
            print(f"Erreur lors de l'interaction: {e}")
            return "Désolé, une erreur est survenue lors du traitement de votre message."

if __name__ == "__main__":
    # Créer une instance de l'agent
    agent = InteractionalAgent()
    
    print("Bienvenue !")
    print("Tapez 'Entrée' pour quitter.")
    print("-" * 50)
    
    while True:
        # Demander l'entrée de l'utilisateur
        user_input = input("\nVous: ")
        
        # Vérifier si l'utilisateur veut quitter
        if user_input.lower() == "":
            print("\nAu revoir!")
            break
            
        # Traiter la réponse
        try:
            response = agent.run(user_input)
            print("\nAssistant:", response)
        except Exception as e:
            print(f"\nErreur: {e}")