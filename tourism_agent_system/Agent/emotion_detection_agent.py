# emotion_detection_agent.py
from Agent import Agent
import ollama 

class EmotionDetectionAgent(Agent):
    """
    Agent qui détecte l’émotion dans un message utilisateur.
    """
    
    def __init__(self, name: str = "emotion_detector"):
        super().__init__(name)
        self._llm = ollama.Client()
    
    def run(self, prompt: str) -> str:
        """
        Analyse l’émotion du message utilisateur.
        
        Args:
            prompt (str): Le message utilisateur
            
        Returns:
            str: L’émotion détectée
        """
        try:
            response = self._llm.chat(
                model="gemma3",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un détecteur d'émotions. Donne uniquement l'état émotionnel global exprimé dans ce message "
                            "parmi : 'joie', 'tristesse', 'colère', 'peur', 'surprise', 'dégout', 'neutre'."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return response["message"]["content"].strip().lower()
        except Exception as e:
            print(f"Erreur lors de la détection d'émotion: {e}")
            return "neutre"
