# emotion_detection_agent.py
from Agent import Agent
import ollama 

class EmotionDetectionAgent(Agent):
    """
    Agent qui détecte l'émotion dans un message utilisateur.
    """
    
    def __init__(self, name: str = "emotion_detector"):
        super().__init__(name)
        self._llm = ollama.Client()
    
    def run(self, prompt: str) -> str:
        """
        Analyse l'émotion du message utilisateur.
        
        Args:
            prompt (str): Le message utilisateur
            
        Returns:
            str: L'émotion détectée
        """
        try:
            response = self._llm.chat(
                model="gemma3",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un expert en analyse émotionnelle. Ton rôle est UNIQUEMENT de détecter l'émotion dans un message.\n\n"
                            "Instructions strictes :\n"
                            "1. Ne réponds JAMAIS avec une explication ou une justification\n"
                            "2. Ne réponds JAMAIS avec une phrase complète\n"
                            "3. Retourne UNIQUEMENT un des mots suivants : 'joie', 'tristesse', 'colère', 'peur', 'surprise', 'dégout', 'neutre'\n\n"
                            "Exemples de réponses correctes :\n"
                            "Message: 'Je suis super content aujourd'hui !' -> Réponse: joie\n"
                            "Message: 'Je suis triste de cette nouvelle' -> Réponse: tristesse\n"
                            "Message: 'Comment ça va ?' -> Réponse: neutre\n\n"
                            "Critères d'analyse :\n"
                            "- Joie : enthousiasme, satisfaction, bonheur, positivité\n"
                            "- Tristesse : mélancolie, déception, peine\n"
                            "- Colère : frustration, irritation, agacement\n"
                            "- Peur : inquiétude, anxiété, appréhension\n"
                            "- Surprise : étonnement, stupéfaction\n"
                            "- Dégout : répulsion, aversion\n"
                            "- Neutre : absence d'émotion claire, questions factuelles"
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            # Nettoyer la réponse pour ne garder que l'émotion
            emotion = response["message"]["content"].strip().lower()
            valid_emotions = ['joie', 'tristesse', 'colère', 'peur', 'surprise', 'dégout', 'neutre']
            return emotion if emotion in valid_emotions else 'neutre'
        except Exception as e:
            print(f"Erreur lors de la détection d'émotion: {e}")
            return "neutre"
