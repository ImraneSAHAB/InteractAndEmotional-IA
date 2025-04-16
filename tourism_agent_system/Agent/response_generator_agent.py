from Agent import Agent
import ollama
from typing import Dict, Any, List, Optional
import time

class ResponseGeneratorAgent(Agent):
    """
    Agent qui génère des réponses contextuelles basées sur l'intent et les slots fournis
    """
    
    def __init__(self, name: str = "response_generator"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = self._config["model"]
        
    def generate_response(self, slots: Dict[str, Any], intent: str, user_message: str = "") -> str:
        """
        Génère une réponse contextuelle basée sur les informations disponibles.
        
        Args:
            slots (Dict[str, Any]): Les informations disponibles (slots)
            intent (str): L'intention détectée
            user_message (str): Le message de l'utilisateur
            
        Returns:
            str: Une réponse contextuelle et utile
        """
        try:
            # Construire le prompt pour la génération de réponse
            prompt = [
                {"role": "system", "content": f"Vous êtes un Assistant Touristique. {self._goal}"},
                {"role": "user", "content": f"""
                Générer une réponse contextuelle et utile.
                
                Message de l'utilisateur: {user_message}
                Intention détectée: {intent}
                
                Informations disponibles:
                {self._format_slots(slots)}
                
                Instructions spécifiques selon l'intention:
                - Si l'intention est "salutation": Répondez de manière chaleureuse et accueillante, présentez-vous brièvement comme un assistant touristique sans mentionner de nom spécifique. Si l'utilisateur mentionne une ville, adaptez votre réponse en conséquence.
                - Si l'intention est "presentation": Répondez en vous présentant comme un assistant touristique et demandez comment vous pouvez aider, sans mentionner de nom spécifique.
                - Si l'intention est "remerciement": Répondez poliment et encouragez l'utilisateur à continuer la conversation.
                - Si l'intention est "confirmation": Confirmez la compréhension et proposez la suite logique.
                - Si l'intention est "negation": Reconnaissez la négation et demandez des précisions.
                - Si l'intention est "information_generale": Fournissez des informations générales sur le tourisme et demandez des précisions.
                - Si l'intention est "restaurant_search": Suggérez des restaurants en fonction des informations disponibles.
                - Si l'intention est "activity_search": Suggérez des activités en fonction des informations disponibles.
                - Si l'intention est "hotel_booking": Suggérez des hôtels en fonction des informations disponibles.
                - Si l'intention est "unknown": Demandez poliment des précisions sur ce que l'utilisateur souhaite.
                
                La réponse doit être:
                1. Naturelle et conversationnelle
                2. Adaptée à l'intention de l'utilisateur
                3. Pertinente par rapport aux informations disponibles
                4. Directe mais polie
                5. Ne pas mentionner de nom spécifique pour l'assistant
                6. Personnalisée en fonction du contenu du message de l'utilisateur
                
                Répondez uniquement avec la réponse, sans explications supplémentaires.
                """}
            ]

            # Obtenir la réponse du LLM
            response = self._get_llm_response(prompt)
            return response.strip()
            
        except Exception as e:
            return f"Désolé, je n'ai pas pu générer une réponse appropriée. Erreur: {str(e)}"
        
    def _format_slots(self, slots: Dict[str, Any]) -> str:
        """
        Formate les slots pour l'affichage.
        
        Args:
            slots (Dict[str, Any]): Les slots à formater
            
        Returns:
            str: Les slots formatés
        """
        formatted = []
        for key, value in slots.items():
            if value is not None:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted) if formatted else "Aucune information disponible"
        
    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Obtient une réponse du LLM.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer au LLM
            
        Returns:
            str: La réponse du LLM
        """
        try:
            response = self._llm.chat(
                model=self._model_config["name"],
                messages=prompt,
                options={
                    "temperature": self._model_config["temperature"],
                    "max_tokens": self._model_config["max_tokens"]
                }
            )
            
            # Vérifier si la réponse contient le contenu attendu
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                return "Désolé, je n'ai pas pu générer une réponse appropriée. Format de réponse inattendu."
        except Exception as e:
            return "Désolé, je n'ai pas pu générer une réponse appropriée. Veuillez réessayer."
            
    def run(self, prompt: str) -> str:
        """
        Méthode générique pour exécuter l'agent.
        
        Args:
            prompt (str): Le prompt à traiter
            
        Returns:
            str: La réponse générée
        """
        # Cette méthode est requise par la classe de base mais n'est pas utilisée
        # car nous avons des méthodes spécifiques pour générer des réponses
        return "Cette méthode n'est pas utilisée pour cet agent."
        