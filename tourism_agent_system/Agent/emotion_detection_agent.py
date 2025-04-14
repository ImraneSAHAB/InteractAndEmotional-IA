# emotion_detection_agent.py
from Agent import Agent
import ollama
from typing import Dict, Any, List, Optional

class EmotionDetectionAgent(Agent):
    """
    Agent qui détecte les émotions dans les messages
    """
    
    # Liste des émotions valides
    VALID_EMOTIONS = ["joie", "tristesse", "colère", "peur", "surprise", "dégoût", "neutre"]
    
    # Mots-clés associés à chaque émotion
    EMOTION_KEYWORDS = {
        "tristesse": ["seul", "solitude", "isolé", "triste", "malheureux", "déprimé", "chagrin", "peine"],
        "joie": ["heureux", "content", "joie", "heureux", "satisfait", "enthousiaste", "excitant", "génial"],
        "colère": ["fâché", "colère", "énervé", "frustré", "TOUT LE TEMPS", "TOUJOURS", "jamais", "insupportable"],
        "peur": ["peur", "anxieux", "inquiet", "stressé", "paniqué", "terrifié", "appréhension", "angoisse"],
        "surprise": ["surpris", "étonné", "stupéfait", "incroyable", "impensable", "inouï", "extraordinaire"],
        "dégoût": ["dégoûté", "répugnant", "écœuré", "horrible", "détestable", "méprisable", "insupportable"]
    }
    
    def __init__(self, name: str = "emotion_detector"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = {
            "name": "gemma3",
            "temperature": 0.5,
            "max_tokens": 100
        }
        
    def run(self, message: str) -> List[str]:
        """
        Détecte les émotions dans un message.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            List[str]: Liste des émotions détectées
        """
        try:
            # Construire le prompt pour la détection d'émotion
            prompt = self._build_prompt(message)
            
            # Obtenir la réponse du LLM
            response = self._get_llm_response(prompt)
            
            # Analyser la réponse pour extraire les émotions
            emotions = self._parse_emotions(response)
            
            return emotions
            
        except Exception as e:
            print(f"Erreur lors de la détection d'émotion: {e}")
            return ["neutre"]
            
    def _build_prompt(self, message: str) -> List[Dict[str, str]]:
        """
        Construit le prompt pour la détection d'émotion.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            List[Dict[str, str]]: Le prompt formaté
        """
        system_message = """Vous êtes un expert en analyse d'émotions. Votre tâche est de détecter TOUTES les émotions présentes dans un message avec précision et nuance.

        Voici les émotions possibles avec leurs caractéristiques détaillées :
        
        - joie : bonheur, enthousiasme, excitation, satisfaction, contentement, plaisir, optimisme, fierté
        - tristesse : peine, chagrin, déception, solitude, nostalgie, mélancolie, désespoir, regret
        - colère : frustration, agacement, rage, irritation, indignation, hostilité, mécontentement, ressentiment
        - peur : anxiété, inquiétude, panique, terreur, appréhension, nervosité, stress, alarme
        - surprise : étonnement, stupéfaction, stupéfaction, émerveillement, confusion, déconnexion
        - dégoût : mépris, répulsion, aversion, dédain, écœurement, rejet, détestation
        - neutre : absence d'émotion particulière, équilibre émotionnel, calme, sérénité
        
        Analysez attentivement le message en considérant :
        1. Le vocabulaire utilisé (mots émotionnels, intensité)
        2. Le contexte du message
        3. Les expressions et formulations
        4. Les éventuelles émotions secondaires ou contradictoires
        
        Répondez avec la liste des émotions détectées, séparées par des virgules, en minuscules.
        Exemple: "joie, surprise" ou "tristesse, colère" ou "neutre"
        """
            
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Message à analyser : {message}"}
        ]
        
    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Obtient la réponse du LLM.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer
            
        Returns:
            str: La réponse du LLM
        """
        response = self._llm.chat(
            model=self._model_config["name"],
            messages=prompt,
            options={
                "temperature": self._model_config["temperature"],
                "max_tokens": self._model_config["max_tokens"]
            }
        )
        
        return response["message"]["content"]
        
    def _parse_emotions(self, response: str) -> List[str]:
        """
        Analyse la réponse du LLM pour extraire les émotions.
        
        Args:
            response (str): La réponse du LLM
        
        Returns:
            List[str]: Liste des émotions détectées
        """
        # Nettoyer et diviser la réponse
        emotions = [e.strip().lower() for e in response.split(",")]
        
        # Filtrer pour ne garder que les émotions valides
        valid_detected_emotions = [e for e in emotions if e in self.VALID_EMOTIONS]
        
        # Si aucune émotion valide n'est détectée, essayer d'inférer l'émotion à partir du message
        if not valid_detected_emotions:
            detected_emotions = []
            
            # Vérifier les mots-clés pour chaque émotion
            for emotion, keywords in self.EMOTION_KEYWORDS.items():
                if any(keyword.lower() in response.lower() for keyword in keywords):
                    detected_emotions.append(emotion)
            
            # Si aucune émotion n'est détectée, retourner neutre
            if not detected_emotions:
                return ["neutre"]
            
            return detected_emotions
        
        return valid_detected_emotions
