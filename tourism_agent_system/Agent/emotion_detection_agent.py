# emotion_detection_agent.py
from .base_agent import BaseAgent
import requests
from typing import Dict, Any, List, Optional
import json
import re

class EmotionDetectionAgent(BaseAgent):
    """
    Agent qui détecte les émotions dans les messages
    """
    
    # Liste des émotions valides
    VALID_EMOTIONS = ["joie", "tristesse", "colère", "peur", "surprise", "dégoût", "neutre"]
    
    def __init__(self, name: str = "emotion_detector"):
        super().__init__(name)
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        
    def run(self, message: str) -> List[str]:
        """
        Détecte les émotions dans un message.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            List[str]: Liste des émotions détectées
        """
        try:
            prompt = self._build_prompt(message)
            response = self._get_llm_response(prompt)
            emotions = self._parse_emotions(response)
            
            return emotions if emotions else ["neutre"]
            
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
        system_message = """Vous êtes un expert en analyse d'émotions. Votre tâche est de détecter l'émotion PRINCIPALE dans un message avec précision.

        Voici les émotions possibles avec leurs caractéristiques détaillées :
        
        - joie : bonheur, enthousiasme, excitation, satisfaction, contentement, plaisir, optimisme, fierté
        - tristesse : peine, chagrin, déception, solitude, nostalgie, mélancolie, désespoir, regret
        - colère : frustration, agacement, rage, irritation, indignation, hostilité, mécontentement, ressentiment
        - peur : anxiété, inquiétude, panique, terreur, appréhension, nervosité, stress, alarme
        - surprise : étonnement, stupéfaction, stupéfaction, émerveillement, confusion, déconnexion
        - dégoût : mépris, répulsion, aversion, dédain, écœurement, rejet, détestation
        - neutre : absence d'émotion particulière, équilibre émotionnel, calme, sérénité
        
        Analysez attentivement le message en considérant :
        1. Le contexte global du message
        2. Le ton et le style d'écriture
        3. Les expressions idiomatiques et le langage figuré
        4. Les sous-entendus et implications
        5. La structure et la ponctuation
        6. Les références culturelles
        
        Répondez avec UNE SEULE émotion, en minuscules.
        Exemple: "joie" ou "tristesse" ou "neutre"
        """
            
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Message à analyser : {message}"}
        ]
        
    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Obtient une réponse de l'API Mistral.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer au LLM
            
        Returns:
            str: La réponse du LLM
        """
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-tiny",
                "messages": prompt,
                "temperature": self._model_config["temperature"],
                "max_tokens": self._model_config["max_tokens"]
            }
            
            response = requests.post(
                f"{self._api_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Erreur API Mistral: {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API Mistral: {e}")
            raise
        
    def _parse_emotions(self, response: str) -> List[str]:
        """
        Analyse la réponse du LLM pour extraire les émotions.
        
        Args:
            response (str): La réponse du LLM
        
        Returns:
            List[str]: Liste des émotions détectées
        """
        # Nettoyer la réponse
        emotion = response.strip().lower()
        
        # Vérifier si l'émotion est valide
        if emotion in self.VALID_EMOTIONS:
            return [emotion]
        
        return ["neutre"]
