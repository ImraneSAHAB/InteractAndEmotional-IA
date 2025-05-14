# emotion_detection_agent.py
from .base_agent import BaseAgent
import requests
from typing import Dict, Any, List, Optional
import json
import re

class EmotionDetectionAgent(BaseAgent):
    """
    Agent spécialisé dans la détection des émotions dans les messages.
    Utilise Mistral pour analyser le sentiment et l'émotion de manière dynamique.
    """
    
    def __init__(self, name: str = "emotion"):
        super().__init__(name)
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        
    def run(self, message: str) -> List[str]:
        """
        Implémentation de la méthode run de BaseAgent.
        Retourne une liste d'émotions détectées.
        """
        result = self.detect_emotion(message)
        return [result["emotion"]]

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
            elif response.status_code == 429:  # Rate limit
                return "neutre"  # Retourner une émotion neutre en cas de rate limit
            else:
                raise Exception(f"Erreur API Mistral: {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API Mistral: {e}")
            return "neutre"  # Retourner une émotion neutre en cas d'erreur

    def detect_emotion(self, message: str) -> Dict[str, str]:
        """
        Détecte l'émotion principale dans un message de manière dynamique.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            Dict[str, str]: Dictionnaire contenant l'émotion détectée et sa confiance
        """
        try:
            prompt = [
                {"role": "system", "content": """Tu es un expert en analyse émotionnelle.
                Analyse le message et détermine l'émotion principale exprimée.
                Tu peux identifier n'importe quelle émotion humaine, pas seulement une liste prédéfinie.
                
                Réponds au format JSON avec deux champs :
                {
                    "emotion": "nom de l'émotion en minuscules",
                    "confidence": "high/medium/low"
                }
                
                L'émotion doit être un mot simple et clair en français."""},
                {"role": "user", "content": message}
            ]
            
            response = self._get_llm_response(prompt).strip()
            
            try:
                result = json.loads(response)
                return {
                    "emotion": result.get("emotion", "neutre").lower(),
                    "confidence": result.get("confidence", "medium")
                }
            except json.JSONDecodeError:
                # Si le format JSON n'est pas respecté, on extrait l'émotion du texte brut
                emotion = response.lower().strip()
                return {
                    "emotion": emotion if emotion else "neutre",
                    "confidence": "medium"
                }
            
        except Exception as e:
            print(f"Erreur lors de la détection d'émotion: {e}")
            return {
                "emotion": "neutre",
                "confidence": "low"
            }

    def _build_prompt(self, message: str) -> List[Dict[str, str]]:
        """
        Construit le prompt pour la détection d'émotion.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            List[Dict[str, str]]: Le prompt formaté
        """
        system_message = """Vous êtes un expert en analyse d'émotions. Votre tâche est de détecter l'émotion PRINCIPALE dans un message avec précision.

        Analysez attentivement le message en considérant :
        1. Le contexte global du message
        2. Le ton et le style d'écriture
        3. Les expressions idiomatiques et le langage figuré
        4. Les sous-entendus et implications
        5. La structure et la ponctuation
        6. Les références culturelles
        
        Répondez au format JSON avec deux champs :
        {
            "emotion": "nom de l'émotion en minuscules",
            "confidence": "high/medium/low"
        }
        
        L'émotion doit être un mot simple et clair en français qui décrit le mieux l'état émotionnel exprimé.
        """
            
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Message à analyser : {message}"}
        ]
