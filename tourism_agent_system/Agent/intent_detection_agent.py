# intent_detection_agent.py
from .base_agent import BaseAgent
import requests
from typing import Dict, Any, List, Optional
import re
import json

class IntentDetectionAgent(BaseAgent):
    """
    Agent responsable de la détection des intentions et des slots dans les messages utilisateur.
    Utilise le LLM pour une détection dynamique des intentions et des slots.
    """

    def __init__(self, name: str = "intent"):
        super().__init__(name)
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        self._intent_config = self._config.get("intent", {})

    def run(self, message: str) -> Dict[str, Any]:
        """
        Analyse un message pour détecter l'intention et extraire les slots.

        Args:
            message (str): Le message à analyser

        Returns:
            Dict[str, Any]: Dictionnaire contenant l'intention, les slots et la confiance
        """
        try:
            prompt = self._build_prompt(message)
            response = self._get_llm_response(prompt)
            result = self._parse_response(response)
            
            return {
                "intent": result.get("intent", "unknown"),
                "slots": result.get("slots", {}),
                "confidence": result.get("confidence", "medium")
            }
            
        except Exception as e:
            print(f"Erreur lors de la détection d'intention: {e}")
            return {
                "intent": "unknown",
                "slots": {},
                "confidence": "low"
            }

    def _build_prompt(self, message: str) -> List[Dict[str, str]]:
        """
        Construit le prompt pour l'analyse d'intention et de slots.

        Args:
            message (str): Le message à analyser

        Returns:
            List[Dict[str, str]]: Prompt formaté
        """
        system_prompt = """Vous êtes un expert en analyse du langage naturel. Votre tâche est d'analyser un message utilisateur et d'en extraire :
1. L'intention principale de l'utilisateur
2. Les informations pertinentes (slots) mentionnées dans le message

Instructions :
1. Identifiez l'intention principale de l'utilisateur (par exemple : recherche, réservation, information, question, etc.)
2. Extrayez toutes les informations pertinentes mentionnées dans le message
3. Soyez attentif au contexte et aux sous-entendus
4. Identifiez les entités nommées (lieux, dates, prix, etc.)
5. Détectez les préférences et contraintes exprimées

Répondez au format JSON suivant :
{
    "intent": "intention principale en minuscules",
    "confidence": "high/medium/low",
    "slots": {
        "slot1": "valeur1",
        "slot2": "valeur2",
        ...
    }
}

Exemples de slots possibles (mais non limités) :
- location : lieux, villes, adresses
- date : dates, moments, périodes
- price : prix, budgets, coûts
- type : types de services, catégories
- preferences : préférences, contraintes
- quantity : nombres, quantités
- time : heures, durées
- person : personnes, groupes
- etc.

Important :
- L'intention doit être un mot simple et clair
- Les slots doivent être pertinents au contexte
- Utilisez des valeurs vides ("") pour les slots non détectés
- La confiance doit refléter votre certitude dans l'analyse
"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Message à analyser : {message}"}
        ]

    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Interroge l'API Mistral

        Args:
            prompt (List[Dict[str, str]]): prompt formaté

        Returns:
            str: Réponse du LLM
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
            return json.dumps({
                "intent": "unknown",
                "confidence": "low",
                "slots": {}
            })

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse la réponse du LLM pour extraire l'intention et les slots.

        Args:
            response (str): Réponse du LLM

        Returns:
            Dict[str, Any]: Dictionnaire contenant l'intention et les slots
        """
        try:
            # Essayer de parser la réponse comme du JSON
            result = json.loads(response)
            
            # Vérifier la structure minimale
            if not isinstance(result, dict):
                raise ValueError("La réponse n'est pas un dictionnaire")
                
            # S'assurer que les champs requis sont présents
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", "medium")
            slots = result.get("slots", {})
            
            return {
                "intent": intent.lower(),
                "confidence": confidence,
                "slots": slots
            }
            
        except json.JSONDecodeError:
            # Si le JSON n'est pas valide, essayer d'extraire l'information du texte
            intent_match = re.search(r'intent["\s:]+([^"\n,}]+)', response, re.IGNORECASE)
            intent = intent_match.group(1).strip().lower() if intent_match else "unknown"
            
            # Extraire les slots du texte
            slots = {}
            slot_matches = re.finditer(r'"([^"]+)":\s*"([^"]+)"', response)
            for match in slot_matches:
                key, value = match.groups()
                slots[key] = value
            
            return {
                "intent": intent,
                "confidence": "medium",
                "slots": slots
            }