# intent_detection_agent.py
from Agent import Agent
import ollama
from typing import Dict, Any, List, Optional

class intentDetectionAgent(Agent):
    """
    Agent pour détecter l'intention et extraire les slots remplis ou manquants
    """

    VALID_INTENTS = ["restaurant_search", "activity_search", "hotel_booking"]

    def __init__(self, name: str = "intent_slot_extractor"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = {
            "name": "gemma3",
            "temperature": 0.4,
            "max_tokens": 200
        }

    def run(self, message: str) -> Dict[str, Any]:
        """
        Analyse un message pour détecter l'intention et extraire les slots.

        Args:
            message (str): Le message à analyser

        Returns:
            Dict[str, Any]: Dictionnaire contenant 'intent' et 'slots'
        """
        try:
            prompt = self._build_prompt(message)
            response = self._get_llm_response(prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"Erreur lors de la détection d'intention/slot: {e}")
            return {"intent": None, "slots": {}}

    def _build_prompt(self, message: str) -> List[Dict[str, str]]:
        """
        Construit le prompt d'analyse.

        Args:
            message (str): Le message à analyser

        Returns:
            List[Dict[str, str]]: Prompt formaté
        """
        system_prompt = """
Vous êtes un agent expert en traitement du langage naturel. Votre tâche est d'analyser un message utilisateur et d'en extraire :
1. L'intention principale de l'utilisateur parmi : restaurant_search, activity_search, hotel_booking
2. Les slots (informations extraites) remplis, sous forme de paires clé/valeur.

Répondez dans le format suivant (sans texte en dehors) :
Intent: <intent>
Slots:
- <slot_name>: <slot_value>
- ...
Si aucune intention claire n'est identifiable, utilisez "Intent: unknown" et "Slots: {}"
"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Message à analyser : {message}"}
        ]

    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Interroge le LLM

        Args:
            prompt (List[Dict[str, str]]): prompt formaté

        Returns:
            str: contenu brut de la réponse
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

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse la réponse du LLM pour en extraire l'intention et les slots

        Args:
            response (str): Réponse texte brute

        Returns:
            Dict[str, Any]: Dictionnaire { intent: str, slots: dict }
        """
        lines = response.strip().splitlines()
        intent = None
        slots = {}

        for line in lines:
            if line.lower().startswith("intent:"):
                intent = line.split(":", 1)[1].strip()
            elif line.strip().startswith("-"):
                if ":" in line:
                    key, value = line.strip("- ").split(":", 1)
                    slots[key.strip()] = value.strip()

        if intent not in self.VALID_INTENTS and intent != "unknown":
            intent = "unknown"

        return {"intent": intent, "slots": slots}