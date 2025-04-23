# threshold_agent.py

from Agent import Agent
from typing import Dict, Any, List, Optional

class ThresholdAgent(Agent):
    """
    Agent responsable de vérifier si tous les slots requis sont remplis
    et de signaler à l'orchestrator si une réponse finale peut être générée.
    """

    def __init__(self, name: str = "threshold"):
        super().__init__(name)

    def check_slots(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        required_slots: List[str]
    ) -> Dict[str, Any]:
        """
        Vérifie si tous les slots requis sont remplis.

        Args:
            intent (str): L'intention détectée.
            filled_slots (Dict[str, Any]): Dictionnaire des slots déjà remplis.
            required_slots (List[str]): Liste des slots requis pour cette intention.

        Returns:
            Dict[str, Any]: Un dictionnaire contenant:
                - 'is_complete': bool indiquant si tous les slots sont remplis
                - 'missing_slots': liste des slots manquants
                - 'intent': l'intention détectée
                - 'filled_slots': les slots remplis
        """
        # Vérifier les slots manquants
        missing_slots = [
            slot for slot in required_slots
            if slot not in filled_slots or not filled_slots.get(slot)
        ]

        return {
            "is_complete": len(missing_slots) == 0,
            "missing_slots": missing_slots,
            "intent": intent,
            "filled_slots": filled_slots
        }

    def get_missing_slots_message(self, missing_slots: List[str]) -> str:
        """
        Génère un message informatif sur les slots manquants.

        Args:
            missing_slots (List[str]): Liste des slots manquants.

        Returns:
            str: Message informatif sur les informations manquantes.
        """
        if not missing_slots:
            return "Toutes les informations nécessaires sont disponibles."

        slot_descriptions = {
            "location": "la ville ou le lieu",
            "food_type": "le type de cuisine",
            "budget": "le niveau de prix",
            "time": "le moment",
            "activity_type": "le type d'activité",
            "date": "la date ou la période"
        }

        descriptions = [
            slot_descriptions.get(slot, slot)
            for slot in missing_slots
        ]

        if len(descriptions) == 1:
            return f"J'ai besoin de connaître {descriptions[0]}."
        else:
            return f"J'ai besoin de connaître {', '.join(descriptions[:-1])} et {descriptions[-1]}." 