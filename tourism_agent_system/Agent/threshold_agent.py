# threshold_agent.py

from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional

class ThresholdAgent(BaseAgent):
    """
    Agent responsable de la gestion des seuils de confiance pour les différentes actions.
    """

    def __init__(self, name: str = "threshold"):
        super().__init__(name)
        self._threshold_config = self._config.get("threshold", {})
        self._default_threshold = 0.7

    def check_threshold(self, value: float, action_type: str) -> bool:
        """
        Vérifie si une valeur dépasse le seuil pour un type d'action donné.
        
        Args:
            value (float): La valeur à vérifier
            action_type (str): Le type d'action (ex: "search", "booking", etc.)
            
        Returns:
            bool: True si la valeur dépasse le seuil, False sinon
        """
        threshold = self._threshold_config.get(action_type, self._default_threshold)
        return value >= threshold

    def get_threshold(self, action_type: str) -> float:
        """
        Récupère le seuil pour un type d'action donné.
        
        Args:
            action_type (str): Le type d'action
            
        Returns:
            float: Le seuil pour ce type d'action
        """
        return self._threshold_config.get(action_type, self._default_threshold)

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Méthode principale pour exécuter l'agent.
        
        Args:
            prompt (str): Le message contenant la valeur et le type d'action
            
        Returns:
            Dict[str, Any]: Résultat de la vérification du seuil
        """
        try:
            # Analyse du prompt pour extraire la valeur et le type d'action
            parts = prompt.split("|")
            if len(parts) != 2:
                raise ValueError("Format invalide. Attendu: 'valeur|action_type'")
                
            value = float(parts[0].strip())
            action_type = parts[1].strip()
            
            # Vérification du seuil
            threshold = self.get_threshold(action_type)
            is_above = self.check_threshold(value, action_type)
            
            return {
                "success": True,
                "value": value,
                "action_type": action_type,
                "threshold": threshold,
                "is_above_threshold": is_above
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "value": 0.0,
                "action_type": "unknown",
                "threshold": self._default_threshold,
                "is_above_threshold": False
            }

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
        missing_slots = []
        for slot in required_slots:
            # Vérifier si le slot existe et a une valeur non-nulle
            if slot not in filled_slots or filled_slots.get(slot) is None or filled_slots.get(slot) == "":
                missing_slots.append(slot)

        # Vérifier si tous les slots requis sont remplis
        is_complete = len(missing_slots) == 0

        return {
            "is_complete": is_complete,
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

    def check_thresholds(
        self,
        intent: Dict[str, Any],
        emotion: Dict[str, Any],
        search_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Vérifie les seuils pour l'intent, l'émotion et les résultats de recherche.
        
        Args:
            intent (Dict[str, Any]): Résultat de la détection d'intention
            emotion (Dict[str, Any]): Résultat de la détection d'émotion
            search_results (Dict[str, Any]): Résultats de la recherche
            
        Returns:
            Dict[str, Any]: Résultat de la vérification des seuils
        """
        try:
            # Vérifier la confiance de l'intent
            intent_confidence = intent.get("confidence", 0.0)
            intent_above = self.check_threshold(intent_confidence, "intent")
            
            # Vérifier la confiance de l'émotion
            emotion_confidence = emotion.get("confidence", 0.0)
            emotion_above = self.check_threshold(emotion_confidence, "emotion")
            
            # Vérifier la qualité des résultats de recherche
            search_quality = len(search_results.get("results", [])) / 3.0  # Normaliser sur 3 résultats
            search_above = self.check_threshold(search_quality, "search")
            
            return {
                "status": "success",
                "thresholds": {
                    "intent": {
                        "value": intent_confidence,
                        "threshold": self.get_threshold("intent"),
                        "is_above": intent_above
                    },
                    "emotion": {
                        "value": emotion_confidence,
                        "threshold": self.get_threshold("emotion"),
                        "is_above": emotion_above
                    },
                    "search": {
                        "value": search_quality,
                        "threshold": self.get_threshold("search"),
                        "is_above": search_above
                    }
                },
                "all_above_threshold": intent_above and emotion_above and search_above
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "thresholds": {
                    "intent": {"value": 0.0, "threshold": self._default_threshold, "is_above": False},
                    "emotion": {"value": 0.0, "threshold": self._default_threshold, "is_above": False},
                    "search": {"value": 0.0, "threshold": self._default_threshold, "is_above": False}
                },
                "all_above_threshold": False
            } 