# dialogue_planner_agent.py

from Agent import Agent
from typing import Dict, Any, List

class DialoguePlannerAgent(Agent):
    """
    Agent responsable de déterminer la prochaine question à poser à l'utilisateur
    en fonction des slots manquants pour une intention donnée.

    Hérite de la classe de base Agent pour charger sa configuration et ses métadonnées.
    """

    def __init__(self, name: str = "dialogue_planner"):
        super().__init__(name)
        # Dictionnaire de templates pour les questions, basé sur les slots
        self._slot_questions: Dict[str, str] = {
            "location": "Dans quelle ville ou région recherchez-vous ?",
            "food_type": "Quel type de cuisine vous intéresse ?",
            "budget": "Quel budget prévoyez-vous pour le repas ?",
            "time": "À quelle heure prévoyez-vous d'y aller ?",
            "activity_type": "Quel type d'activité aimeriez-vous faire ?",
            "date": "Préférez-vous une date spécifique pour cela ?",
            "price_range": "Quelle gamme de prix recherchez-vous ?"
        }

    def run(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        required_slots: Dict[str, List[str]]
    ) -> str:
        """
        Détermine la prochaine question pertinente à poser en fonction des slots manquants.

        Args:
            intent (str): L'intention détectée (ex: restaurant_search).
            filled_slots (Dict[str, Any]): Dictionnaire des slots déjà remplis.
            required_slots (Dict[str, List[str]]): Dictionnaire des intentions vers leurs slots requis.

        Returns:
            str: La question naturelle la plus pertinente à poser pour combler les informations manquantes.
        """
        # Récupérer la liste des slots requis pour cette intention
        slots_list = required_slots.get(intent, [])

        # Déterminer les slots manquants
        missing_slots = [
            slot for slot in slots_list
            if slot not in filled_slots or not filled_slots.get(slot)
        ]

        # Pour chaque slot manquant, rechercher une question prédéfinie
        for slot in missing_slots:
            if slot in self._slot_questions:
                return self._slot_questions[slot]

        # Si aucun slot manquant n'est couvert, retourner une question générique
        return "Pouvez-vous fournir plus de détails sur votre demande ?"

    def get_next_question(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        required_slots: Any
    ) -> str:
        """
        Alias pour run(), accepte aussi une liste de slots requis.

        Args:
            intent (str): L'intention détectée.
            filled_slots (Dict[str, Any]): Slots déjà remplis.
            required_slots (Union[Dict[str, List[str]], List[str]]): Soit dict intent->slots, soit liste de slots pour l'intent.

        Returns:
            str: Prochaine question.
        """
        # Si required_slots est une liste, l'envelopper dans un dict
        if isinstance(required_slots, list):
            required_slots = {intent: required_slots}
        return self.run(intent, filled_slots, required_slots)
