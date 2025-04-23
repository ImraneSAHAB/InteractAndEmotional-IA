# dialogue_planner_agent.py

from Agent import Agent
from typing import Dict, Any, List
import ollama
import json

class DialoguePlannerAgent(Agent):
    """
    Agent responsable de déterminer la prochaine question à poser à l'utilisateur
    en fonction des slots manquants pour une intention donnée.

    Hérite de la classe de base Agent pour charger sa configuration et ses métadonnées.
    """

    def __init__(self, name: str = "dialogue_planner"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = self._config["model"]
        
        # Dictionnaire des descriptions des slots
        self._slot_descriptions = {
            "location": "la ville ou le lieu",
            "food_type": "le type de cuisine (traditionnelle, moderne, végétarienne, asiatique, italienne, française)",
            "budget": "le niveau de prix (budget, moyen, luxe)",
            "time": "le moment (ce soir, demain, ce week-end, déjeuner, dîner)",
            "activity_type": "le type d'activité (culturelle, sportive, gastronomique, etc.)",
            "date": "la date ou la période"
        }

    def get_next_question(
        self,
        intent: str,
        filled_slots: Dict[str, Any],
        required_slots: List[str]
    ) -> str:
        """
        Génère la prochaine question à poser en fonction des slots manquants.

        Args:
            intent (str): L'intention détectée (ex: restaurant_search).
            filled_slots (Dict[str, Any]): Dictionnaire des slots déjà remplis.
            required_slots (List[str]): Liste des slots requis pour cette intention.

        Returns:
            str: La question naturelle à poser pour obtenir les informations manquantes.
        """
        # Déterminer les slots manquants
        missing_slots = [
            slot for slot in required_slots
            if slot not in filled_slots or not filled_slots.get(slot)
        ]

        if not missing_slots:
            return "Pouvez-vous me donner plus de détails sur votre demande ?"

        # Construire le prompt pour générer la question
        prompt = self._build_question_prompt(intent, missing_slots, filled_slots)
        
        try:
            # Obtenir la réponse du LLM
            response = self._llm.chat(
                model=self._model_config["name"],
                messages=prompt,
                options={
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            )
            
            # Nettoyer et retourner la réponse
            question = response["message"]["content"].strip()
            if question.startswith('"') and question.endswith('"'):
                question = question[1:-1]
            return question
            
        except Exception as e:
            print(f"Erreur lors de la génération de la question: {e}")
            # Retourner une question par défaut
            return f"Pouvez-vous me préciser {self._slot_descriptions.get(missing_slots[0], 'cette information')} ?"

    def _build_question_prompt(
        self,
        intent: str,
        missing_slots: List[str],
        filled_slots: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Construit le prompt pour générer une question naturelle.

        Args:
            intent (str): L'intention détectée.
            missing_slots (List[str]): Liste des slots manquants.
            filled_slots (Dict[str, Any]): Slots déjà remplis.

        Returns:
            List[Dict[str, str]]: Le prompt formaté.
        """
        # Construire la description du contexte
        context = []
        if filled_slots:
            context.append("Informations déjà connues:")
            for slot, value in filled_slots.items():
                if value:
                    context.append(f"- {self._slot_descriptions.get(slot, slot)}: {value}")
        
        # Construire la liste des informations manquantes
        missing_info = [
            f"- {self._slot_descriptions.get(slot, slot)}"
            for slot in missing_slots
        ]

        system_prompt = """Vous êtes un assistant touristique expert en communication naturelle. 
Votre tâche est de formuler une question unique et naturelle pour obtenir les informations manquantes.

Instructions:
1. Formulez une seule question claire et naturelle
2. Utilisez un ton amical et professionnel
3. Ne mentionnez pas que vous êtes un assistant
4. Ne faites pas référence aux "slots" ou "informations"
5. Adaptez la question au contexte et aux informations déjà connues
6. Utilisez un langage conversationnel et naturel
7. Si plusieurs informations sont manquantes, choisissez la plus pertinente à demander en premier

Exemples de bonnes questions:
- "Dans quelle ville souhaitez-vous dîner ?"
- "Quel type de cuisine préférez-vous ?"
- "Quel est votre budget pour cette activité ?"
- "Quand souhaitez-vous réserver ?"

Répondez uniquement avec la question, sans explications supplémentaires."""

        user_prompt = f"""Contexte de la conversation:
Intention: {intent}

{chr(10).join(context)}

Informations manquantes:
{chr(10).join(missing_info)}

Générez une question naturelle pour obtenir ces informations manquantes."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
