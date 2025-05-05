# dialogue_planner_agent.py

from .Agent import Agent
from typing import Dict, Any, List
import ollama
import json

class DialoguePlannerAgent(Agent):
    """
    Agent qui planifie les questions à poser pour obtenir les informations manquantes.
    """
    
    def __init__(self, name: str = "dialogue_planner"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = self._config["model"]

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
        prompt = [
            {"role": "system", "content": """Vous êtes un assistant touristique qui aide les utilisateurs à trouver des restaurants.
            Votre tâche est de poser une question naturelle pour obtenir les informations manquantes.
            
            Instructions importantes:
            1. Posez UNE SEULE question claire et naturelle
            2. Ne donnez PAS d'exemples de réponses possibles
            3. Ne mentionnez pas que vous êtes un assistant
            4. Ne faites pas référence aux "slots" ou "informations manquantes"
            5. Utilisez un langage conversationnel
            6. Adaptez la question au contexte de la conversation
            7. NE LISTEZ PAS les options possibles
            8. Posez une question OUVERTE
            9. NE DEMANDEZ PAS le nom du restaurant (vous devez le trouver)
            10. Concentrez-vous sur les informations nécessaires pour faire une recommandation
            
            Répondez uniquement avec la question, sans explications supplémentaires."""},
            {"role": "user", "content": f"""
            Intention: {intent}
            
            Ce que nous savons déjà:
            {self._format_known_slots(filled_slots)}
            
            Information(s) à obtenir: {', '.join(self._get_slot_descriptions(missing_slots))}
            
            Générez une question naturelle pour obtenir ces informations."""}
        ]
        
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
            return question
            
        except Exception as e:
            print(f"Erreur lors de la génération de la question: {e}")
            return "Pouvez-vous me donner plus d'informations ?"
            
    def _format_known_slots(self, slots: Dict[str, Any]) -> str:
        """
        Formate les slots connus pour l'affichage.
        
        Args:
            slots (Dict[str, Any]): Les slots à formater
            
        Returns:
            str: Les slots formatés
        """
        formatted = []
        for key, value in slots.items():
            if value:
                description = self._get_slot_description(key)
                formatted.append(f"- {description}: {value}")
        return "\n".join(formatted) if formatted else "Aucune information connue"
        
    def _get_slot_description(self, slot: str) -> str:
        """
        Retourne la description d'un slot.
        
        Args:
            slot (str): Le nom du slot
            
        Returns:
            str: La description du slot
        """
        slot_descriptions = {
            "location": "la ville ou le lieu",
            "food_type": "le type de cuisine",
            "budget": "le budget",
            "time": "l'horaire"
        }
        return slot_descriptions.get(slot, slot)
        
    def _get_slot_descriptions(self, slots: List[str]) -> List[str]:
        """
        Retourne les descriptions des slots.
        
        Args:
            slots (List[str]): Liste des slots
            
        Returns:
            List[str]: Liste des descriptions
        """
        return [self._get_slot_description(slot) for slot in slots]
