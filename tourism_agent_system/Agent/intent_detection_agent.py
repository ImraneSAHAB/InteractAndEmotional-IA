# intent_detection_agent.py
from .Agent import Agent
import requests
from typing import Dict, Any, List, Optional
import re
import json

class IntentDetectionAgent(Agent):
    """
    Agent pour détecter l'intention et extraire les slots remplis ou manquants
    """

    VALID_INTENTS = ["restaurant_search", "activity_search", "hotel_booking", "salutation", "presentation", "remerciement", "confirmation", "negation", "information_generale", "demande_information"]

    def __init__(self, name: str = "intent_slot_extractor"):
        super().__init__(name)
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        # Initialisation des slots par défaut
        self._current_slots = {
            "location": "",
            "food_type": "",
            "budget": "",
            "time": "",
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
            # Initialiser les slots avec des valeurs vides
            self._current_slots = {
                "location": "",
                "food_type": "",
                "budget": "",
                "time": "",
                "activity_type": "",
                "date": ""
            }
            
            # Utiliser le LLM pour l'analyse complète
            prompt = self._build_prompt(message)
            response = self._get_llm_response(prompt)
            result = self._parse_response(response)
            
            # Mettre à jour les slots et l'intention
            intent = result["intent"]
            self._current_slots = result["slots"]
            
            # Vérifier si c'est une demande d'information
            if intent == "demande_information":
                search_query = self._extract_search_query(message)
                return {
                    "intent": intent,
                    "slots": self._current_slots.copy(),
                    "search_query": search_query
                }
            
            return {
                "intent": intent,
                "slots": self._current_slots.copy(),
                "search_query": None
            }
            
        except Exception as e:
            return {
                "intent": "unknown",
                "slots": {
                    "location": "",
                    "food_type": "",
                    "budget": "",
                    "time": "",
                    "activity_type": "",
                    "date": ""
                }
            }

    def _extract_slots(self, message: str, intent: str) -> Dict[str, str]:
        """
        Extrait les slots pertinents du message en fonction de l'intention détectée.
        Utilise le LLM pour extraire les informations de manière plus flexible.

        Args:
            message (str): Le message de l'utilisateur
            intent (str): L'intention détectée

        Returns:
            Dict[str, str]: Dictionnaire des slots extraits
        """
        # Définir les slots requis pour chaque intention
        required_slots = {
            "restaurant_search": ["location", "food_type", "budget", "time"],
            "activity_search": ["location", "activity_type", "date", "price_range"]
        }

        # Si l'intention n'a pas de slots requis, retourner un dictionnaire vide
        if intent not in required_slots:
            return {}

        # Construire le prompt pour l'extraction des slots
        slots_prompt = f"""
        En tant qu'agent de détection d'intention, extrayez les informations pertinentes du message suivant.
        Intention détectée : {intent}
        Slots à extraire : {', '.join(required_slots[intent])}
        
        Message : {message}
        
        Pour chaque slot, extrayez la valeur si elle est présente dans le message.
        Si une information n'est pas présente, laissez le champ vide.
        
        Format de réponse attendu (JSON) :
        {{
            "slot1": "valeur1",
            "slot2": "valeur2",
            ...
        }}
        """

        try:
            # Obtenir la réponse du LLM
            response = self._get_llm_response([{"role": "user", "content": slots_prompt}])
            
            # Parser la réponse JSON
            slots = json.loads(response)
            
            # S'assurer que tous les slots requis sont présents
            for slot in required_slots[intent]:
                if slot not in slots:
                    slots[slot] = ""
            
            return slots
            
        except Exception as e:
            print(f"Erreur lors de l'extraction des slots : {str(e)}")
            # En cas d'erreur, retourner un dictionnaire avec des slots vides
            return {slot: "" for slot in required_slots[intent]}

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
1. L'intention principale de l'utilisateur parmi : restaurant_search, activity_search, hotel_booking, salutation, presentation, remerciement, confirmation, negation, information_generale, demande_information
2. Les slots (informations extraites) remplis, sous forme de paires clé/valeur.

Les slots possibles sont :
- location : la ville ou le lieu mentionné (ex: "Paris", "Dijon", etc.)
- food_type : le type de cuisine (chinois, italien, français, japonais, indien, etc.)
- budget : le niveau de prix (20€, 30€, pas cher, moyen, luxe, etc.)
- time : le moment (ce soir, demain, ce weekend, midi, soir, etc.)

Instructions spécifiques :
1. Si le message contient une demande de recherche de restaurant (ex: "je veux trouver un restaurant", "je cherche un restaurant", "je souhaite trouver un restaurant"), détectez l'intention comme "restaurant_search"
2. Si le message contient une demande de recherche d'activité (ex: "je veux faire une activité", "je cherche des activités", "je souhaite trouver des choses à faire"), détectez l'intention comme "activity_search"
3. Si le message contient une demande d'activité, détectez l'intention comme "activity_search"
4. Si le message contient une demande de réservation d'hôtel (ex: "je veux réserver un hôtel", "je cherche un hôtel", "je souhaite trouver un logement"), détectez l'intention comme "hotel_booking"
5. Si le message contient une demande d'hôtel, détectez l'intention comme "hotel_booking"
6. Pour les questions comme "Où habites-tu ?", détectez l'intention comme "demande_information"
7. Ne laissez jamais un slot vide, utilisez une chaîne vide ("") si aucune information n'est trouvée
8. Soyez attentif aux variations de formulation (ex: "je suis à", "je vis à", "j'habite à", etc.)

Instructions pour l'extraction des slots :
1. Pour le type de cuisine (food_type), détectez tous les types de cuisine mentionnés (chinois, italien, français, etc.)
2. Pour le budget, détectez les montants exacts (20€, 30€) et les niveaux de prix (pas cher, moyen, luxe)
3. Pour l'heure (time), détectez les moments précis (ce soir, demain) et les périodes (midi, soir)
4. Pour la localisation (location), détectez les villes et les quartiers mentionnés

Répondez dans le format suivant (sans texte en dehors) :
Intent: <intent>
Slots:
- location: <ville ou lieu>
- food_type: <type de cuisine>
- budget: <niveau de prix>
- time: <moment>
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
            raise

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse la réponse du LLM pour en extraire l'intention et les slots.

        Args:
            response (str): Réponse texte brute

        Returns:
            Dict[str, Any]: Dictionnaire { intent: str, slots: dict }
        """
        lines = response.strip().splitlines()
        intent = None
        slots = {
            "location": None,
            "food_type": None,
            "budget": None,
            "time": None,
        }

        for line in lines:
            if line.lower().startswith("intent:"):
                intent = line.split(":", 1)[1].strip()
            elif line.strip().startswith("-"):
                if ":" in line:
                    key, value = line.strip("- ").split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Supprimer les guillemets si présents
                    value = value.strip('"')
                    if key in slots:
                        # Convertir les valeurs vides en None
                        slots[key] = value if value else None

        if intent not in self.VALID_INTENTS and intent != "unknown":
            intent = "unknown"

        return {"intent": intent, "slots": slots}

    def check_missing_info(self, slots: Dict[str, Any]) -> List[str]:
        """
        Vérifie quelles informations sont manquantes dans les slots.
        
        Args:
            slots (Dict[str, Any]): Les slots actuels
            
        Returns:
            List[str]: Liste des informations manquantes
        """
        missing_info = []
        
        # Vérifier si la localisation est manquante
        if not slots.get("location"):
            missing_info.append("location")
            
        # Vérifier si le type de nourriture est manquant (pour les recherches de restaurants)
        if slots.get("intent") == "restaurant_search" and not slots.get("food_type"):
            missing_info.append("food_type")
            
        # Vérifier si le budget est manquant (pour les recherches de restaurants et d'hôtels)
        if slots.get("intent") in ["restaurant_search", "hotel_booking"] and not slots.get("budget"):
            missing_info.append("budget")
            
        # Vérifier si l'heure est manquante (pour les recherches de restaurants)
        if slots.get("intent") == "restaurant_search" and not slots.get("time"):
            missing_info.append("time")
            
        return missing_info

    def _extract_search_query(self, message: str) -> str:
        """
        Extrait la requête de recherche d'un message en utilisant le LLM.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            str: La requête de recherche
        """
        prompt = [
            {"role": "system", "content": """
Vous êtes un expert en extraction d'informations. Votre tâche est d'extraire la requête de recherche d'un message.
Pour les questions comme "Où habites-tu ?", "Quel est ton budget ?", etc., extrayez la question complète.
Pour les autres types de messages, retournez le message tel quel.

Répondez uniquement avec la requête extraite, sans explications ni texte supplémentaire.
"""},
            {"role": "user", "content": f"Message à analyser : {message}"}
        ]

        try:
            response = self._get_llm_response(prompt)
            return response.strip()
        except Exception as e:
            return message