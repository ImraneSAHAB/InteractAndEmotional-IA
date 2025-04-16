# intent_detection_agent.py
from Agent import Agent
import ollama
from typing import Dict, Any, List, Optional
import re

class intentDetectionAgent(Agent):
    """
    Agent pour détecter l'intention et extraire les slots remplis ou manquants
    """

    VALID_INTENTS = ["restaurant_search", "activity_search", "hotel_booking", "salutation", "presentation", "remerciement", "confirmation", "negation", "information_generale"]

    # Liste des villes communes pour améliorer la détection
    COMMON_CITIES = [
        "dijon", "paris", "lyon", "marseille", "bordeaux", "toulouse", "lille", 
        "nantes", "strasbourg", "rennes", "nice", "toulon", "grenoble", "montpellier",
        "rouen", "nancy", "orleans", "tours", "amiens", "caen", "reims", "le havre",
        "saint-etienne", "brest", "le mans", "dijon", "clermont-ferrand", "toulon",
        "limoges", "villeurbanne", "nimes", "tours", "pau", "poitiers", "perpignan",
        "metz", "lens", "argenteuil", "orleans", "roubaix", "montreuil", "mulhouse",
        "saint-denis", "nancy", "rouen", "argenteuil", "toulon", "fort-de-france"
    ]

    # Mots-clés pour la détection d'intention
    INTENT_KEYWORDS = {
        "restaurant_search": [
            "restaurant", "manger", "dîner", "diner", "déjeuner", "repas", "table", 
            "gastronomie", "cuisine", "bistrot", "brasserie", "café", "cafe", "bar",
            "food", "dinner", "lunch", "breakfast", "petit-déjeuner", "petit dejeuner"
        ],
        "activity_search": [
            "activité", "activite", "visiter", "voir", "faire", "découvrir", "découverte",
            "excursion", "tour", "visite", "monument", "musée", "musee", "parc", "jardin",
            "activity", "visit", "see", "do", "discover", "discovery", "tour", "monument",
            "museum", "park", "garden"
        ],
        "hotel_booking": [
            "hôtel", "hotel", "logement", "hébergement", "hebergement", "chambre", "lit",
            "nuit", "séjour", "sejour", "réservation", "reservation", "booking", "book",
            "accommodation", "room", "bed", "night", "stay", "reserve"
        ],
        "salutation": [
            "bonjour", "salut", "hey", "coucou", "bonsoir", "hello", "hi", "hey", "bonsoir",
            "bonne journée", "bonne soirée", "à bientôt", "au revoir", "bye", "goodbye"
        ],
        "presentation": [
            "je m'appelle", "mon nom est", "je suis", "présentation", "presentation",
            "qui es-tu", "qui êtes-vous", "comment vous appelez-vous", "comment t'appelles-tu"
        ],
        "remerciement": [
            "merci", "thanks", "thank you", "merci beaucoup", "je vous remercie",
            "je te remercie", "c'est gentil", "c'est sympa", "c'est cool"
        ],
        "confirmation": [
            "oui", "d'accord", "ok", "parfait", "super", "génial", "excellent",
            "je confirme", "c'est noté", "je valide", "je suis d'accord"
        ],
        "negation": [
            "non", "pas du tout", "absolument pas", "jamais", "ne pas", "ne plus",
            "refuser", "refuse", "refusé", "refusee"
        ],
        "information_generale": [
            "quoi", "comment", "pourquoi", "quand", "où", "qui", "quel", "quelle",
            "quels", "quelles", "explique", "expliquer", "détails", "details",
            "plus d'information", "plus d'infos", "en savoir plus"
        ]
    }

    def __init__(self, name: str = "intent_slot_extractor"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = self._config["model"]
        # Initialisation des slots par défaut
        self._current_slots = {
            "location": None,
            "food_type": None,
            "budget": None,
            "time": None
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
            # Extraire les slots avec les patterns (rapide)
            extracted_slots = self._extract_slots(message)
            
            # Détecter l'intention avec les mots-clés (rapide)
            intent = self._detect_intent_with_keywords(message)
            
            # Si l'intention n'est pas claire, utiliser le LLM (lent)
            if intent == "unknown":
                prompt = self._build_prompt(message)
                response = self._get_llm_response(prompt)
                result = self._parse_response(response)
                intent = result["intent"]
                
                # Fusionner les slots extraits par le LLM avec ceux extraits par les patterns
                llm_slots = result["slots"]
                for key, value in llm_slots.items():
                    if value is not None and (key not in extracted_slots or extracted_slots[key] is None):
                        extracted_slots[key] = value
            
            # Mettre à jour les slots actuels
            for key, value in extracted_slots.items():
                if value is not None:
                    self._current_slots[key] = value
            
            return {
                "intent": intent,
                "slots": self._current_slots
            }
            
        except Exception as e:
            return {
                "intent": "unknown",
                "slots": self._current_slots
            }

    def _detect_intent_with_keywords(self, message: str) -> str:
        """Détecte l'intention en utilisant des mots-clés"""
        message = message.lower()
        intent_counts = {intent: 0 for intent in self.VALID_INTENTS}
        
        # Compter les occurrences des mots-clés pour chaque intention
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message:
                    intent_counts[intent] += 1
        
        # Trouver l'intention avec le plus de mots-clés
        max_count = max(intent_counts.values())
        if max_count > 0:
            return max(intent_counts.items(), key=lambda x: x[1])[0]
        
        return "unknown"

    def _extract_slots(self, message: str) -> Dict[str, Any]:
        """
        Extrait les slots du message en utilisant des patterns et des mots-clés.

        Args:
            message (str): Le message à analyser

        Returns:
            Dict[str, Any]: Les slots extraits
        """
        message_lower = message.lower()
        slots = {
            "location": None,
            "food_type": None,
            "budget": None,
            "time": None
        }

        # Détection de la localisation
        # Patterns pour détecter les villes
        location_patterns = [
            r'à\s+([a-zéèêëàâäôöûüçîï]+(?:\s+[a-zéèêëàâäôöûüçîï-]+)*)',
            r'dans\s+([a-zéèêëàâäôöûüçîï]+(?:\s+[a-zéèêëàâäôöûüçîï-]+)*)',
            r'en\s+([a-zéèêëàâäôöûüçîï]+(?:\s+[a-zéèêëàâäôöûüçîï-]+)*)',
            r'à\s+([a-zéèêëàâäôöûüçîï]+(?:\s+[a-zéèêëàâäôöûüçîï-]+)*)',
            r'dans\s+([a-zéèêëàâäôöûüçîï]+(?:\s+[a-zéèêëàâäôöûüçîï-]+)*)',
            r'en\s+([a-zéèêëàâäôöûüçîï]+(?:\s+[a-zéèêëàâäôöûüçîï-]+)*)'
        ]

        # Vérifier d'abord dans la liste des villes communes
        for city in self.COMMON_CITIES:
            if city in message_lower:
                slots["location"] = city.title()
                break

        # Si aucune ville commune n'est trouvée, utiliser les patterns
        if slots["location"] is None:
            for pattern in location_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    potential_city = match.group(1).strip()
                    # Vérifier si la ville extraite est dans la liste des villes communes
                    if potential_city.lower() in self.COMMON_CITIES:
                        slots["location"] = potential_city.title()
                        break

        # Type de nourriture
        food_patterns = {
            "traditional": ["tradition", "traditionnel", "classique", "classic"],
            "modern": ["moderne", "contemporain", "fusion", "innovant"],
            "vegetarian": ["végétarien", "végé", "vegan", "végétalien"],
            "asian": ["asiatique", "chinois", "japonais", "thaï", "vietnamien"],
            "italian": ["italien", "pizza", "pasta", "risotto"],
            "french": ["français", "bistrot", "brasserie", "gastronomique"]
        }

        for food_type, keywords in food_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                slots["food_type"] = food_type
                break

        # Budget
        budget_patterns = {
            "budget": ["pas cher", "économique", "budget", "petit prix", "bon marché"],
            "mid-range": ["moyen", "modéré", "standard", "normal"],
            "luxury": ["luxe", "haut de gamme", "gastronomique", "premium", "exclusif"]
        }

        for budget_type, keywords in budget_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                slots["budget"] = budget_type
                break

        # Temps
        time_patterns = {
            "tonight": ["ce soir", "tonight", "dîner", "diner"],
            "tomorrow": ["demain", "tomorrow"],
            "this_weekend": ["weekend", "semaine", "week-end", "samedi", "dimanche"],
            "lunch": ["déjeuner", "midi", "lunch"],
            "dinner": ["dîner", "soir", "dinner"]
        }

        for time_type, keywords in time_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                slots["time"] = time_type
                break

        return slots

    def _merge_slots(self, llm_slots: Dict[str, Any], extracted_slots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusionne les slots extraits par le LLM et par les patterns.

        Args:
            llm_slots (Dict[str, Any]): Slots extraits par le LLM
            extracted_slots (Dict[str, Any]): Slots extraits par les patterns

        Returns:
            Dict[str, Any]: Slots fusionnés
        """
        merged = self._current_slots.copy()
        
        # Mettre à jour avec les slots du LLM
        for key, value in llm_slots.items():
            if value is not None:
                merged[key] = value
                
        # Mettre à jour avec les slots extraits par les patterns
        for key, value in extracted_slots.items():
            if value is not None:
                merged[key] = value
                
        # Mettre à jour les slots courants
        self._current_slots = merged
        
        return merged

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
1. L'intention principale de l'utilisateur parmi : restaurant_search, activity_search, hotel_booking, salutation, presentation, remerciement, confirmation, negation, information_generale
2. Les slots (informations extraites) remplis, sous forme de paires clé/valeur.

Les slots possibles sont :
- location : la ville ou le lieu mentionné
- food_type : le type de cuisine (traditional, modern, vegetarian, asian, italian, french)
- budget : le niveau de prix (budget, mid-range, luxury)
- time : le moment (tonight, tomorrow, this_weekend, lunch, dinner)

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
        try:
            response = self._llm.chat(
                model=self._model_config["name"],
                messages=prompt,
                options={
                    "temperature": self._model_config["temperature"],
                    "max_tokens": self._model_config["max_tokens"]
                }
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Erreur lors de l'appel au LLM: {e}")
            return "Intent: unknown\nSlots: {}"

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