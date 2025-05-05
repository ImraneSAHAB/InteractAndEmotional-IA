from .Agent import Agent
from .memory_agent import MemoryAgent
from .emotion_detection_agent import EmotionDetectionAgent
from .intent_detection_agent import IntentDetectionAgent
from .dialogue_planner_agent import DialoguePlannerAgent
from .threshold_agent import ThresholdAgent
from .search_agent import SearchAgent
from .response_generator_agent import ResponseGeneratorAgent

import ollama
import json
from typing import Dict, Any, List, Optional
class AgentOrchestrator(Agent):
    """
    Orchestrateur qui gère les interactions entre les différents agents.
    Hérite de la classe de base Agent pour charger nom, rôle, goal et backstory.
    """
    
    def __init__(self, name: str = "coordinator"):
        super().__init__(name)  # Initialise configuration et métadonnées via Agent
        # Initialiser le client LLM
        self._llm = ollama.Client()
        # Charger la configuration du modèle (nom, temperature, max_tokens)
        self._model_config = self._config["model"]
        
        # Instancier les agents auxiliaires
        self._memory_agent = MemoryAgent()                # A4: gère la mémoire
        self._emotion_agent = EmotionDetectionAgent()      # A3: détecte l'émotion
        self._intent_agent = IntentDetectionAgent()        # A5: extrait intent et slots
        self._response_generator = ResponseGeneratorAgent()# A7: génère les réponses
        self._dialogue_planner = DialoguePlannerAgent()    # A6: planifie la question suivante
        self._threshold_agent = ThresholdAgent()           # A8: vérifie les slots
        self._search_agent = SearchAgent()                 # A9: effectue les recherches web
        
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Traite un message utilisateur et retourne un dictionnaire complet.
        Étapes :
          1. Détection émotionnelle
          2. Chargement de l'historique
          3. Extraction d'intention et slots
          4. Recherche d'informations en mémoire si besoin
          5. Vérification des slots avec le ThresholdAgent
          6. Décision de poser une question ou générer réponse finale
          7. Enregistrement de l'interaction dans la mémoire
          8. Retour du résultat

        Args:
            message (str): Le message de l'utilisateur.

        Returns:
            Dict[str, Any]: Résultat incluant success, response, context, emotions, slots, intent, found_information.
        """
        try:
            # 1. Détecter les émotions dans le message
            emotions = self._emotion_agent.run(message)
            current_emotion = emotions[0] if emotions else "neutre"

            # 2. Récupérer l'historique des conversations
            conversation_history = self._memory_agent.get_messages()

            # 3. Détecter l'intention et extraire les slots
            intent_result = self._intent_agent.run(message)
            new_intent = intent_result["intent"]
            new_slots = intent_result["slots"]
            current_slots = self._memory_agent._current_slots.copy()
            
            # Définir les slots requis par intention
            required_slots = {
                "restaurant_search": ["location", "food_type", "budget", "time"],
                "activity_search": ["location", "activity_type", "date"],
                "hotel_search": ["location", "date", "budget"],
                "information_generale": ["location"]
            }

            # Vérifier si nous avons une intention en cours
            current_intent = next((intent for intent, slots in required_slots.items() 
                                 if any(current_slots.get(slot) for slot in slots)), None)

            # Si nous avons une intention en cours et que tous les slots ne sont pas remplis
            if current_intent and not all(current_slots.get(slot) for slot in required_slots[current_intent]):
                # Conserver l'intention en cours
                intent = current_intent
                # Mettre à jour uniquement les slots manquants
                current_slots.update({k: v for k, v in new_slots.items() 
                                   if v and v.strip() and not current_slots.get(k)})
            else:
                # Si aucune intention en cours ou tous les slots sont remplis, accepter la nouvelle intention
                intent = new_intent
                # Mettre à jour les slots en fonction de l'intention
                if intent == "information_generale":
                    # Pour les demandes d'information, extraire les informations du message
                    current_slots = self._extract_info(message)
                else:
                    # Pour les autres intentions, mettre à jour tous les slots
                    current_slots.update({k: v for k, v in new_slots.items() 
                                       if v and v.strip()})

            # Si l'utilisateur demande une info déjà connue, chercher en mémoire
            found_information = None
            if intent == "demande_information" and intent_result.get("search_query"):
                search_result = self._memory_agent.search_in_conversations(
                    intent_result["search_query"]
                )
                # Valider la confiance avant d'utiliser l'info
                if search_result["found"] and search_result["confidence"] in ["high", "medium"]:
                    found_information = search_result["information"]

            # 6. Vérifier les slots avec le ThresholdAgent
            threshold_result = self._threshold_agent.check_slots(
                intent=intent,
                filled_slots=current_slots,
                required_slots=required_slots.get(intent, [])
            )

            # 7. Générer la réponse appropriée
            if threshold_result["is_complete"]:
                # Effectuer une recherche web pour obtenir des informations à jour
                search_query = f"{'restaurant' if intent == 'restaurant_search' else 'hôtel' if intent == 'hotel_search' else 'activité'} {current_slots.get('food_type', '')} à {current_slots.get('location', '')} {current_slots.get('budget', '')} adresse horaires d'ouverture"
                search_results = self._search_agent.search_web(search_query)
                
                # Générer la réponse finale avec les résultats de recherche
                response = self._response_generator.generate_response(
                    message=message,
                    emotion=current_emotion,
                    intent=intent,
                    slots=current_slots,
                    search_results=search_results
                )
            else:
                # Vérifier tous les slots manquants
                missing_slots = [slot for slot in required_slots.get(intent, []) 
                               if not current_slots.get(slot)]
                
                # Générer la prochaine question
                response = self._response_generator.generate_question(
                    missing_slots=missing_slots,
                    filled_slots=current_slots,
                    message=message,
                    emotion=current_emotion
                )

            # 8. Sauvegarder l'interaction et mettre à jour les slots
            self._memory_agent.add_message(
                role="user",
                content=message,
                emotion=current_emotion,
                slots=current_slots,
                intent=intent
            )
            self._memory_agent.add_message(
                role="assistant",
                content=response,
                emotion="",  # Pas d'émotion pour l'assistant
                slots=current_slots,
                intent=intent
            )
            
            # Mettre à jour les slots dans la mémoire
            self._memory_agent._current_slots = current_slots.copy()

            # 9. Retourner le résultat structuré
            return {
                "success": True,
                "response": response,
                "context": conversation_history,
                "emotions": emotions,
                "slots": current_slots,
                "intent": intent,
                "found_information": found_information
            }

        except Exception:
            # En cas d'erreur, retourner success=False avec le message
            return {
                "success": False,
                "error": "Une erreur est survenue",
                "response": "Désolé, une erreur est survenue lors du traitement de votre message."
            }

    def generate_response(self, slots: Dict[str, Any], intent: str) -> str:
        """
        Proxy vers ResponseGeneratorAgent pour générer la réponse finale.

        Args:
            slots (Dict[str, Any]): Les slots disponibles.
            intent (str): L'intention détectée.

        Returns:
            str: Réponse générée par l'assistant.
        """
        return self._response_generator.generate_response(slots, intent)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Expose l'historique des conversations stockées.
        """
        return self._memory_agent.get_messages()

    def clear_memory(self) -> None:
        """
        Efface toute la mémoire de l'agent et réinitialise les slots.
        """
        self._memory_agent.clear_memory()
        # Réinitialiser les slots actuels
        self._memory_agent._current_slots = {
            "location": "",
            "food_type": "",
            "budget": "",
            "time": "",
        }

    def _extract_info(self, message: str) -> Dict[str, str]:
        """
        Extrait les informations pertinentes d'un message en utilisant le LLM.
        
        Args:
            message (str): Le message à analyser
            
        Returns:
            Dict[str, str]: Dictionnaire contenant les informations extraites
        """
        try:
            response = self._llm.chat(
                model=self._model_config["name"],
                messages=[
                    {"role": "system", "content": "Extrayez les informations au format JSON: {\"establishment\": \"nom\", \"location\": \"lieu\"}"},
                    {"role": "user", "content": message}
                ],
                options={"temperature": 0.1, "max_tokens": 200}
            )
            return json.loads(response["message"]["content"])
        except Exception:
            return {"establishment": "", "location": ""}