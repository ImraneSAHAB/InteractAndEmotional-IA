from Agent import Agent
from memory_agent import MemoryAgent
from emotion_detection_agent import EmotionDetectionAgent
from response_generator_agent import ResponseGeneratorAgent
import ollama
from typing import Dict, Any, List, Optional
from intent_detection_agent import intentDetectionAgent
from dialogue_planner_agent import DialoguePlannerAgent
import json

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
        self._intent_agent = intentDetectionAgent()        # A5: extrait intent et slots
        self._response_generator = ResponseGeneratorAgent()# A7: génère les réponses
        self._dialogue_planner = DialoguePlannerAgent()    # A6: planifie la question suivante
        
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Traite un message utilisateur et retourne un dictionnaire complet.
        Étapes :
          1. Détection émotionnelle
          2. Chargement de l'historique
          3. Extraction d'intention et slots
          4. Recherche d'informations en mémoire si besoin
          5. Calcul des slots manquants
          6. Décision de poser une question, renvoyer info mémorisée ou générer réponse finale
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

            # 2. Récupérer l'historique des conversations
            conversation_history = self._memory_agent.get_messages()

            # 3. Détecter l'intention et extraire les slots
            intent_result = self._intent_agent.run(message)
            intent = intent_result["intent"]  # ex: 'restaurant_search'
            slots = intent_result["slots"]    # dict des slots extraits

            # Si l'utilisateur demande une info déjà connue, chercher en mémoire
            found_information = None
            if intent == "demande_information" and intent_result.get("search_query"):
                search_result = self._memory_agent.search_in_conversations(
                    intent_result["search_query"]
                )
                # Valider la confiance avant d'utiliser l'info
                if search_result["found"] and search_result["confidence"] in ["high", "medium"]:
                    found_information = search_result["information"]

            # 4. Définir les slots requis par intention
            required_slots = {
                "restaurant_search": ["location", "food_type", "budget", "time"],
                "activity_search": ["location", "activity_type", "date"],
                # ... autres intentions
            }

            # 5. Calculer les slots manquants
            missing_slots = [
                slot for slot in required_slots.get(intent, [])
                if slot not in slots or not slots[slot]
            ]

            # 6. Sélection de la réponse
            if missing_slots:
                # a) Question suivante selon slots manquants
                response = self._dialogue_planner.get_next_question(
                    intent,
                    slots,
                    required_slots[intent]
                )
            elif found_information:
                # b) Renvoyer l'information déjà mémorisée
                response = f"D'après nos conversations précédentes, {found_information}"
            else:
                # c) Générer la réponse finale avec les slots disponibles
                response = self._response_generator.generate_response(slots, intent)

            # 7. Sauvegarder l'interaction
            self._memory_agent.add_message(
                role="user",
                content=message,
                emotion=emotions[0] if emotions else "",
                slots=slots,
                intent=intent
            )
            self._memory_agent.add_message(
                role="assistant",
                content=response,
                emotion="",  # Pas d'émotion pour l'assistant
                slots=slots,
                intent=intent
            )

            # 8. Retourner le résultat structuré
            return {
                "success": True,
                "response": response,
                "context": conversation_history,
                "emotions": emotions,
                "slots": slots,
                "intent": intent,
                "found_information": found_information
            }

        except Exception as e:
            # En cas d'erreur, retourner success=False avec le message
            return {
                "success": False,
                "error": str(e),
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
        Efface toute la mémoire de l'agent.
        """
        self._memory_agent.clear_memory()
