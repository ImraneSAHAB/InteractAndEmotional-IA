from Agent import Agent
from memory_agent import MemoryAgent
from emotion_detection_agent import EmotionDetectionAgent
from response_generator_agent import ResponseGeneratorAgent
from threshold_agent import ThresholdAgent
import ollama
from typing import Dict, Any, List, Optional
from intent_detection_agent import IntentDetectionAgent
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
        self._intent_agent = IntentDetectionAgent()        # A5: extrait intent et slots
        self._response_generator = ResponseGeneratorAgent()# A7: génère les réponses
        self._dialogue_planner = DialoguePlannerAgent()    # A6: planifie la question suivante
        self._threshold_agent = ThresholdAgent()           # A8: vérifie les slots
        
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
            intent = intent_result["intent"]  # ex: 'restaurant_search'
            new_slots = intent_result["slots"]    # dict des slots extraits

            # 4. Fusionner les nouveaux slots avec les slots existants
            current_slots = self._memory_agent._current_slots.copy()
            
            # Vérifier si c'est une sélection de restaurant
            is_selection = False
            if intent == "restaurant_search":
                # Utiliser le LLM pour détecter si c'est une sélection
                prompt = [
                    {"role": "system", "content": """Vous êtes un assistant qui analyse les messages des utilisateurs.
                    Votre tâche est de déterminer si le message est une sélection de restaurant parmi des suggestions.
                    
                    Répondez uniquement par "oui" si c'est une sélection, "non" sinon.
                    
                    Exemples de sélections:
                    - "Je choisis le premier"
                    - "Je prends le Wagamama"
                    - "Le premier restaurant"
                    - "La deuxième option"
                    
                    Exemples de non-sélections:
                    - "Je veux un restaurant chinois"
                    - "Quels sont les restaurants disponibles ?"
                    - "Je cherche un restaurant pas cher"
                    """},
                    {"role": "user", "content": f"Le message suivant est-il une sélection de restaurant ? : {message}"}
                ]
                
                response = self._llm.chat(
                    model=self._model_config["name"],
                    messages=prompt,
                    options={
                        "temperature": 0.1,
                        "max_tokens": 10
                    }
                )
                
                is_selection = "oui" in response["message"]["content"].lower()
            
            if is_selection:
                # Conserver tous les slots précédents
                pass
            else:
                # Mettre à jour les slots avec les nouvelles valeurs
                for key, value in new_slots.items():
                    if value is not None:  # Mettre à jour même si la valeur est vide
                        current_slots[key] = value

            # Si l'utilisateur demande une info déjà connue, chercher en mémoire
            found_information = None
            if intent == "demande_information" and intent_result.get("search_query"):
                search_result = self._memory_agent.search_in_conversations(
                    intent_result["search_query"]
                )
                # Valider la confiance avant d'utiliser l'info
                if search_result["found"] and search_result["confidence"] in ["high", "medium"]:
                    found_information = search_result["information"]

            # 5. Définir les slots requis par intention
            required_slots = {
                "restaurant_search": ["location", "food_type", "budget", "time"],
                "activity_search": ["location", "activity_type", "date"],
                "hotel_search": ["location", "date", "budget"]
            }

            # 6. Vérifier les slots avec le ThresholdAgent
            threshold_result = self._threshold_agent.check_slots(
                intent=intent,
                filled_slots=current_slots,
                required_slots=required_slots.get(intent, [])
            )

            # 7. Sélection de la réponse
            if threshold_result["is_complete"]:
                # Générer la réponse finale
                response = self._response_generator.generate_response(
                    message=message,
                    emotion=current_emotion,
                    intent=intent,
                    slots=current_slots
                )
            else:
                # Vérifier tous les slots manquants
                missing_slots = []
                for slot in required_slots.get(intent, []):
                    if not current_slots.get(slot):
                        missing_slots.append(slot)
                
                # Générer la prochaine question
                response = self._response_generator.generate_question(
                    missing_slots=missing_slots,
                    filled_slots=current_slots,
                    message=message,
                    emotion=current_emotion
                )

            # 8. Sauvegarder l'interaction
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
