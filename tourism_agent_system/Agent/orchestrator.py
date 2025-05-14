from .base_agent import BaseAgent
from .memory_agent import MemoryAgent
from .emotion_detection_agent import EmotionDetectionAgent
from .intent_detection_agent import IntentDetectionAgent
from .threshold_agent import ThresholdAgent
from .search_agent import SearchAgent
from .response_generator_agent import ResponseGeneratorAgent
from .TrackingAgent import TrackingAgent

import requests
import json
from typing import Dict, Any, List, Optional

class AgentOrchestrator(BaseAgent):
    """
    Orchestrateur qui gère les interactions entre les différents agents.
    Hérite de la classe de base Agent pour charger nom, rôle, goal et backstory.
    """
    
    def __init__(self, name: str = "coordinator"):
        super().__init__(name)  # Initialise configuration et métadonnées via Agent
        # Configuration de l'API Mistral
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        
        # Instancier les agents auxiliaires
        self._memory_agent = MemoryAgent()                # A4: gère la mémoire
        self._emotion_agent = EmotionDetectionAgent()      # A3: détecte l'émotion
        self._intent_agent = IntentDetectionAgent()        # A5: extrait intent et slots
        self._response_generator = ResponseGeneratorAgent()# A7: génère les réponses
        self._threshold_agent = ThresholdAgent()           # A8: vérifie les slots
        self._search_agent = SearchAgent()                 # A9: effectue les recherches web
        self.tracking_agent = TrackingAgent()              # Agent de suivi
        
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Traite un message utilisateur en orchestrant les différents agents.
        """
        try:
            # Log de l'étape de détection d'intention
            self.tracking_agent.log_execution(
                agent_name="intent_detection",
                action="Détection de l'intention de l'utilisateur",
                status="démarrage"
            )
            
            # 1. Détection de l'intention
            intent_result = self._intent_agent.run(message)
            
            self.tracking_agent.log_execution(
                agent_name="intent_detection",
                action=f"Détection de l'intention: {intent_result['intent']}",
                status="succès"
            )
            
            # Log de l'étape de détection d'émotion
            self.tracking_agent.log_execution(
                agent_name="emotion_detection",
                action="Analyse de l'état émotionnel",
                status="démarrage"
            )
            
            # 2. Détection de l'émotion
            emotion = self._emotion_agent.detect_emotion(message)
            
            self.tracking_agent.log_execution(
                agent_name="emotion_detection",
                action=f"Détection de l'émotion: {emotion['emotion']}",
                status="succès"
            )
            
            # Log de l'étape de recherche
            self.tracking_agent.log_execution(
                agent_name="search",
                action="Recherche d'informations pertinentes",
                status="démarrage"
            )
            
            # 3. Recherche d'informations
            search_results = self._search_agent.search(message)
            
            self.tracking_agent.log_execution(
                agent_name="search",
                action=f"Recherche terminée: {len(search_results)} résultats",
                status="succès"
            )
            
            # Log de l'étape de seuil
            self.tracking_agent.log_execution(
                agent_name="threshold",
                action="Vérification des seuils de confiance",
                status="démarrage"
            )
            
            # 4. Vérification des seuils
            threshold_check = self._threshold_agent.check_thresholds(
                intent=intent_result,
                emotion=emotion,
                search_results={"results": search_results}
            )
            
            self.tracking_agent.log_execution(
                agent_name="threshold",
                action=f"Vérification des seuils: {threshold_check['status']}",
                status="succès"
            )
            
            # Log de l'étape de mémoire
            self.tracking_agent.log_execution(
                agent_name="memory",
                action="Mise à jour de la mémoire contextuelle",
                status="démarrage"
            )
            
            # 5. Mise à jour de la mémoire
            self._memory_agent.add_message(
                role="user",
                content=message,
                emotion=emotion["emotion"],
                slots=intent_result["slots"],
                intent=intent_result["intent"]
            )
            
            self.tracking_agent.log_execution(
                agent_name="memory",
                action="Mise à jour de la mémoire terminée",
                status="succès"
            )
            
            # Log de l'étape de génération de réponse
            self.tracking_agent.log_execution(
                agent_name="response_generator",
                action="Génération de la réponse finale",
                status="démarrage"
            )
            
            # 6. Génération de la réponse
            response = self._response_generator.generate_response(
                message=message,
                emotion=emotion["emotion"],
                intent=intent_result["intent"],
                slots=intent_result["slots"],
                search_results=search_results
            )
            
            # Ajouter la réponse à la mémoire
            self._memory_agent.add_message(
                role="assistant",
                content=response,
                emotion=emotion["emotion"],
                intent=intent_result["intent"]
            )
            
            self.tracking_agent.log_execution(
                agent_name="response_generator",
                action="Réponse générée avec succès",
                status="succès"
            )
            
            # Log de l'étape finale
            self.tracking_agent.log_execution(
                agent_name="orchestrator",
                action="Traitement complet de la demande",
                status="succès"
            )
            
            return {
                "response": response,
                "success": True
            }
            
        except Exception as e:
            # Log de l'erreur
            self.tracking_agent.log_execution(
                agent_name="orchestrator",
                action="Erreur lors du traitement",
                status=f"erreur: {str(e)}"
            )
            raise

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

    def _call_mistral_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Appelle l'API Mistral pour obtenir une réponse.
        
        Args:
            messages (List[Dict[str, str]]): Liste des messages pour la conversation
            
        Returns:
            str: Réponse du modèle
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistral-tiny",
            "messages": messages,
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

    def _extract_info(self, message: str) -> Dict[str, str]:
        """
        Extrait les informations pertinentes d'un message en utilisant l'API Mistral.
        """
        try:
            messages = [
                {"role": "system", "content": "Extrayez les informations pertinentes du message au format JSON."},
                {"role": "user", "content": message}
            ]
            response = self._call_mistral_api(messages)
            return json.loads(response)
        except Exception:
            return {}