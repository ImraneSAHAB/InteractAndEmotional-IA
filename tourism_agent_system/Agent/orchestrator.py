from Agent import Agent
from memory_agent import MemoryAgent
from emotion_detection_agent import EmotionDetectionAgent
from response_generator_agent import ResponseGeneratorAgent
import ollama
from typing import Dict, Any, List, Optional
from intent_detection_agent import intentDetectionAgent
import json

import re

class AgentOrchestrator(Agent):
    """
    Orchestrateur qui gère les interactions entre les différents agents
    """
    
    def __init__(self, name: str = "coordinator"):
        super().__init__(name)
        # Initialiser le LLM avec la configuration
        self._llm = ollama.Client()
        self._model_config = self._config["model"]
        # Initialiser les agents
        self._memory_agent = MemoryAgent()
        self._emotion_agent = EmotionDetectionAgent()
        self._intent_agent = intentDetectionAgent()
        self._response_generator = ResponseGeneratorAgent()
        
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Traite un message et coordonne les actions des différents agents.
        
        Args:
            message (str): Le message de l'utilisateur
            
        Returns:
            Dict[str, Any]: Le résultat du traitement
        """
        try:
            # 1. Détecter les émotions dans le message
            emotions = self._emotion_agent.run(message)
            
            # 2. Récupérer l'historique des conversations
            conversation_history = self._memory_agent.get_messages()
            
            # 3. Détecter l'intention et extraire les slots via l'IntentDetectionAgent
            intent_result = self._intent_agent.run(message)
            intent = intent_result["intent"]
            slots = intent_result["slots"]
            
            # 4. Générer une réponse avec les informations disponibles
            response = self._response_generator.generate_response(slots, intent)
            
            # 5. Sauvegarder la conversation avec les émotions détectées, les slots et l'intent
            # Ajouter le message de l'utilisateur
            self._memory_agent.add_message(
                role="user",
                content=message,
                emotion=json.dumps(emotions) if emotions else None,
                slots=slots,
                intent=intent
            )
            
            # Ajouter la réponse de l'assistant
            self._memory_agent.add_message(
                role="assistant",
                content=response,
                slots=slots,
                intent=intent
            )
            
            return {
                "success": True,
                "response": response,
                "context": conversation_history,
                "emotions": emotions,
                "slots": slots,
                "intent": intent
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "Désolé, une erreur est survenue lors du traitement de votre message."
            }
    
    def generate_response(self, slots: Dict[str, Any], intent: str) -> str:
        """
        Génère une réponse contextuelle basée sur les informations disponibles.
        
        Args:
            slots (Dict[str, Any]): Les informations disponibles (slots)
            intent (str): L'intention détectée
            
        Returns:
            str: Une réponse contextuelle et utile
        """
        return self._response_generator.generate_response(slots, intent)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Récupère l'historique des conversations depuis ChromaDB.
        
        Returns:
            List[Dict[str, str]]: Liste des messages
        """
        return self._memory_agent.get_messages()
        
    def clear_memory(self) -> None:
        """
        Efface la mémoire de tous les agents.
        """
        self._memory_agent.clear_memory()
