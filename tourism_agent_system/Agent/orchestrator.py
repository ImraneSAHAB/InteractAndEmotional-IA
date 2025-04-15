from Agent import Agent
from memory_agent import MemoryAgent
from emotion_detection_agent import EmotionDetectionAgent
from response_generator_agent import ResponseGeneratorAgent
import ollama
from typing import Dict, Any, List, Optional
from intent_detection_agent import intentDetectionAgent

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
            
            # 2. Détection de l'intention de l'utilisateur 
            intent_result = self._intent_agent.run(message) 
            intent = intent_result.get("intent", "unknown")  
            slots = intent_result.get("slots", {})     
            
            # 3. Récupérer l'historique des conversations
            conversation_history = self._memory_agent.get_messages()
            
            # 4. Construire le prompt avec le message, l'historique et les émotions
            prompt = self._build_prompt(message, conversation_history, emotions, intent, slots)
            
            # 5. Générer une réponse avec le LLM
            response = self._get_llm_response(prompt)
            # 3. Extraire les informations du message (slots) via le générateur de réponses
            slots = self._response_generator.extract_slots(message)
            
            # 4. Vérifier si des informations sont manquantes via le générateur de réponses
            missing_info = self._response_generator.check_missing_info(slots)
            
            if missing_info:
                # 5a. Générer une question pour collecter l'information manquante
                emotion_str = emotions[0] if emotions else "neutral"
                context = {k: v for k, v in slots.items() if v is not None}
                response = self.generate_question(missing_info, emotion_str, context)
            else:
                # 5b. Générer une réponse avec les informations disponibles
                response = self.generate_response(slots)
            
            # 6. Convertir la liste d'émotions en chaîne pour ChromaDB
            emotions_str = ", ".join(emotions)
            
            # 7. Sauvegarder la conversation avec les émotions détectées
            self._memory_agent.add_message("user", message, emotions_str, slots)
            self._memory_agent.add_message("assistant", response)
            
            return {
                "success": True,
                "response": response,
                "context": conversation_history,
                "emotions": emotions,
                "intent": intent,
                "slots": slots
            }
            
        except Exception as e:
            print(f"Erreur dans l'orchestration: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Désolé, une erreur est survenue lors du traitement de votre message."
            }
    
    def generate_question(self, missing_info: str, emotion: str, context: Dict[str, Any]) -> str:
        """
        Génère une question naturelle pour collecter des informations manquantes.
        
        Args:
            missing_info (str): L'information manquante à collecter
            emotion (str): L'émotion détectée chez l'utilisateur
            context (Dict[str, Any]): Le contexte actuel de la conversation
            
        Returns:
            str: Une question naturelle pour collecter l'information manquante
        """
        return self._response_generator.generate_question(missing_info, emotion, context)
        
    def generate_response(self, slots: Dict[str, Any]) -> str:
        """
        Génère une réponse contextuelle basée sur les informations disponibles.
        
        Args:
            slots (Dict[str, Any]): Les informations disponibles (slots)
            
        Returns:
            str: Une réponse contextuelle et utile
        """
        return self._response_generator.generate_response(slots)
    
    def _build_prompt(self, message: str, conversation_history: List[Dict[str, str]], emotions: List[str], intent: str, slots: Dict[str, str]) -> List[Dict[str, str]]:
        system_message = f"Vous êtes {self._role}. {self._goal}"

        if emotions:
            system_message += f"\nL'utilisateur exprime les émotions suivantes : {', '.join(emotions)}."

        if intent and intent != "unknown":
            system_message += f"\nL'intention détectée est : {intent}."
            if slots:
                slots_str = ", ".join([f"{k}: {v}" for k, v in slots.items()])
                system_message += f" Les informations extraites (slots) sont : {slots_str}."

        prompt = [{"role": "system", "content": system_message}]

        for msg in conversation_history[-10:]:
            prompt.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        prompt.append({
            "role": "user",
            "content": message
        })

        return prompt

    
    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Obtient une réponse du LLM.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer au LLM
            
        Returns:
            str: La réponse du LLM
        """
        response = self._llm.chat(
            model=self._model_config["name"],
            messages=prompt,
            options={
                "temperature": self._model_config["temperature"],
                "max_tokens": self._model_config["max_tokens"]
            }
        )
        
        return response["message"]["content"]
            
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
