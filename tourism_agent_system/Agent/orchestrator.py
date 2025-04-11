from Agent import Agent
from memory_agent import MemoryAgent
from emotion_detection_agent import EmotionDetectionAgent
import ollama
from typing import Dict, Any, List, Optional

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
            
            # 3. Construire le prompt avec le message, l'historique et les émotions
            prompt = self._build_prompt(message, conversation_history, emotions)
            
            # 4. Générer une réponse avec le LLM
            response = self._get_llm_response(prompt)
            
            # 5. Convertir la liste d'émotions en chaîne pour ChromaDB
            emotions_str = ", ".join(emotions)
            
            # 6. Sauvegarder la conversation avec les émotions détectées
            self._memory_agent.add_message("user", message, emotions_str)
            self._memory_agent.add_message("assistant", response)
            
            return {
                "success": True,
                "response": response,
                "context": conversation_history,
                "emotions": emotions
            }
            
        except Exception as e:
            print(f"Erreur dans l'orchestration: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Désolé, une erreur est survenue lors du traitement de votre message."
            }
    
    def _build_prompt(self, message: str, conversation_history: List[Dict[str, str]], emotions: List[str]) -> List[Dict[str, str]]:
        """
        Construit le prompt pour le LLM.
        
        Args:
            message (str): Le message de l'utilisateur
            conversation_history (List[Dict[str, str]]): L'historique des conversations
            emotions (List[str]): Les émotions détectées dans le message
            
        Returns:
            List[Dict[str, str]]: Le prompt formaté pour le LLM
        """
        # Commencer avec le message système
        system_message = f"Vous êtes {self._role}. {self._goal}"
        if emotions:
            emotions_str = ", ".join(emotions)
            system_message += f"\nL'utilisateur exprime les émotions suivantes : {emotions_str}. Adaptez votre réponse en conséquence."
            
        prompt = [{"role": "system", "content": system_message}]
        
        # Ajouter l'historique des conversations récentes
        for msg in conversation_history[-10:]:
            prompt.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        # Ajouter le message actuel
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
