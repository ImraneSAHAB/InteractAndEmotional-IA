from Agent import Agent
from memory_agent import MemoryAgent
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
        
    def process_message(self, message: str, emotion: Optional[str] = None) -> Dict[str, Any]:
        """
        Traite un message et coordonne les actions des différents agents.
        
        Args:
            message (str): Le message de l'utilisateur
            emotion (Optional[str]): L'émotion détectée dans le message
            
        Returns:
            Dict[str, Any]: Le résultat du traitement
        """
        try:
            # 1. Récupérer le contexte historique depuis ChromaDB via MemoryAgent
            conversation_history = self._memory_agent.get_messages()
            
            # 2. Construire le prompt avec le contexte et l'émotion
            prompt = self._build_prompt(message, conversation_history, emotion)
            
            # 3. Générer une réponse avec le LLM
            response = self._get_llm_response(prompt)
            
            # 4. Sauvegarder la conversation dans ChromaDB via MemoryAgent
            self._memory_agent.add_message("user", message, emotion)
            self._memory_agent.add_message("assistant", response)
            
            return {
                "success": True,
                "response": response,
                "context": conversation_history,
                "emotion": emotion
            }
            
        except Exception as e:
            print(f"Erreur dans l'orchestration: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Désolé, une erreur est survenue lors du traitement de votre message."
            }
    
    def _build_prompt(self, message: str, conversation_history: List[Dict[str, str]], emotion: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Construit le prompt pour le LLM en incluant le contexte et l'émotion.
        
        Args:
            message (str): Le message de l'utilisateur
            conversation_history (List[Dict[str, str]]): L'historique des conversations
            emotion (Optional[str]): L'émotion détectée dans le message
            
        Returns:
            List[Dict[str, str]]: Le prompt formaté pour le LLM
        """
        # Commencer avec le message système
        system_message = f"Vous êtes {self._role}. {self._goal}"
        if emotion:
            system_message += f"\nL'utilisateur exprime une émotion de {emotion}. Adaptez votre réponse en conséquence."
            
        prompt = [{"role": "system", "content": system_message}]
        
        # Ajouter l'historique des conversations récentes (limité aux 5 derniers échanges)
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
