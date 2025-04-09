from Agent import Agent
import ollama

class AgentOrchestrator(Agent):
    """
    Orchestrateur qui gère le ollama et un agent spécialisé
    
    """
    
    def __init__(self, name: str, tavily_api_key: str):
        super().__init__(name)
        # Initialise le LLM pour les réponses directes
        self.llm = ollama()
    
    def run(self, prompt: str) -> str:
        """
        Si le prompt contient "search:", on extrait la requête et on délègue
        à l'agent pertinent. Sinon, le LLM répond directement.
        """
        prompt_lower = prompt.lower()
        
        if "search:" in prompt_lower:
            query_part = prompt_lower.split("search:")[1].strip()
            answer = self.search_agent.run(query_part)
            return answer
        else:
            answer = self.llm.respond(prompt)
            return answer
