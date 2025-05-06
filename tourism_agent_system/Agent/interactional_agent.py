import logging
from typing import Optional
from .Agent import Agent
from .orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

class InteractionalAgent(Agent):
    """
    Agent qui gère la boucle d'interaction avec l'utilisateur
    """
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        prompt: str = "Vous: ",
        typing_delay: float = 0.03
    ):
        super().__init__(name="interactional")
        self.orchestrator = orchestrator
        self.prompt = prompt
        self.typing_delay = typing_delay

    def run(self) -> None:
        self._display_welcome()
        while True:
            try:
                user_input = self._read_input()
                if user_input in ("", "quit"):
                    self._say("Au revoir ! À bientôt !")
                    break
                if user_input == "clear":
                    self.orchestrator.clear_memory()
                    self._say("Historique effacé.")
                    continue

                result = self.orchestrator.process_message(user_input)
                if result["success"]:
                    self._print_typing(result["response"])
                else:
                    logger.error("Erreur de traitement : %s", result.get("error"))
                    self._say(f"Erreur interne : {result.get('error')}")
            except KeyboardInterrupt:
                self._say("\nInterruption, fermeture du programme.")
                break
            except Exception as e:
                logger.exception("Exception inattendue")
                self._say(f"Erreur inattendue : {e}")

    def _display_welcome(self) -> None:
        banner = "\nBienvenue dans le système d'agent touristique !"
        self._say(banner)
        self._say("Tapez 'quit' pour quitter ou 'clear' pour effacer l'historique.")
        self._say("-" * 50)

    def _read_input(self) -> str:
        return input(self.prompt).strip()

    def _say(self, message: str) -> None:
        print(message, flush=True)

    def _print_typing(self, text: str) -> None:
        for word in text.split():
            print(word, end=" ", flush=True)
            time.sleep(self.typing_delay)
        print()

        

if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    agent = InteractionalAgent(orchestrator=orchestrator)
    agent.run()
