class Agent:
    """Classe de base pour un agent IA."""
    
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def run(self, prompt: str):
        """Méthode générique à redéfinir par les sous-classes."""
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes.")