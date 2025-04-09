import json
from typing import Dict, Any

class Agent:
    """Classe de base pour un agent IA."""
    
    def __init__(self, name: str):
        self._config = self._load_config()
        self._name = self._config["agents"][name]["name"]
        self._role = self._config["agents"][name]["role"]
        self._goal = self._config["agents"][name]["goal"]
        self._backstory = self._config["agents"][name]["backstory"]

    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier config.json"""
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Le fichier config.json est introuvable")
        except json.JSONDecodeError:
            raise ValueError("Le fichier config.json est mal formaté")
            


    @property
    def name(self) -> str:
        return self._name
        
    @property
    def role(self) -> str:
        return self._role
        
    @property
    def goal(self) -> str:
        return self._goal
        
    @property
    def backstory(self) -> str:
        return self._backstory

    def run(self, prompt: str):
        """Méthode générique à redéfinir par les sous-classes."""
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes.")