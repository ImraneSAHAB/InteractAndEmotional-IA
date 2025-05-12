# search_agent.py

import json
import requests
from .base_agent import BaseAgent
from typing import Dict, Any, List

class SearchAgent(BaseAgent):
    """
    Agent responsable d'effectuer des recherches web via une API et de renvoyer les résultats.

    Hérite de la classe de base Agent pour charger nom, rôle, goal et backstory depuis config.json.
    """

    def __init__(self, name: str = "searcher"):
        super().__init__(name)
        self._search_config = self._config.get("search", {})

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        try:
            if not self._search_config.get("api_key") or not self._search_config.get("url"):
                return self._get_fallback_results(query)

            response = requests.post(
                self._search_config.get("url", ""),
                json={
                    "api_key": self._search_config.get("api_key", ""),
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True
                },
                timeout=15
            )
            
            response.raise_for_status()
            data = response.json()
            
            results = []
            for r in data.get("results", []):
                if self._is_valid_result(r):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "url": r.get("url", ""),
                        "score": r.get("score", 0)
                    })
                    if len(results) >= 3:
                        break
                        
            return results if results else self._get_fallback_results(query)

        except Exception:
            return self._get_fallback_results(query)

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """
        Vérifie si un résultat est valide et fiable.
        """
        if not all(result.get(field) for field in ["title", "content", "url"]):
            return False
            
        if result.get("score", 0) < 0.5:
            return False
            
        url = result.get("url", "").lower()
        if not any(domain in url for domain in ["pagesjaunes.fr", "tripadvisor.fr"]):
            return False
            
        content = result.get("content", "").lower()
        if len(content) < 50 or "n'existe pas" in content or "n'existe plus" in content:
            return False
            
        return True

    def _get_fallback_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Génère des résultats de secours en cas d'échec.
        """
        return [{
            "title": "Information non disponible",
            "snippet": f"Je n'ai pas pu trouver d'informations fiables pour {query}. Je vous suggère de consulter directement le site web officiel ou de contacter l'établissement.",
            "url": "N/A",
            "score": 0
        }]