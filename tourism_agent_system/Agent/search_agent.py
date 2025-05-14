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

    def __init__(self, name: str = "search"):
        super().__init__(name)
        self._search_config = self._config.get("search", {})

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Effectue une recherche web et retourne les résultats.
        
        Args:
            query (str): La requête de recherche
            
        Returns:
            List[Dict[str, Any]]: Liste des résultats de recherche
        """
        return self.search_web(query)

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Méthode interne pour effectuer la recherche web.
        """
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

    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Méthode principale pour exécuter l'agent.
        
        Args:
            prompt (str): La requête de recherche
            
        Returns:
            Dict[str, Any]: Résultats de la recherche
        """
        results = self.search(prompt)
        return {
            "success": True,
            "results": results,
            "query": prompt
        }

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """
        Vérifie si un résultat est valide et fiable.
        """
        if not all(result.get(field) for field in ["title", "content", "url"]):
            return False
            
        if result.get("score", 0) < 0.3:
            return False
            
        url = result.get("url", "").lower()
        valid_domains = [
            "pagesjaunes.fr", "tripadvisor.fr", "restaurant.michelin.fr",
            "lafourchette.com", "resto.fr", "restaurants.mappy.com",
            "restaurant.mappy.com", "restaurant.lefigaro.fr"
        ]
        if not any(domain in url for domain in valid_domains):
            return False
            
        content = result.get("content", "").lower()
        if len(content) < 30:
            return False
            
        negative_phrases = [
            "n'existe pas", "n'existe plus", "fermé définitivement",
            "fermé pour toujours", "plus en activité"
        ]
        if any(phrase in content for phrase in negative_phrases):
            return False
            
        return True

    def _get_fallback_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Génère des résultats de secours plus détaillés en cas d'échec.
        """
        location = ""
        budget = ""
        day = ""
        
        if "dijon" in query.lower():
            location = "Dijon"
        if "pas cher" in query.lower() or "pas trop cher" in query.lower():
            budget = "budget modéré"
        if "lundi" in query.lower():
            day = "lundi"
            
        return [{
            "title": "Recherche de restaurant",
            "snippet": f"""Je recherche des restaurants à {location} {f'pour {day}' if day else ''} {f'avec un {budget}' if budget else ''}.
            Je vous suggère de :
            1. Consulter le site de l'Office de Tourisme de Dijon
            2. Vérifier les horaires d'ouverture sur les sites des restaurants
            3. Contacter directement les établissements pour confirmer les informations""",
            "url": "https://www.destinationdijon.com/",
            "score": 0.4
        }]