# search_agent.py

import json
import requests
from Agent import Agent
from typing import Dict, Any, List

class SearchAgent(Agent):
    """
    Agent responsable d'effectuer des recherches web via une API et de renvoyer les résultats.

    Hérite de la classe de base Agent pour charger nom, rôle, goal et backstory depuis config.json.
    """

    def __init__(self, name: str = "searcher"):
        super().__init__(name)
        # Charger la configuration de recherche depuis le config.json
        self._search_config = self._config.get("search", {})

    def run(self, query: str) -> List[Dict[str, Any]]:
        """
        Surcharge de la méthode run() pour effectuer directement la recherche.

        Args:
            query (str): La requête à rechercher.

        Returns:
            List[Dict[str, Any]]: Liste des résultats structurés.
        """
        return self.search_web(query)

    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Réalise une requête POST vers l'API de recherche configurée et retourne
        une liste de résultats structurés.

        Args:
            query (str): La requête à rechercher.

        Returns:
            List[Dict[str, Any]]: Liste des résultats avec titre, snippet, url, source_type, published_date.
        """
        try:
            # Préparer la payload en mixant api_key et paramètres globaux
            payload = {
                "api_key": self._search_config.get("api_key", ""),
                "query": query,
                **{k: v for k, v in self._search_config.items() if k not in ["api_key"]}
            }
            response = requests.post(
                self._search_config.get("url", ""),
                json=payload,
                timeout=self._search_config.get("timeout", 15)
            )
            response.raise_for_status()  # Lever une exception si le statut HTTP est une erreur
            data = response.json()

            # Vérifier que les données contiennent les champs attendus
            if not isinstance(data, dict) or "results" not in data:
                raise ValueError("La réponse de l'API est mal formatée ou ne contient pas de résultats.")

            # Parser les résultats
            results = self._parse_results(data)
            if not results:
                return [{
                    "title": "Aucun résultat",
                    "snippet": "Essayez de reformuler votre requête.",
                    "url": "N/A",
                    "source_type": "aucun"
                }]
            return results

        except requests.exceptions.RequestException as e:
            return [{
                "title": "Erreur de connexion",
                "snippet": f"Impossible de se connecter à l'API : {e}",
                "url": "N/A",
                "source_type": "erreur"
            }]
        except Exception as e:
            return [{
                "title": "Erreur",
                "snippet": str(e),
                "url": "N/A",
                "source_type": "erreur"
            }]

    def _parse_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transforme la réponse brute de l'API en une liste de dicts normalisés.
        """
        results: List[Dict[str, Any]] = []
        # Ajouter le résumé si présent
        if data.get("answer"):
            results.append({
                "title": "Résumé",
                "snippet": data["answer"],
                "url": "API résumé",
                "source_type": "résumé"
            })
        # Ajouter les résultats web
        for r in data.get("results", [])[: self._search_config.get("max_results", 5)]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
                "url": r.get("url", ""),
                "source_type": "web",
                "published_date": r.get("published_date", "non disponible")
            })
        return results