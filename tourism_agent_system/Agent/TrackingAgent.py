import datetime
from typing import List, Dict, Any
from .base_agent import BaseAgent
import requests
import json

class TrackingAgent(BaseAgent):
    """
    Agent de suivi qui analyse les interactions entre agents et génère des insights.
    Utilise Mistral pour analyser les patterns et générer des recommandations.
    """
    def __init__(self, name: str = "tracking"):
        super().__init__(name)
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        self.logs: List[Dict[str, str]] = []
        self.execution_sequence: List[Dict[str, str]] = []

    def log_execution(self, agent_name: str, action: str, status: str = "succès"):
        """Enregistre une étape d'exécution dans la séquence."""
        self.execution_sequence.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": agent_name,
            "action": action,
            "status": status
        })

    def _get_llm_response(self, prompt: List[Dict[str, str]]) -> str:
        """
        Obtient une réponse de l'API Mistral.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer au LLM
            
        Returns:
            str: La réponse du LLM
        """
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "mistral-tiny",
                "messages": prompt,
                "temperature": self._model_config["temperature"],
                "max_tokens": self._model_config["max_tokens"]
            }
            
            response = requests.post(
                f"{self._api_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Erreur API Mistral: {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API Mistral: {e}")
            raise

    def log(self, agent_name: str, input_data: str, output_data: str):
        """Ajoute une entrée horodatée pour un agent donné."""
        self.logs.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent": agent_name,
            "input": input_data,
            "output": output_data
        })

    def analyze_interactions(self) -> Dict[str, Any]:
        """Analyse les interactions avec Mistral et retourne des insights."""
        if not self.logs:
            return {"status": "warning", "message": "Aucun log à analyser"}
        
        # Préparation des logs pour l'analyse
        logs_text = "\n".join([
            f"Agent: {log['agent']}\nEntrée: {log['input']}\nSortie: {log['output']}\n"
            for log in self.logs[-10:]  # Analyse les 10 dernières interactions
        ])

        # Préparation de la séquence d'exécution
        sequence_text = "\n".join([
            f"- [{step['agent']}] {step['action']} (Statut: {step['status']})"
            for step in self.execution_sequence[-10:]  # Dernières 10 étapes
        ])
        
        # Analyse avec Mistral
        try:
            prompt = [
                {"role": "system", "content": """Tu es un expert en analyse de systèmes d'agents IA.
                Analyse les logs d'interaction et fournis des insights pertinents en français sur :
                1. Les patterns d'interaction entre les agents
                2. Les points d'amélioration potentiels
                3. Les problèmes détectés
                4. Les recommandations d'optimisation
                
                Structure ton analyse de la manière suivante :
                
                ## Séquence d'Exécution
                [Liste des étapes d'exécution des agents]
                
                ## Patterns d'Interaction
                [Analyse des patterns observés]
                
                ## Points d'Amélioration
                [Liste des points à améliorer]
                
                ## Problèmes Détectés
                [Description des problèmes identifiés]
                
                ## Recommandations
                [Suggestions d'optimisation]
                
                Réponds de manière structurée et concise, en français."""},
                {"role": "user", "content": f"""Analyse ces informations :

Séquence d'exécution des agents :
{sequence_text}

Logs d'interaction :
{logs_text}"""}
            ]
            
            response = self._get_llm_response(prompt)
            return {
                "status": "success",
                "analysis": response,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "execution_sequence": self.execution_sequence[-10:]  # Inclure les 10 dernières étapes
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur lors de l'analyse : {str(e)}"
            }

    def write_report(self, filepath: str = "agent_analysis_report.md"):
        """Génère un rapport détaillé incluant les logs et l'analyse."""
        analysis = self.analyze_interactions()
        
        lines = [
            "# Rapport d'Analyse des Agents\n",
            f"*Généré le {datetime.datetime.utcnow().strftime('%d/%m/%Y à %H:%M:%S')}*\n",
            "## Séquence d'Exécution\n",
            "```\n",
            "\n".join([
                f"- [{step['agent']}] {step['action']} (Statut: {step['status']})"
                for step in self.execution_sequence
            ]),
            "\n```\n",
            "## Analyse des Interactions\n",
            f"```\n{analysis.get('analysis', 'Pas d\'analyse disponible')}\n```\n",
            "## Logs Détaillés\n"
        ]
        
        for entry in self.logs:
            lines.extend([
                f"### [{entry['timestamp']}] {entry['agent']}",
                f"- **Entrée** : `{entry['input']}`",
                f"- **Sortie** : `{entry['output']}`\n"
            ])
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def run(self, prompt: str) -> Dict[str, Any]:
        """Implémentation de la méthode run de BaseAgent."""
        return self.analyze_interactions()
