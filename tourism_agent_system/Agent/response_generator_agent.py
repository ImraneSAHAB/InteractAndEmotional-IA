from Agent import Agent
import ollama
from typing import Dict, Any, List, Optional
import re

class ResponseGeneratorAgent(Agent):
    """
    Agent qui génère des réponses contextuelles et des questions naturelles
    """
    
    def __init__(self, name: str = "response_generator"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = self._config["model"]
        
    def extract_slots(self, message: str) -> Dict[str, Any]:
        """
        Extrait les informations (slots) du message.
        
        Args:
            message (str): Le message de l'utilisateur
            
        Returns:
            Dict[str, Any]: Les informations extraites
        """
        # Informations par défaut
        slots = {
            "location": None,
            "food_type": None,
            "budget": None,
            "time": None
        }
        
        # Extraction simple basée sur des mots-clés
        message_lower = message.lower()
        
        # Localisation - Détection plus globale des villes
        # Recherche de motifs comme "à [ville]", "dans [ville]", "en [ville]", etc.
        location_patterns = [
            r'à\s+([a-zéèêëàâäôöûüçîï]+)',
            r'dans\s+([a-zéèêëàâäôöûüçîï]+)',
            r'en\s+([a-zéèêëàâäôöûüçîï]+)',
            r'à\s+([a-zéèêëàâäôöûüçîï]+\s+[a-zéèêëàâäôöûüçîï]+)',
            r'dans\s+([a-zéèêëàâäôöûüçîï]+\s+[a-zéèêëàâäôöûüçîï]+)',
            r'en\s+([a-zéèêëàâäôöûüçîï]+\s+[a-zéèêëàâäôöûüçîï]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, message_lower)
            if match:
                # Capitaliser la première lettre de chaque mot
                city = match.group(1).title()
                slots["location"] = city
                break
        
        # Si aucune ville n'est trouvée avec les patterns, essayer avec une liste de villes communes
        if slots["location"] is None:
            common_cities = ["dijon", "paris", "lyon", "marseille", "bordeaux", "toulouse", "lille", "nantes", "strasbourg", "rennes"]
            for city in common_cities:
                if city in message_lower:
                    slots["location"] = city.title()
                    break
            
        # Type de nourriture
        if "tradition" in message_lower or "traditionnel" in message_lower:
            slots["food_type"] = "traditional"
        elif "moderne" in message_lower or "contemporain" in message_lower:
            slots["food_type"] = "modern"
        elif "végétarien" in message_lower or "végé" in message_lower:
            slots["food_type"] = "vegetarian"
            
        # Budget
        if "pas cher" in message_lower or "économique" in message_lower or "budget" in message_lower:
            slots["budget"] = "budget"
        elif "moyen" in message_lower or "modéré" in message_lower:
            slots["budget"] = "mid-range"
        elif "luxe" in message_lower or "haut de gamme" in message_lower or "gastronomique" in message_lower:
            slots["budget"] = "luxury"
            
        # Temps
        if "ce soir" in message_lower or "tonight" in message_lower:
            slots["time"] = "tonight"
        elif "demain" in message_lower or "tomorrow" in message_lower:
            slots["time"] = "tomorrow"
        elif "weekend" in message_lower or "semaine" in message_lower:
            slots["time"] = "this_weekend"
            
        return slots
        
    def check_missing_info(self, slots: Dict[str, Any]) -> Optional[str]:
        """
        Vérifie si des informations sont manquantes.
        
        Args:
            slots (Dict[str, Any]): Les informations extraites
            
        Returns:
            Optional[str]: L'information manquante ou None si toutes les informations sont présentes
        """
        # Liste des informations requises
        required_info = ["location", "food_type", "budget", "time"]
        
        # Vérifier si une information est manquante
        for info in required_info:
            if slots.get(info) is None:
                return info
                
        return None
        
    def generate_question(self, missing_info: str, emotion: str, context: Dict[str, Any]) -> str:
        """
        Génère une question naturelle pour collecter des informations manquantes.
        
        Args:
            missing_info (str): L'information manquante à collecter
            emotion (str): L'émotion détectée chez l'utilisateur
            context (Dict[str, Any]): Le contexte actuel de la conversation
            
        Returns:
            str: Une question naturelle pour collecter l'information manquante
        """
        # Construire le prompt pour générer une question
        prompt = [
            {"role": "system", "content": f"Vous êtes {self._role}. {self._goal}"},
            {"role": "user", "content": f"""
            Générer une question naturelle pour collecter l'information manquante suivante: "{missing_info}".
            
            Contexte actuel:
            {self._format_context(context)}
            
            Émotion de l'utilisateur: {emotion}
            
            La question doit être:
            1. Naturelle et conversationnelle
            2. Adaptée à l'émotion de l'utilisateur
            3. Pertinente par rapport au contexte
            4. Directe mais polie
            
            Répondez uniquement avec la question, sans explications supplémentaires.
            """}
        ]
        
        # Obtenir la réponse du LLM
        response = self._get_llm_response(prompt)
        return response.strip()
        
    def generate_response(self, slots: Dict[str, Any]) -> str:
        """
        Génère une réponse contextuelle basée sur les informations disponibles.
        
        Args:
            slots (Dict[str, Any]): Les informations disponibles (slots)
            
        Returns:
            str: Une réponse contextuelle et utile
        """
        # Construire le prompt pour générer une réponse
        prompt = [
            {"role": "system", "content": f"Vous êtes {self._role}. {self._goal}"},
            {"role": "user", "content": f"""
            Générer une réponse utile et polie avec les informations suivantes:
            
            {self._format_slots(slots)}
            
            La réponse doit être:
            1. Informative et pertinente
            2. Polie et professionnelle
            3. Structurée de manière claire
            4. Inclure des recommandations ou un résumé approprié
            5. Utiliser explicitement les informations fournies
            
            Si une ville est mentionnée, utilisez-la dans votre réponse.
            Si un type de nourriture est mentionné, adaptez votre réponse en conséquence.
            
            Répondez uniquement avec la réponse, sans explications supplémentaires.
            """}
        ]
        
        # Obtenir la réponse du LLM
        response = self._get_llm_response(prompt)
        return response.strip()
        
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Formate le contexte pour l'inclure dans un prompt.
        
        Args:
            context (Dict[str, Any]): Le contexte à formater
            
        Returns:
            str: Le contexte formaté
        """
        if not context:
            return "Aucun contexte disponible"
            
        formatted = []
        for key, value in context.items():
            formatted.append(f"- {key}: {value}")
            
        return "\n".join(formatted)
        
    def _format_slots(self, slots: Dict[str, Any]) -> str:
        """
        Formate les slots pour l'inclure dans un prompt.
        
        Args:
            slots (Dict[str, Any]): Les slots à formater
            
        Returns:
            str: Les slots formatés
        """
        if not slots:
            return "Aucune information disponible"
            
        formatted = []
        for key, value in slots.items():
            formatted.append(f"- {key}: {value}")
            
        return "\n".join(formatted)
        
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
        
    def run(self, prompt: str) -> str:
        """
        Méthode générique pour exécuter l'agent.
        
        Args:
            prompt (str): Le prompt à traiter
            
        Returns:
            str: La réponse générée
        """
        # Cette méthode est requise par la classe de base mais n'est pas utilisée
        # car nous avons des méthodes spécifiques pour générer des questions et des réponses
        return "Cette méthode n'est pas utilisée pour cet agent." 