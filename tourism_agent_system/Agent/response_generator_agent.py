from .base_agent import BaseAgent
import requests
from typing import Dict, Any, List, Optional
import json
import re

class ResponseGeneratorAgent(BaseAgent):
    """
    Agent qui génère des réponses contextuelles basées sur l'intent, les slots et l'émotion de l'utilisateur
    """
    
    def __init__(self, name: str = "response"):
        super().__init__(name)
        self._model_config = self._config["model"]
        self._api_key = self._model_config["api_key"]
        self._api_url = self._model_config["api_url"]
        
    def generate_response(self, message: str, emotion: str, intent: str, slots: Dict[str, Any], search_results: List[Dict[str, Any]] = None) -> str:
        """
        Génère une réponse finale basée sur les informations disponibles.
        """
        try:
            if intent == "restaurant_search":
                prompt = [
                    {"role": "system", "content": """Vous êtes un assistant touristique qui aide les utilisateurs à trouver des restaurants.
                    Votre tâche est de fournir une réponse utile et informative basée sur les informations disponibles.
                    
                    Instructions:
                    1. Si vous avez des informations vérifiées sur des restaurants, mentionnez-les en priorité
                    2. Pour chaque restaurant mentionné, incluez :
                       - Le nom exact
                       - L'adresse (si disponible)
                       - Les horaires d'ouverture (si disponibles)
                       - Le budget moyen (si disponible)
                    3. Si vous n'avez pas toutes les informations pour un restaurant mais que vous avez des informations partielles fiables, vous pouvez les mentionner en précisant ce qui est vérifié
                    4. Si vous n'avez pas d'informations vérifiées sur des restaurants spécifiques :
                       - Suggérez des sources fiables pour trouver l'information (sites web officiels, etc.)
                       - Proposez des alternatives (autres jours, autres quartiers, etc.)
                       - Demandez des précisions si nécessaire
                    5. Adaptez votre ton à l'émotion de l'utilisateur
                    6. Soyez concis mais informatif
                    7. Terminez par une question ouverte ou une suggestion d'action"""},
                    {"role": "user", "content": f"""
                    Message: {message}
                    Émotion: {emotion}
                    Intention: {intent}
                    Informations disponibles:
                    - Localisation: {slots.get('location')}
                    - Type de cuisine: {slots.get('food_type')}
                    - Budget: {slots.get('budget')}
                    
                    Résultats de recherche web:
                    {json.dumps(search_results, indent=2) if search_results else "Aucun résultat de recherche disponible"}
                    
                    Génère une réponse utile basée sur les informations disponibles."""}
                ]
            else:
                prompt = [
                    {"role": "system", "content": """Vous êtes un assistant touristique qui aide les utilisateurs.
                    Votre tâche est de fournir une réponse finale basée sur les informations disponibles.
                    
                    Instructions:
                    1. Soyez concis et direct
                    2. Ne faites pas de suppositions
                    3. Si des informations sont manquantes, demandez-les
                    4. Terminez par une question ouverte"""},
                    {"role": "user", "content": f"""
                    Message: {message}
                    Émotion: {emotion}
                    Intention: {intent}
                    Informations disponibles:
                    {self._format_known_slots(slots)}
                    
                    Génère une réponse appropriée."""}
                ]
            
            response = self._get_llm_response(prompt)
            return response.strip()
            
        except Exception:
            return "Désolé, je n'ai pas pu générer une réponse appropriée. Veuillez réessayer."

    def generate_question(self, missing_slots: List[str], filled_slots: Dict[str, Any], message: str, emotion: str) -> str:
        """
        Génère une question naturelle pour obtenir les informations manquantes.

        Args:
            missing_slots (List[str]): Liste des slots manquants
            filled_slots (Dict[str, Any]): Les slots déjà remplis
            message (str): Le message de l'utilisateur
            emotion (str): L'émotion détectée
            
        Returns:
            str: Une question naturelle
        """
        try:
            prompt = [
                {"role": "system", "content": """Vous êtes un assistant touristique qui aide les utilisateurs.
                Votre tâche est de poser une question naturelle pour obtenir les informations manquantes nécessaires.
                
                Instructions importantes:
                1. Adaptez votre ton à l'émotion de l'utilisateur
                2. Posez UNE SEULE question claire et naturelle
                3. Ne donnez PAS d'exemples de réponses possibles
                4. Ne mentionnez pas que vous êtes un assistant
                5. Ne faites pas référence aux "slots" ou "informations manquantes"
                6. Utilisez un langage conversationnel
                7. Adaptez la question au contexte de la conversation
                8. NE LISTEZ PAS les options possibles
                9. Posez une question OUVERTE
                10. NE FAITES PAS de suppositions sur les informations manquantes
                11. Concentrez-vous sur les informations nécessaires pour faire une recommandation
                
                Répondez uniquement avec la question, sans explications supplémentaires."""},
                {"role": "user", "content": f"""
                Message de l'utilisateur: {message}
                Émotion détectée: {emotion}
                
                Ce que nous savons déjà:
                {self._format_known_slots(filled_slots)}
                
                Information(s) à obtenir: {', '.join(missing_slots)}
                
                Générez une question naturelle pour obtenir ces informations."""}
            ]
            
            response = self._get_llm_response(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Erreur lors de la génération de la question: {e}")
            return "Pouvez-vous me donner plus d'informations ?"
            
    def _format_known_slots(self, slots: Dict[str, Any]) -> str:
        """
        Formate les slots connus pour l'affichage.
        
        Args:
            slots (Dict[str, Any]): Les slots à formater
            
        Returns:
            str: Les slots formatés
        """
        formatted = []
        for key, value in slots.items():
            if value:
                formatted.append(f"- {key}: {value}")
        return "\n".join(formatted) if formatted else "Aucune information connue"
        
    def _get_slot_description(self, slot: str) -> str:
        """
        Retourne la description d'un slot.
        
        Args:
            slot (str): Le nom du slot
            
        Returns:
            str: La description du slot
        """
        slot_descriptions = {
            "location": "la ville ou le lieu",
            "food_type": "le type de cuisine",
            "budget": "le budget",
            "time": "l'horaire"
        }
        return slot_descriptions.get(slot, slot)
        
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
        
    def run(self, message: str, emotions: List[str], context: Optional[Dict] = None) -> str:
        """
        Génère une réponse en fonction du message et des émotions.
        
        Args:
            message (str): Le message de l'utilisateur
            emotions (List[str]): Liste des émotions détectées
            context (Optional[Dict]): Contexte supplémentaire
            
        Returns:
            str: La réponse générée
        """
        try:
            prompt = self._build_prompt(message, emotions, context)
            response = self._get_llm_response(prompt)
            final_response = self._parse_response(response)
                
            return final_response
            
        except Exception as e:
            print(f"Erreur lors de la génération de réponse: {e}")
            return "Je suis désolé, je n'ai pas pu générer de réponse appropriée."
        
    def _build_prompt(self, message: str, emotions: List[str], context: Optional[Dict] = None) -> List[Dict[str, str]]:
        """
        Construit le prompt pour la génération de réponse.
        
        Args:
            message (str): Le message de l'utilisateur
            emotions (List[str]): Liste des émotions détectées
            context (Optional[Dict]): Contexte supplémentaire
            
        Returns:
            List[Dict[str, str]]: Le prompt formaté
        """
        system_message = """Vous êtes un assistant touristique empathique et attentionné. Votre rôle est de répondre aux questions des utilisateurs en tenant compte de leurs émotions.

        Règles importantes :
        1. Adaptez votre ton et votre style en fonction des émotions détectées
        2. Soyez concis et direct dans vos réponses
        3. Utilisez un langage simple et accessible
        4. Évitez les réponses trop techniques ou complexes
        5. Restez toujours professionnel et respectueux
        6. Si vous n'êtes pas sûr, demandez des précisions
        
        Émotions détectées : {emotions}
        
        Contexte : {context}
        
        Répondez de manière naturelle et conversationnelle.
        """
        
        # Formatage du contexte
        context_str = json.dumps(context, ensure_ascii=False) if context else "Aucun contexte supplémentaire"
        
        return [
            {"role": "system", "content": system_message.format(emotions=", ".join(emotions), context=context_str)},
            {"role": "user", "content": message}
        ]
        
    def _parse_response(self, response: str) -> str:
        """
        Nettoie et formatte la réponse du LLM.
        
        Args:
            response (str): La réponse brute du LLM
            
        Returns:
            str: La réponse nettoyée
        """
        # Supprimer les marqueurs de formatage
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        response = re.sub(r'`.*?`', '', response)
        
        # Nettoyer les espaces et sauts de ligne
        response = ' '.join(response.split())
        
        return response.strip()
        