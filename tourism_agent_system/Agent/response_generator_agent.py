from Agent import Agent
import ollama
from typing import Dict, Any, List, Optional
import json
import re

class ResponseGeneratorAgent(Agent):
    """
    Agent qui génère des réponses contextuelles basées sur l'intent, les slots et l'émotion de l'utilisateur
    """
    
    def __init__(self, name: str = "response_generator"):
        super().__init__(name)
        self._llm = ollama.Client()
        self._model_config = self._config["model"]
        
    def generate_response(self, message: str, emotion: str, intent: str, slots: Dict[str, Any]) -> str:
        """
        Génère une réponse finale basée sur toutes les informations disponibles.
        
        Args:
            message (str): Le message de l'utilisateur
            emotion (str): L'émotion détectée
            intent (str): L'intention détectée
            slots (Dict[str, Any]): Les slots extraits
            
        Returns:
            str: La réponse finale
        """
        try:
            prompt = [
                {"role": "system", "content": """Vous êtes un assistant touristique qui aide les utilisateurs à trouver des restaurants.
                Votre tâche est de fournir une réponse finale basée sur toutes les informations disponibles.
                
                Instructions:
                1. Adaptez votre ton à l'émotion de l'utilisateur
                2. Soyez concis et direct
                3. Ne répétez pas les informations déjà connues
                4. Proposez des suggestions pertinentes
                5. Terminez par une question ouverte pour continuer la conversation
                6. Ne mentionnez pas que vous êtes un assistant
                7. Utilisez un langage naturel et conversationnel"""},
                {"role": "user", "content": f"""
                Message: {message}
                Émotion: {emotion}
                Intention: {intent}
                Informations disponibles:
                - Localisation: {slots.get('location')}
                - Type de cuisine: {slots.get('food_type')}
                - Budget: {slots.get('budget')}
                - Heure: {slots.get('time')}
                
                Génère une réponse finale qui prend en compte toutes ces informations."""}
            ]
            
            response = self._get_llm_response(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse: {e}")
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
            # Construire le contexte pour le LLM
            slot_descriptions = {
                "location": "la ville ou le lieu",
                "food_type": "le type de cuisine",
                "budget": "le budget",
                "time": "l'horaire"
            }
            
            missing_descriptions = [slot_descriptions[slot] for slot in missing_slots]
            
            prompt = [
                {"role": "system", "content": """Vous êtes un assistant touristique qui aide les utilisateurs à trouver des restaurants.
                Votre tâche est de poser une question naturelle pour obtenir les informations manquantes nécessaires pour faire une recommandation.
                
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
                10. NE DEMANDEZ PAS le nom du restaurant (vous devez le trouver)
                11. Concentrez-vous sur les informations nécessaires pour faire une recommandation
                
                Exemples de bonnes questions:
                - "Quel est votre budget pour ce repas ?"
                - "À quelle heure souhaitez-vous dîner ?"
                - "Dans quel quartier préférez-vous manger ?"
                
                Répondez uniquement avec la question, sans explications supplémentaires."""},
                {"role": "user", "content": f"""
                Message de l'utilisateur: {message}
                Émotion détectée: {emotion}
                
                Ce que nous savons déjà:
                {self._format_known_slots(filled_slots)}
                
                Information(s) à obtenir: {', '.join(missing_descriptions)}
                
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
                description = self._get_slot_description(key)
                formatted.append(f"- {description}: {value}")
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
        Obtient une réponse du LLM.
        
        Args:
            prompt (List[Dict[str, str]]): Le prompt à envoyer au LLM
            
        Returns:
            str: La réponse du LLM
        """
        try:
            response = self._llm.chat(
                model=self._model_config["name"],
                messages=prompt,
                options={
                    "temperature": self._model_config["temperature"],
                    "max_tokens": self._model_config["max_tokens"]
                }
            )
            
            # Vérifier si la réponse contient le contenu attendu
            if "message" in response and "content" in response["message"]:
                return response["message"]["content"]
            else:
                return "Désolé, je n'ai pas pu générer une réponse appropriée. Format de réponse inattendu."
        except Exception as e:
            return "Désolé, je n'ai pas pu générer une réponse appropriée. Veuillez réessayer."
        
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
        