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
        
    def generate_response(self, slots: Dict[str, Any], intent: str, emotion: str = None) -> str:
        """
        Génère une réponse finale basée sur les informations disponibles et l'émotion de l'utilisateur.

        Args:
            slots (Dict[str, Any]): Les informations disponibles (slots)
            intent (str): L'intention détectée
            emotion (str, optional): L'émotion détectée dans le message de l'utilisateur
            
        Returns:
            str: Une réponse contextuelle et utile
        """
        try:
            # Construire le prompt pour le LLM
            prompt = [
                {"role": "system", "content": """Vous êtes un assistant touristique expert qui génère des réponses finales.
                Votre tâche est de fournir une réponse complète et utile basée sur toutes les informations disponibles,
                en adaptant votre ton à l'émotion de l'utilisateur.

                Instructions:
                1. Utilisez un ton adapté à l'émotion de l'utilisateur:
                   - Si l'utilisateur est heureux: soyez enthousiaste et positif
                   - Si l'utilisateur est frustré: soyez empathique et rassurant
                   - Si l'utilisateur est neutre: soyez professionnel et amical
                   - Si l'utilisateur est triste: soyez compatissant et encourageant
                2. Soyez concis et direct
                3. Ne répétez pas les informations déjà connues
                4. Proposez des suggestions pertinentes
                5. Terminez par une question ouverte pour continuer la conversation
                6. Ne mentionnez pas que vous êtes un assistant
                7. Ne faites pas de listes ou de sections explicites
                8. Utilisez un langage naturel et conversationnel

                Format de réponse:
                Une réponse fluide et naturelle qui:
                - Accueille l'utilisateur de manière personnalisée
                - Fournit des informations pertinentes
                - Propose des suggestions adaptées
                - Termine par une question ouverte"""},
                {"role": "user", "content": f"""
                Intention: {intent}
                Émotion détectée: {emotion if emotion else "neutre"}
                Informations disponibles:
                {self._format_slots(slots)}

                Générez une réponse naturelle et conversationnelle adaptée à l'émotion de l'utilisateur."""}
            ]

            # Obtenir la réponse du LLM
            response = self._get_llm_response(prompt)
            return response.strip()
            
        except Exception as e:
            return f"Désolé, je n'ai pas pu générer une réponse appropriée. Erreur: {str(e)}"

    def generate_question(self, missing_slot: str, filled_slots: Dict[str, Any], emotion: str = None) -> str:
        """
        Génère une question naturelle pour obtenir un slot manquant.

        Args:
            missing_slot (str): Le slot manquant à demander
            filled_slots (Dict[str, Any]): Les slots déjà remplis
            emotion (str, optional): L'émotion détectée dans le message de l'utilisateur
            
        Returns:
            str: Une question naturelle pour obtenir le slot manquant
        """
        try:
            # Construire le prompt pour le LLM
            prompt = [
                {"role": "system", "content": """Vous êtes un assistant touristique expert en communication naturelle.
                Votre tâche est de formuler une question pour obtenir une information manquante,
                en adaptant votre ton à l'émotion de l'utilisateur.

                Instructions:
                1. Utilisez un ton adapté à l'émotion de l'utilisateur:
                   - Si l'utilisateur est heureux: soyez enthousiaste et positif
                   - Si l'utilisateur est frustré: soyez empathique et rassurant
                   - Si l'utilisateur est neutre: soyez professionnel et amical
                   - Si l'utilisateur est triste: soyez compatissant et encourageant
                2. Formulez une seule question claire et naturelle
                3. Ne mentionnez pas que vous êtes un assistant
                4. Ne faites pas référence aux "slots" ou "informations"
                5. Adaptez la question au contexte et aux informations déjà connues
                6. Utilisez un langage conversationnel et naturel

                Exemples de bonnes questions:
                - "Dans quelle ville souhaitez-vous dîner ?"
                - "Quel type de cuisine préférez-vous ?"
                - "Quel est votre budget pour cette activité ?"
                - "Quand souhaitez-vous réserver ?"

                Répondez uniquement avec la question, sans explications supplémentaires."""},
                {"role": "user", "content": f"""
                Émotion détectée: {emotion if emotion else "neutre"}
                Information à demander: {self._get_slot_description(missing_slot)}
                Informations déjà connues:
                {self._format_slots(filled_slots)}

                Générez une question naturelle pour obtenir cette information."""}
            ]

            # Obtenir la réponse du LLM
            response = self._get_llm_response(prompt)
            return response.strip()
            
        except Exception as e:
            print(f"Erreur lors de la génération de la question: {e}")
            return f"Pouvez-vous me préciser {self._get_slot_description(missing_slot)} ?"
        
    def _format_slots(self, slots: Dict[str, Any]) -> str:
        """
        Formate les slots pour l'affichage.
        
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
        return "\n".join(formatted) if formatted else "Aucune information disponible"

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
            "food_type": "le type de cuisine (traditionnelle, moderne, végétarienne, asiatique, italienne, française)",
            "budget": "le niveau de prix (budget, moyen, luxe)",
            "time": "le moment (ce soir, demain, ce week-end, déjeuner, dîner)",
            "activity_type": "le type d'activité (culturelle, sportive, gastronomique, etc.)",
            "date": "la date ou la période"
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
            
            # Debug pour indiquer si la réponse est finale ou une question
            if "?" in final_response:
                print("[DEBUG] La réponse est une question")
            else:
                print("[DEBUG] La réponse est finale")
                
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
        