# test_response_generator.py
import unittest
import sys
import os

# Ajouter le chemin du dossier parent au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Agent.response_generator_agent import ResponseGeneratorAgent
from typing import Dict, Any, List

class TestResponseGeneratorAgent(unittest.TestCase):
    """
    Tests pour le ResponseGeneratorAgent
    """

    def setUp(self):
        """
        Initialisation des tests
        """
        self.agent = ResponseGeneratorAgent()

    def test_generate_question_for_missing_slots(self):
        """
        Test de la génération de questions pour les slots manquants
        """
        test_cases = [
            {
                "name": "Test avec un slot manquant",
                "missing_slots": ["budget"],
                "filled_slots": {
                    "destination": "Paris",
                    "dates": "15-20 juillet"
                },
                "emotion": "neutre",
                "expected_keywords": ["budget", "prix", "coût", "dépenser", "argent"]
            },
            {
                "name": "Test avec plusieurs slots manquants",
                "missing_slots": ["budget", "type_activite"],
                "filled_slots": {
                    "destination": "Paris"
                },
                "emotion": "joie",
                "expected_keywords": ["budget", "prix", "coût", "dépenser", "argent", "activité", "visite", "faire"]
            },
            {
                "name": "Test avec émotion négative",
                "missing_slots": ["budget"],
                "filled_slots": {
                    "destination": "Paris",
                    "dates": "15-20 juillet"
                },
                "emotion": "frustration",
                "expected_keywords": ["budget", "prix", "coût", "dépenser", "argent"]
            }
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                print(f"\nTest: {test_case['name']}")
                print(f"Slots manquants: {test_case['missing_slots']}")
                print(f"Slots remplis: {test_case['filled_slots']}")
                print(f"Émotion: {test_case['emotion']}")
                question = self.agent.generate_question(
                    test_case["missing_slots"],
                    test_case["filled_slots"],
                    test_case["emotion"]
                )
                print(f"Question générée: {question}")
                print(f"Mots-clés attendus: {test_case['expected_keywords']}")
                self.assertIsInstance(question, str)
                self.assertTrue(len(question) > 0)
                self.assertTrue(
                    any(word in question.lower() for word in test_case["expected_keywords"])
                )

    def test_generate_final_response(self):
        """
        Test de la génération de réponses finales avec tous les slots remplis
        """
        test_cases = [
            {
                "name": "Test avec une demande d'information",
                "intent": "demande_information",
                "slots": {
                    "destination": "Paris",
                    "dates": "15-20 juillet",
                    "budget": "150€ par nuit",
                    "type_activite": "culturel"
                },
                "emotion": "joie",
                "search_results": [
                    {
                        "title": "Guide touristique Paris",
                        "description": "Les meilleures attractions culturelles de Paris",
                        "url": "https://example.com/paris"
                    }
                ],
                "expected_keywords": ["culturel", "musée", "visite", "attraction", "monument"]
            },
            {
                "name": "Test avec une demande de réservation",
                "intent": "reservation",
                "slots": {
                    "destination": "Paris",
                    "dates": "15-20 juillet",
                    "budget": "150€ par nuit",
                    "type_hebergement": "hôtel",
                    "nombre_personnes": "2"
                },
                "emotion": "neutre",
                "search_results": [
                    {
                        "title": "Hôtels à Paris",
                        "description": "Liste des meilleurs hôtels à Paris",
                        "url": "https://example.com/hotels-paris"
                    }
                ],
                "expected_keywords": ["hôtel", "réservation", "chambre", "nuit", "séjour"]
            },
            {
                "name": "Test avec une demande de recommandation",
                "intent": "recommandation",
                "slots": {
                    "destination": "Paris",
                    "type_activite": "gastronomie",
                    "budget": "moyen"
                },
                "emotion": "excitation",
                "search_results": [
                    {
                        "title": "Restaurants à Paris",
                        "description": "Les meilleurs restaurants de Paris",
                        "url": "https://example.com/restaurants-paris"
                    }
                ],
                "expected_keywords": ["gastronomie", "restaurant", "cuisine", "manger", "dîner"]
            }
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                print(f"\nTest: {test_case['name']}")
                print(f"Intent: {test_case['intent']}")
                print(f"Slots: {test_case['slots']}")
                print(f"Émotion: {test_case['emotion']}")
                print(f"Résultats de recherche: {test_case['search_results']}")
                response = self.agent.run(
                    test_case["intent"],
                    test_case["slots"],
                    test_case["emotion"],
                    test_case["search_results"]
                )
                print(f"Réponse générée: {response}")
                print(f"Mots-clés attendus: {test_case['expected_keywords']}")
                self.assertIsInstance(response, str)
                self.assertTrue(len(response) > 0)
                self.assertIn(test_case["slots"]["destination"], response)
                self.assertTrue(
                    any(word in response.lower() for word in test_case["expected_keywords"])
                )

    def test_generate_response_with_negative_emotion(self):
        """
        Test de la génération de réponses avec des émotions négatives
        """
        test_cases = [
            {
                "name": "Test avec frustration",
                "intent": "demande_information",
                "slots": {
                    "destination": "Paris",
                    "budget": "50€ par nuit"
                },
                "emotion": "frustration",
                "search_results": [
                    {
                        "title": "Budget voyage Paris",
                        "description": "Comment visiter Paris avec un petit budget",
                        "url": "https://example.com/budget-paris"
                    }
                ],
                "expected_keywords": ["comprends", "frustration", "difficile", "comprendre", "sais", "entends"]
            },
            {
                "name": "Test avec colère",
                "intent": "demande_information",
                "slots": {
                    "destination": "Paris",
                    "budget": "50€ par nuit"
                },
                "emotion": "colère",
                "search_results": [
                    {
                        "title": "Budget voyage Paris",
                        "description": "Comment visiter Paris avec un petit budget",
                        "url": "https://example.com/budget-paris"
                    }
                ],
                "expected_keywords": ["comprends", "colère", "difficile", "comprendre", "sais", "entends"]
            }
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                print(f"\nTest: {test_case['name']}")
                print(f"Intent: {test_case['intent']}")
                print(f"Slots: {test_case['slots']}")
                print(f"Émotion: {test_case['emotion']}")
                print(f"Résultats de recherche: {test_case['search_results']}")
                response = self.agent.run(
                    test_case["intent"],
                    test_case["slots"],
                    test_case["emotion"],
                    test_case["search_results"]
                )
                print(f"Réponse générée: {response}")
                print(f"Mots-clés attendus: {test_case['expected_keywords']}")
                self.assertIsInstance(response, str)
                self.assertTrue(len(response) > 0)
                self.assertIn(test_case["slots"]["destination"], response)
                self.assertIn("budget", response.lower())
                self.assertTrue(
                    any(word in response.lower() for word in test_case["expected_keywords"])
                )

    def test_generate_response_without_search_results(self):
        """
        Test de la génération de réponses sans résultats de recherche
        """
        test_cases = [
            {
                "name": "Test sans résultats de recherche",
                "intent": "demande_information",
                "slots": {
                    "destination": "Paris",
                    "dates": "15-20 juillet"
                },
                "emotion": "neutre",
                "expected_keywords": ["Paris", "juillet"]
            }
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                print(f"\nTest: {test_case['name']}")
                print(f"Intent: {test_case['intent']}")
                print(f"Slots: {test_case['slots']}")
                print(f"Émotion: {test_case['emotion']}")
                response = self.agent.run(
                    test_case["intent"],
                    test_case["slots"],
                    test_case["emotion"]
                )
                print(f"Réponse générée: {response}")
                print(f"Mots-clés attendus: {test_case['expected_keywords']}")
                self.assertIsInstance(response, str)
                self.assertTrue(len(response) > 0)
                self.assertTrue(
                    all(word in response for word in test_case["expected_keywords"])
                )

if __name__ == '__main__':
    unittest.main() 