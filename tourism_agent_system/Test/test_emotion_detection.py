import unittest
import sys
import os

# Ajouter le chemin du projet au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Agent.emotion_detection_agent import EmotionDetectionAgent

class TestEmotionDetectionAgent(unittest.TestCase):
    """Tests pour l'agent de détection d'émotions"""

    def setUp(self):
        """Initialisation avant chaque test"""
        self.agent = EmotionDetectionAgent()

    def _print_detection_result(self, message: str, result: dict):
        """Affiche les résultats de la détection d'émotion"""
        print("\n" + "="*50)
        print(f"Message : {message}")
        print(f"Émotions détectées : {', '.join(result['emotions'])}")
        print(f"Émotion principale : {result['primary_emotion']}")
        print(f"Intensité : {result['intensity']}")
        print(f"Confiance : {result['confidence']}")
        print("="*50 + "\n")

    def test_joy(self):
        """Test de la détection d'émotion de joie"""
        test_cases = [
            {
                "message": "Je suis vraiment ravi de visiter Paris pour la première fois ! C'est un rêve qui se réalise.",
                "expected_emotions": ["joie", "excitation"],
                "expected_primary": "joie",
                "expected_intensity": "forte"
            },
            {
                "message": "J'ai réussi mon examen, je suis content !",
                "expected_emotions": ["joie", "satisfaction"],
                "expected_primary": "joie",
                "expected_intensity": "modérée"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertIn(test_case["expected_primary"], result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

    def test_sadness(self):
        """Test de la détection d'émotion de tristesse"""
        test_cases = [
            {
                "message": "Je suis vraiment déçu de ne pas pouvoir partir en vacances cette année.",
                "expected_emotions": ["tristesse", "déception"],
                "expected_primary": "tristesse",
                "expected_intensity": "forte"
            },
            {
                "message": "C'est dommage que le temps soit mauvais aujourd'hui.",
                "expected_emotions": ["tristesse"],
                "expected_primary": "tristesse",
                "expected_intensity": "faible"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertIn(test_case["expected_primary"], result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

    def test_anger(self):
        """Test de la détection d'émotion de colère"""
        test_cases = [
            {
                "message": "Je suis furieux contre ce service client qui ne répond jamais !",
                "expected_emotions": ["colère", "frustration"],
                "expected_primary": "colère",
                "expected_intensity": "forte"
            },
            {
                "message": "C'est agaçant d'attendre si longtemps.",
                "expected_emotions": ["colère"],
                "expected_primary": "colère",
                "expected_intensity": "faible"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertIn(test_case["expected_primary"], result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

    def test_fear(self):
        """Test de la détection d'émotion de peur"""
        test_cases = [
            {
                "message": "J'ai très peur de prendre l'avion, c'est une phobie.",
                "expected_emotions": ["peur", "anxiété"],
                "expected_primary": "peur",
                "expected_intensity": "forte"
            },
            {
                "message": "Je suis un peu inquiet pour mon rendez-vous demain.",
                "expected_emotions": ["peur"],
                "expected_primary": "peur",
                "expected_intensity": "faible"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertIn(test_case["expected_primary"], result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

    def test_surprise(self):
        """Test de la détection d'émotion de surprise"""
        test_cases = [
            {
                "message": "Oh ! Je ne m'attendais vraiment pas à ce cadeau !",
                "expected_emotions": ["surprise", "joie"],
                "expected_primary": "surprise",
                "expected_intensity": "forte"
            },
            {
                "message": "Tiens, c'est étrange de te voir ici.",
                "expected_emotions": ["surprise"],
                "expected_primary": "surprise",
                "expected_intensity": "faible"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertIn(test_case["expected_primary"], result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

    def test_disgust(self):
        """Test de la détection d'émotion de dégoût"""
        test_cases = [
            {
                "message": "Je trouve ce comportement vraiment méprisable.",
                "expected_emotions": ["dégoût", "mépris"],
                "expected_primary": "dégoût",
                "expected_intensity": "forte"
            },
            {
                "message": "Ce n'est pas très appétissant.",
                "expected_emotions": ["dégoût"],
                "expected_primary": "dégoût",
                "expected_intensity": "faible"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertIn(test_case["expected_primary"], result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

    def test_neutral(self):
        """Test de la détection d'émotion neutre"""
        test_cases = [
            {
                "message": "Le temps est nuageux aujourd'hui.",
                "expected_emotions": ["neutre"],
                "expected_primary": "neutre",
                "expected_intensity": "faible"
            },
            {
                "message": "Je vais au travail.",
                "expected_emotions": ["neutre"],
                "expected_primary": "neutre",
                "expected_intensity": "faible"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertIn(test_case["expected_primary"], result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

    def test_mixed_emotions(self):
        """Test de la détection d'émotions mixtes"""
        test_cases = [
            {
                "message": "Je suis à la fois excité et un peu inquiet pour mon voyage.",
                "expected_emotions": ["joie", "peur"],
                "expected_primary": "joie",
                "expected_intensity": "modérée"
            },
            {
                "message": "C'est à la fois triste et beau de voir partir les oiseaux migrateurs.",
                "expected_emotions": ["tristesse", "joie"],
                "expected_primary": "tristesse",
                "expected_intensity": "modérée"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                for emotion in test_case["expected_emotions"]:
                    self.assertIn(emotion, result["emotions"])
                self.assertEqual(result["primary_emotion"], test_case["expected_primary"])
                self.assertEqual(result["intensity"], test_case["expected_intensity"])
                self.assertGreater(result["confidence"], 0.5)

if __name__ == '__main__':
    unittest.main(verbosity=2) 