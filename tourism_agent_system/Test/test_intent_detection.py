import unittest
import sys
import os

# Ajouter le chemin du projet au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Agent.intent_detection_agent import IntentDetectionAgent

class TestIntentDetectionAgent(unittest.TestCase):
    """Tests for the intent detection agent"""

    def setUp(self):
        """Initialization before each test"""
        self.agent = IntentDetectionAgent()

    def _print_detection_result(self, message: str, result: dict):
        """Display the intent detection results"""
        print("\n" + "="*50)
        print(f"Message: {message}")
        print(f"Detected intent: {result['intent']}")
        print("\nDetected slots:")
        for slot, value in result['slots'].items():
            if value:  # Only display non-empty slots
                print(f"- {slot}: {value}")
        if result.get('search_query'):
            print(f"\nSearch query: {result['search_query']}")
        print("="*50 + "\n")

    def test_restaurant_search(self):
        """Test restaurant search intent detection"""
        test_cases = [
            {
                "message": "I'm looking for an Italian restaurant in Paris for tonight, medium budget",
                "expected_intent": "restaurant_search",
                "expected_slots": {
                    "location": "Paris",
                    "food_type": "Italian",
                    "budget": "medium",
                    "time": "tonight"
                }
            },
            {
                "message": "I want to eat Chinese food at noon",
                "expected_intent": "restaurant_search",
                "expected_slots": {
                    "food_type": "Chinese",
                    "time": "noon"
                }
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertEqual(result["intent"], test_case["expected_intent"])
                for slot, value in test_case["expected_slots"].items():
                    self.assertEqual(result["slots"].get(slot), value)

    def test_activity_search(self):
        """Test activity search intent detection"""
        test_cases = [
            {
                "message": "I want to do a cultural activity in Lyon this weekend",
                "expected_intent": "activity_search",
                "expected_slots": {
                    "location": "Lyon",
                    "activity_type": "cultural",
                    "date": "this weekend"
                }
            },
            {
                "message": "What sports activities are available?",
                "expected_intent": "activity_search",
                "expected_slots": {
                    "activity_type": "sports"
                }
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertEqual(result["intent"], test_case["expected_intent"])
                for slot, value in test_case["expected_slots"].items():
                    self.assertEqual(result["slots"].get(slot), value)

    def test_hotel_booking(self):
        """Test hotel booking intent detection"""
        test_cases = [
            {
                "message": "I want to book a hotel in Marseille from July 15 to July 20, comfortable budget",
                "expected_intent": "hotel_booking",
                "expected_slots": {
                    "location": "Marseille",
                    "check_in_date": "July 15",
                    "check_out_date": "July 20",
                    "budget": "comfortable"
                }
            },
            {
                "message": "I'm looking for a cheap hotel for this weekend",
                "expected_intent": "hotel_booking",
                "expected_slots": {
                    "budget": "cheap",
                    "date": "this weekend"
                }
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertEqual(result["intent"], test_case["expected_intent"])
                for slot, value in test_case["expected_slots"].items():
                    self.assertEqual(result["slots"].get(slot), value)

    def test_information_request(self):
        """Test information request intent detection"""
        test_cases = [
            {
                "message": "What are the tourist attractions in Bordeaux?",
                "expected_intent": "information_request"
            },
            {
                "message": "Where can I find a good restaurant?",
                "expected_intent": "information_request"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertEqual(result["intent"], test_case["expected_intent"])
                self.assertIsNotNone(result.get("search_query"))

    def test_greeting(self):
        """Test greeting intent detection"""
        test_cases = [
            {
                "message": "Hello, how are you?",
                "expected_intent": "greeting"
            },
            {
                "message": "Hi!",
                "expected_intent": "greeting"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertEqual(result["intent"], test_case["expected_intent"])

    def test_unknown_intent(self):
        """Test unknown intent detection"""
        test_cases = [
            {
                "message": "xyz123",
                "expected_intent": "unknown"
            },
            {
                "message": "!@#$%^&*()",
                "expected_intent": "unknown"
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                self.assertEqual(result["intent"], test_case["expected_intent"])

    def test_missing_info(self):
        """Test missing information detection"""
        test_cases = [
            {
                "message": "I'm looking for a restaurant",
                "expected_missing": ["location", "food_type", "budget", "time"]
            },
            {
                "message": "I want to do an activity",
                "expected_missing": ["location", "activity_type", "date"]
            }
        ]

        for test_case in test_cases:
            with self.subTest(message=test_case["message"]):
                result = self.agent.run(test_case["message"])
                self._print_detection_result(test_case["message"], result)
                missing_info = self.agent.check_missing_info(result["slots"])
                print(f"Missing information: {missing_info}")
                for expected in test_case["expected_missing"]:
                    self.assertIn(expected, missing_info)

if __name__ == '__main__':
    unittest.main(verbosity=2) 