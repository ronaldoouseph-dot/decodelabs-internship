# tests/test_chatbot.py
import unittest
import sys
import os

# Adjust sys.path to allow importing chatbot from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot import sanitize_input, get_response, KNOWLEDGE_BASE, FALLBACK_RESPONSE


class TestChatbot(unittest.TestCase):
    
    def test_sanitize_input_lowercase(self):
        """Test that uppercase input is correctly converted to lowercase."""
        self.assertEqual(sanitize_input("HELLO"), "hello")
        self.assertEqual(sanitize_input("What is AI"), "what is ai")
        self.assertEqual(sanitize_input("PyThOn"), "python")

    def test_sanitize_input_strip(self):
        """Test that leading and trailing whitespaces are stripped."""
        self.assertEqual(sanitize_input("  hello  "), "hello")
        self.assertEqual(sanitize_input("\twhat is ai\n"), "what is ai")

    def test_sanitize_input_combined(self):
        """Test that both converting to lowercase and stripping works together."""
        self.assertEqual(sanitize_input("   HELLO   "), "hello")
        self.assertEqual(sanitize_input("  What Is AI \n"), "what is ai")

    def test_greeting_response(self):
        """Test that greeting inputs return the correct greeting response."""
        expected_greeting = "Hi there! How can I help you?"
        self.assertEqual(get_response("hello"), expected_greeting)
        self.assertEqual(get_response("hi"), expected_greeting)
        self.assertEqual(get_response("hey"), expected_greeting)

    def test_at_least_five_intents(self):
        """Test that at least 5 different intents work and return correct responses."""
        # 1. Greeting intent
        self.assertIn("hello", KNOWLEDGE_BASE)
        # 2. Introduction/About intent
        self.assertIn("who are you", KNOWLEDGE_BASE)
        # 3. Artificial Intelligence intent
        self.assertIn("what is ai", KNOWLEDGE_BASE)
        # 4. Help intent
        self.assertIn("help", KNOWLEDGE_BASE)
        # 5. Programming intent
        self.assertIn("programming", KNOWLEDGE_BASE)
        # 6. Thanks intent
        self.assertIn("thank you", KNOWLEDGE_BASE)
        
        # Verify that their corresponding responses are correct
        self.assertEqual(get_response("who are you"), KNOWLEDGE_BASE["who are you"])
        self.assertEqual(get_response("what is ai"), KNOWLEDGE_BASE["what is ai"])
        self.assertEqual(get_response("help"), KNOWLEDGE_BASE["help"])
        self.assertEqual(get_response("programming"), KNOWLEDGE_BASE["programming"])
        self.assertEqual(get_response("thank you"), KNOWLEDGE_BASE["thank you"])

    def test_fallback_response(self):
        """Test that unknown input produces the fallback response."""
        self.assertEqual(get_response("random unknown query"), FALLBACK_RESPONSE)
        self.assertEqual(get_response("xyz"), FALLBACK_RESPONSE)
        self.assertEqual(get_response("artificial intelligence is cool"), FALLBACK_RESPONSE)

    def test_exit_commands(self):
        """Test that exit, quit, and bye are mapped to the correct exit responses."""
        # These keys should exist in the knowledge base and return the goodbye message
        expected_goodbye = "Goodbye! Have a great day! 👋"
        self.assertEqual(get_response("exit"), expected_goodbye)
        self.assertEqual(get_response("quit"), expected_goodbye)
        self.assertEqual(get_response("bye"), expected_goodbye)


if __name__ == "__main__":
    unittest.main()
