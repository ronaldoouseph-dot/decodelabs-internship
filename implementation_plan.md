# Implementation Plan - Rule-Based AI Chatbot

This implementation plan outlines the creation of the **Rule-Based AI Chatbot** for DecodeLabs Artificial Intelligence Project 1. The chatbot operates deterministically using pure rule-based logic without machine learning models or APIs.

## Proposed Changes

We will create a subdirectory named `rule-based-ai-chatbot` inside the workspace root (`c:\Users\acer\OneDrive\Desktop\AI chatbot`) and populate it with the requested files.

### 1. Main Chatbot Script
#### [NEW] [chatbot.py](file:///c:/Users/acer/OneDrive/Desktop/AI%20chatbot/rule-based-ai-chatbot/chatbot.py)
This is the main entry point of our application. It will contain:
- A predefined knowledge base dictionary mapping normalized strings to responses.
- `sanitize_input(user_input)`: Converts input to lowercase and strips surrounding whitespace.
- `get_response(user_input)`: Resolves the response using dictionary lookup with a fallback message.
- `run_chatbot()`: Implements the continuous `while True` loop, input prompts, exit handling, and terminal output formatting.

### 2. Unit Tests
#### [NEW] [test_chatbot.py](file:///c:/Users/acer/OneDrive/Desktop/AI%20chatbot/rule-based-ai-chatbot/tests/test_chatbot.py)
This script will use Python's standard `unittest` module to verify the following behavior:
- Sanitization handles uppercase input and surrounding whitespace correctly.
- Greeting input matches the correct response.
- Unknown input triggers the fallback response.
- Standard intents (at least 5) resolve to correct answers.
- Exit commands (`exit`, `quit`, `bye`) are correctly identified.

### 3. Documentation
#### [NEW] [README.md](file:///c:/Users/acer/OneDrive/Desktop/AI%20chatbot/rule-based-ai-chatbot/README.md)
A comprehensive markdown guide explaining:
- Project title, objectives, and features.
- Technical explanations of:
  - Input sanitization
  - Dictionary intent matching
  - Continuous input loop
  - Exit mechanism
- Instructions on how to run the chatbot and automated tests.
- Example user interactions.
- Future improvements.

---

## Verification Plan

### Automated Tests
We will run tests using Python's standard test runner from the root of the project:
```powershell
python -m unittest tests/test_chatbot.py
```

### Manual Verification
We will run `python chatbot.py` and manually test:
1. Normal greetings ("hello", "hi").
2. Intent queries with different casing and leading/trailing spacing (e.g., `"  WHAT IS AI   "`).
3. Fallback responses (e.g., typing `"xyz"`).
4. Exit commands (`bye`, `quit`, `exit`) to ensure clean termination.
