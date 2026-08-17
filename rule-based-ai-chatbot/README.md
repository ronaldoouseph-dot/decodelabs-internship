# DecodeLabs AI Project 1: Rule-Based AI Chatbot

A simple, deterministic, rule-based AI chatbot built entirely using Python's standard library. 

> [!IMPORTANT]
> This is a beginner-friendly foundation AI project. It runs entirely on deterministic rules and explicit programming logic. It does **not** use any LLMs (such as OpenAI, Gemini, etc.), machine learning models, vector databases, or external AI services.

---

## Project Objective

The objective of this project is to build a basic chatbot that matches predefined user inputs to specific responses. The project demonstrates core programming and computer science fundamentals:
- Control flow and decision-making logic
- String sanitization and normalization
- Hash-map/dictionary lookup-based intent matching
- Continuous application execution loops
- Fallback/error handling
- Clean program termination

---

## Features

- **Emoji-enhanced Terminal Interface**: Clear, visually clean, and friendly user prompts.
- **Continuous Input Loop**: Repeatedly prompts the user for inputs without exiting.
- **Robust Input Sanitization**: Standardizes uppercase inputs and trims accidental leading/trailing spaces.
- **Hash-Map Based Lookup**: Uses a Python dictionary for fast intent matching instead of nested `if/elif` statements.
- **Predefined Intents**: Matches at least 5 standard intents including greetings, AI explanations, programming questions, help menus, and exit indicators.
- **Fallback Response**: Returns a helpful recommendation message when the input cannot be matched.
- **Clean Exit Strategy**: Gracefully ends the program when `exit`, `quit`, or `bye` is typed, or when interrupted (Ctrl+C).

---

## Technologies Used

- **Python 3** (Standard Library only)
- **unittest** (Python's built-in testing framework)

---

## How It Works

The chatbot executes a deterministic data pipeline:

```
[User Input] 
     ↓
[Sanitization]  --> Convert to Lowercase & Strip Spaces
     ↓
[Exit Check]    --> Check if in ['exit', 'quit', 'bye'] -> End if true
     ↓
[Dict Lookup]   --> Query dictionary keys
     ↓
[Response]      --> Print Matched Response OR Fallback Response
     ↓
[Loop Repeat]
```

### 1. Input Sanitization
Every user message is normalized using Python's built-in string methods.
- Lowercasing: `user_input.lower()` ensures that `"HELLO"` matches `"hello"`.
- Stripping: `user_input.strip()` ensures that `"  hello  "` matches `"hello"`.

### 2. Dictionary Lookup (Intent Matching)
Instead of chaining `if/elif` statements, the bot maps intent keys to predefined string responses in a Python dictionary (`KNOWLEDGE_BASE`). Using `KNOWLEDGE_BASE.get(key, fallback)` ensures:
- Time complexity of $O(1)$ for lookup.
- Cleaner, more maintainable code.
- Automatic routing to `FALLBACK_RESPONSE` when the key is not found.

### 3. Continuous Loop
A `while True` loop keeps the application active, repeatedly polling user input using the `input()` function until an exit criteria is met.

### 4. Exit Strategy
If the sanitized input matches `"exit"`, `"quit"`, or `"bye"`, the chatbot displays a goodbye message and breaks out of the `while` loop, allowing Python to exit cleanly.

---

## How to Run the Project

No external packages or API keys are required. You only need Python 3 installed.

### 1. Run the Chatbot
Navigate to the `rule-based-ai-chatbot` folder in your terminal and run:
```powershell
python chatbot.py
```

### 2. Run the Tests
To run the automated test suite, execute:
```powershell
python -m unittest tests/test_chatbot.py
```

---

## Example Interaction

```text
🤖 Welcome to the Rule-Based AI Chatbot!
Type 'help' to see what I can ask.

You: hello
Bot: Hi there! How can I help you?

You: what is ai
Bot: Artificial Intelligence is the field of creating systems that can perform tasks that normally require human intelligence.

You: xyz
Bot: I don't understand that yet. Try asking about AI, programming, or help.

You: bye
Bot: Goodbye! Have a great day! 👋
```

---

## Future Improvements

While this project is intentionally simple, potential future enhancements include:
1. **Keyword/Substring Matching**: Checking if key phrases are present in the user's input instead of requiring exact matching.
2. **Regular Expressions**: Using Python's `re` module to match patterns in the user input.
3. **JSON Knowledge Base**: Storing rules in an external `.json` file to separate code from data.
4. **Levenshtein Distance**: Introducing basic string-distance algorithms to handle minor typos.
