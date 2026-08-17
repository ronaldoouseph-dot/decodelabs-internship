# chatbot.py

# 1. Knowledge Base: A dictionary mapping predefined intents to responses.
# This represents a deterministic rule-based AI knowledge base using a hash-map lookup.
KNOWLEDGE_BASE = {
    # Greeting intents
    "hello": "Hi there! How can I help you?",
    "hi": "Hi there! How can I help you?",
    "hey": "Hi there! How can I help you?",
    
    # Introduction/About the bot intents
    "who are you": "I am a simple Rule-Based AI Chatbot. 🤖 I match your inputs to predefined rules without using complex ML models.",
    "what are you": "I am a deterministic rule-based chatbot designed to demonstrate logic and control flow in Python.",
    
    # Artificial Intelligence intents
    "what is ai": "Artificial Intelligence is the field of creating systems that can perform tasks that normally require human intelligence.",
    "how does ai work": "AI works by processing data through algorithms. Rule-based AI uses deterministic logical rules, while machine learning finds patterns in datasets.",
    
    # Help intent
    "help": "I can answer questions about: 'hello', 'what is ai', 'programming', 'who are you', or 'thanks'. You can exit by typing 'exit', 'quit', or 'bye'.",
    
    # Programming intents
    "programming": "Programming is the process of writing instructions in a programming language (like Python) to make computers perform specific tasks.",
    "python": "Python is a powerful, high-level, and easy-to-learn programming language widely used in AI, web development, and automation.",
    
    # Thanks intents
    "thank you": "You're very welcome! I'm happy to help. 😊",
    "thanks": "Glad I could help! 👍",
    
    # Goodbye intents
    "bye": "Goodbye! Have a great day! 👋",
    "exit": "Goodbye! Have a great day! 👋",
    "quit": "Goodbye! Have a great day! 👋"
}

# The fallback response when the user input does not match any intent.
FALLBACK_RESPONSE = "I don't understand that yet. Try asking about AI, programming, or help."


def sanitize_input(user_input: str) -> str:
    """
    Normalizes the user input:
    - Converts input to lowercase to make intent matching case-insensitive.
    - Strips leading and trailing whitespaces.
    """
    if not isinstance(user_input, str):
        return ""
    return user_input.strip().lower()


def get_response(user_input: str) -> str:
    """
    Looks up the sanitized input in the knowledge base.
    Returns the matching response or a fallback response if not found.
    """
    # Use dictionary lookup (hash-map) to retrieve the response or return the fallback
    return KNOWLEDGE_BASE.get(user_input, FALLBACK_RESPONSE)


def run_chatbot():
    """
    Runs the main continuous input loop of the chatbot.
    """
    # Display friendly Welcome Message and instructions
    print("🤖 Welcome to the Rule-Based AI Chatbot!")
    print("Type 'help' to see what I can ask.\n")
    
    while True:
        try:
            # 1. Ask the user for input
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            # Clean exit on terminal interrupt (Ctrl+C / Ctrl+D)
            print(f"\nBot: {KNOWLEDGE_BASE['exit']}")
            break
            
        # 2. Sanitize input
        sanitized = sanitize_input(user_input)
        
        # Ignore empty inputs and prompt again
        if not sanitized:
            continue
            
        # 3. Check whether the user wants to exit
        if sanitized in ["exit", "quit", "bye"]:
            response = get_response(sanitized)
            print(f"Bot: {response}")
            break
            
        # 4 & 5. Find matching response OR fallback and display response
        response = get_response(sanitized)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    run_chatbot()
