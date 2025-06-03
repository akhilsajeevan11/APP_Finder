from flask import Flask, render_template, request, jsonify
import sys
import os

# Add agent directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

# Assuming agent.py contains create_agent and get_agent_response
from agent import create_agent, get_agent_response

app = Flask(__name__)

# Initialize the agent and tools once when the app starts
print("Initializing Langchain agent components at startup...")
agent_instance, tools_list = None, None # Initialize to None
try:
    agent_instance, tools_list = create_agent()
    if agent_instance and tools_list:
        print("ConversationalChatAgent and tools initialized successfully at startup.")
    else:
        print("CRITICAL: Agent instance or tools list failed to initialize during startup. The application may not function correctly.")
        # Depending on desired behavior, could raise an error or exit
except Exception as e:
    print(f"CRITICAL: Failed to initialize agent components at startup: {e}")
    # agent_instance and tools_list will remain None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    # Placeholder for session ID management. For now, use a fixed session ID.
    # Proper session management (e.g., using Flask sessions, user IDs) should be implemented for a real application.
    session_id = "default_fixed_session"
    # Example for testing multiple sessions (manual switch by query param for simplicity):
    if request.args.get("session") == "new":
        # This is a very basic way to switch; real apps need robust session handling
        if "counter" not in app.config: app.config["counter"] = 0
        app.config["counter"] += 1
        session_id = f"fixed_session_{app.config['counter']}"

    print(f"Processing chat for session_id: {session_id}")


    if not user_message:
        return jsonify({'response': "Please send a message."}), 400

    # Check if agent components are available
    if agent_instance is None or tools_list is None:
        print("Error: Agent components (agent_instance or tools_list) are not available.")
        return jsonify({'response': "Sorry, the main agent components are not available. Please check server logs."}), 500

    print(f"Received message for session {session_id}: {user_message}")
    try:
        # Call get_agent_response with the agent, tools, and session_id
        bot_response = get_agent_response(user_message, agent_instance, tools_list, session_id)
    except Exception as e:
        print(f"Error during agent processing for session {session_id}: {e}")
        bot_response = "Sorry, I encountered an error while processing your request."

    print(f"Sending response for session {session_id}: {bot_response}")
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == 'YOUR_GOOGLE_API_KEY_HERE':
        print("WARNING: GOOGLE_API_KEY not set or is placeholder. Gemini agent functionality will be limited or fail.")
        print("Please create or update .env file in the root directory with your GOOGLE_API_KEY.")
    else:
        print("Starting Flask app with Gemini agent (API key found)...")

    # Using port 5001 as before
    app.run(debug=True, host='0.0.0.0', port=5001)
