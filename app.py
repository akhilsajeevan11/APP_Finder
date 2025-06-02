from flask import Flask, render_template, request, jsonify
import sys
import os

# Add agent directory to Python path
# This ensures that 'from agent import ...' works
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

from agent import create_agent, get_agent_response

app = Flask(__name__)

# Initialize the agent once when the app starts
print("Initializing Langchain agent with Gemini (placeholder)...")
try:
    agent_executor = create_agent() # This might return None if GOOGLE_API_KEY is missing/invalid
    if agent_executor:
        print("Langchain agent (Gemini) initialized successfully (or ready if API key is valid).")
    else:
        print("Langchain agent (Gemini) failed to initialize (e.g., missing GOOGLE_API_KEY). It will not be available.")
except Exception as e:
    # This catch block might be redundant if create_agent handles its exceptions and returns None
    print(f"Failed to initialize agent (Gemini): {e}")
    agent_executor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    if agent_executor is None:
        return jsonify({'response': "Sorry, the agent is not available (LLM initialization failed, check GOOGLE_API_KEY)."}), 500

    if not user_message:
        return jsonify({'response': "Please send a message."}), 400

    print(f"Received message: {user_message}")
    try:
        # get_agent_response now also checks if agent_executor is None, but good to have primary check above.
        bot_response = get_agent_response(user_message, agent_executor)
    except Exception as e:
        print(f"Error during agent processing: {e}")
        bot_response = "Sorry, I encountered an error while processing your request."

    print(f"Sending response: {bot_response}")
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key and google_api_key != 'YOUR_GOOGLE_API_KEY_HERE':
        print("Starting Flask app with Gemini agent (API key found)...")
    else:
        print("WARNING: GOOGLE_API_KEY not set or is placeholder. Gemini agent functionality will be limited or fail.")
        print("Please create or update .env file in the root directory with your GOOGLE_API_KEY.")
        print("Example .env: GOOGLE_API_KEY='YOUR_ACTUAL_KEY'")

    # Using port 5001 as before
    app.run(debug=True, host='0.0.0.0', port=5001)
