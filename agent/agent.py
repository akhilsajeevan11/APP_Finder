import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global store for session memories
session_memories = {}

def create_agent():
    # LLM Initialization (Gemini)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = None
    if not google_api_key or google_api_key == 'YOUR_GOOGLE_API_KEY_HERE':
        print("WARNING: GOOGLE_API_KEY not found or is a placeholder. Gemini agent will not function.")
    else:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=google_api_key)
            print("ChatGoogleGenerativeAI (Gemini) model initialized.")
        except Exception as e:
            print(f"Error initializing ChatGoogleGenerativeAI: {e}")
            # llm remains None

    if not llm:
        print("LLM not initialized. Agent creation will fail.")
        return None, None # Return None for agent and tools

    # Initialize tools
    tools = [DuckDuckGoSearchRun()]

    # Create the ConversationalChatAgent
    agent = None
    try:
        # This uses a default prompt template optimized for conversational chat with tools
        agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
        print("ConversationalChatAgent initialized.")
    except Exception as e:
        print(f"Error creating ConversationalChatAgent: {e}")
        return None, None # Return None for agent and tools

    return agent, tools

def get_agent_response(query: str, agent, tools, session_id: str):
    """
    Gets a response from the Langchain agent, managing session memory and AgentExecutor.
    """
    if agent is None or tools is None:
        return "Sorry, the core agent components are not properly initialized."

    # Memory Management
    if session_id not in session_memories:
        print(f"Creating new memory for session_id: {session_id}")
        # return_messages=True is generally preferred for chat models and ConversationalChatAgent
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = session_memories[session_id]
    # Log current history before new input for debugging (optional)
    # print(f"Using memory for session_id: {session_id}. Current history length: {len(memory.chat_memory.messages)}")


    # AgentExecutor Creation
    agent_executor = None
    try:
        # verbose=True is good for debugging
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            # max_iterations=5, # Optional: prevent runaway agents
            handle_parsing_errors=True # Useful for robustness against LLM formatting issues
        )
        # print(f"AgentExecutor created successfully for session {session_id}")
    except Exception as e:
        print(f"Error creating AgentExecutor for session {session_id}: {e}")
        return f"Sorry, I encountered an error setting up the agent for your session: {e}"

    # Invoke Agent
    try:
        # The ConversationalChatAgent expects 'input' and 'chat_history'
        # 'chat_history' is automatically populated by the memory object.
        response = agent_executor.invoke({"input": query})
        # The output key for ConversationalChatAgent is usually 'output'
        # Log memory after invoke for debugging (optional)
        # print(f"Memory for session {session_id} after invoke. History length: {len(memory.chat_memory.messages)}")
        return response.get("output", "Sorry, I could not process that.")
    except Exception as e:
        print(f"Agent Error during invoke for session {session_id}: {e}")
        # Attempt to provide a more specific error if it's an OutputParsingError
        # This specific string check might vary based on Langchain versions or error types
        if "Could not parse LLM output:" in str(e) or isinstance(e, Exception): # Broadened for typical parsing errors
            return "Sorry, the agent's response was not in the expected format. Please try rephrasing your question or try again later."
        return f"Sorry, I encountered an error while processing your request with the agent: {e}"


if __name__ == '__main__':
    # Example usage (for testing agent.py directly)
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == 'YOUR_GOOGLE_API_KEY_HERE':
        print("Google API key not found or is placeholder. Please set it in .env or as an environment variable.")
    else:
        print("Attempting to create Gemini ConversationalChatAgent and tools for testing...")
        test_agent, test_tools = create_agent()
        if test_agent and test_tools:
            print("Agent and tools created successfully for testing.")

            # Test get_agent_response
            test_session_id = "test_session_main"
            print(f"\n--- Test Query 1 for session {test_session_id} ---")
            response1 = get_agent_response("Hello, my name is Bob.", test_agent, test_tools, test_session_id)
            print(f"Response 1: {response1}")

            print(f"\n--- Test Query 2 for session {test_session_id} (testing memory) ---")
            response2 = get_agent_response("What is my name?", test_agent, test_tools, test_session_id)
            print(f"Response 2: {response2}")

            test_session_id_2 = "test_session_main_2"
            print(f"\n--- Test Query 1 for session {test_session_id_2} ---")
            response3 = get_agent_response("Hello, my name is Alice.", test_agent, test_tools, test_session_id_2)
            print(f"Response 3: {response3}")

            print(f"\n--- Test Query 2 for session {test_session_id_2} (testing memory) ---")
            response4 = get_agent_response("What is my name?", test_agent, test_tools, test_session_id_2)
            print(f"Response 4: {response4}")

            print(f"\n--- Test Query 3 for session {test_session_id} (testing memory persistence) ---")
            response5 = get_agent_response("What did I say my name was earlier?", test_agent, test_tools, test_session_id)
            print(f"Response 5: {response5}")


        else:
            print("Agent and/or tools creation failed in __main__.")
