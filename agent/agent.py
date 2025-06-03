import os
# from langchain_openai import ChatOpenAI # Commented out OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_agent():
    # Get the prompt template
    prompt = hub.pull("hwchase17/react")

    # Initialize the LLM
    # +++ Google Gemini LLM +++
    # Make sure GOOGLE_API_KEY is set in your environment or .env file
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == 'YOUR_GOOGLE_API_KEY_HERE':
        print("WARNING: GOOGLE_API_KEY not found or is a placeholder. Gemini agent will not function.")
        llm = None
    else:
        try:
            # Updated model name to "models/gemini-1.5-pro-latest"
            # Ensure this model is available for your API key and region.
            llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0, google_api_key=google_api_key)
            print("ChatGoogleGenerativeAI (Gemini) model initialized with 'models/gemini-1.5-pro-latest'.")
        except Exception as e:
            print(f"Error initializing ChatGoogleGenerativeAI: {e}")
            llm = None

    if not llm:
        print("LLM not initialized. Agent creation will likely fail or be non-functional.")
        return None

    # Initialize tools
    tools = [DuckDuckGoSearchRun()]

    # Create the ReAct agent
    try:
        agent = create_react_agent(llm, tools, prompt)
    except Exception as e:
        print(f"Error creating React agent with Gemini: {e}")
        return None

    # Create an agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

def get_agent_response(query: str, agent_executor: AgentExecutor):
    """
    Gets a response from the Langchain agent.
    """
    if agent_executor is None:
        return "Sorry, the agent is not available (LLM not initialized)."

    try:
        response = agent_executor.invoke({"input": query})
        return response.get("output", "Sorry, I could not process that.")
    except Exception as e:
        print(f"Agent Error: {e}")
        return "Sorry, I encountered an error while processing your request with the agent."

if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == 'YOUR_GOOGLE_API_KEY_HERE':
        print("Google API key not found or is placeholder. Please set it in .env or as an environment variable.")
        print("Example .env content: GOOGLE_API_KEY='YOUR_ACTUAL_KEY'")
    else:
        print("Attempting to create Gemini agent (models/gemini-1.5-pro-latest) and get a response...")
        try:
            agent_executor = create_agent()
            if agent_executor:
                test_query = "What's the latest version of Google Chrome using Gemini 1.5 Pro?"
                print(f"Testing agent with query: {test_query}")
                response = get_agent_response(test_query, agent_executor)
                print(f"Agent Response: {response}")
            else:
                print("Agent executor not created. Cannot run test query.")
        except Exception as e:
            print(f"Error during agent test: {e}")
