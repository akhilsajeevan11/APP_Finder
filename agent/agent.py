import os
# from langchain_openai import ChatOpenAI # Commented out OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory # Added for memory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.prompts import PromptTemplate # Added for custom prompt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_agent():
    # Get the prompt template
    # prompt = hub.pull("hwchase17/react") # Commented out to use custom prompt

    # Define the original react prompt text (fetched in a previous step)
    react_prompt_text = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: First, I MUST review the CHAT HISTORY for relevant context and to understand any ambiguous references in the user's input. After reviewing the history, I will think about what to do next to answer the Question.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    # The above react_prompt_text already incorporates the modified 'Thought:' line as per instructions.
    # No explicit .replace() call is needed here if we define it directly with the new 'Thought:' line.

    new_instructions = (
        "You are a helpful and friendly assistant. Your primary goal is to be responsive and follow user instructions accurately.\n"
        "IMPORTANT CHAT HISTORY USAGE:\n"
        "Pay close attention to the CHAT HISTORY. The chat history is part of the information available to you during your 'Thought' process. \n"
        "You MUST use this CHAT HISTORY to understand context, resolve ambiguities in user questions (e.g., references like 'it', 'that', 'those', 'these'), and answer follow-up questions effectively. \n"
        "If a question is vague, check the CHAT HISTORY to see if it refers to something discussed earlier before asking for clarification.\n"
        "\nCRITICAL RESPONSE FORMATTING RULES:\n"
        "1. You MUST strictly follow the format: Thought, Action, Action Input, Observation, and Final Answer.\n"
        "2. Do NOT deviate from this structure. No conversational text outside this structure, except within the 'Thought' or 'Final Answer' sections as appropriate.\n"
        "3. Even if you know the answer directly from CHAT HISTORY or your general knowledge, you MUST still formulate a 'Thought:' explaining how you arrived at the answer, and then provide the 'Final Answer:'.\n"
        "4. If you need to use a tool, you must go through the full Thought, Action, Action Input sequence.\n\n"
    )
    custom_prompt_text = new_instructions + react_prompt_text # react_prompt_text is already the modified version
    prompt = PromptTemplate.from_template(custom_prompt_text)
    # The input variables ['agent_scratchpad', 'input', 'tool_names', 'tools']
    # are expected to be inferred correctly by from_template.
    # If issues arise, uncomment and set explicitly:
    # prompt.input_variables = ['agent_scratchpad', 'input', 'tool_names', 'tools', 'chat_history']
    # Note: 'chat_history' is handled by memory and injected into 'agent_scratchpad' by the agent executor typically for ReAct.
    # The custom instructions refer to CHAT HISTORY, which the agent needs to be aware of conceptually.
    # The memory mechanism (ConversationBufferMemory with memory_key="chat_history") will make actual history available.
    # The agent's internal logic (via agent_scratchpad) should make use of this.

    # Initialize the LLM
    # --- OpenAI LLM (Commented Out) ---
    # llm = ChatOpenAI(temperature=0)
    # Make sure OPENAI_API_KEY is set in your environment or .env file

    # +++ Google Gemini LLM (New) +++
    # Make sure GOOGLE_API_KEY is set in your environment or .env file
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == 'YOUR_GOOGLE_API_KEY_HERE':
        print("WARNING: GOOGLE_API_KEY not found or is a placeholder. Gemini agent will not function.")
        llm = None # Or handle this case as preferred, e.g., raise an error
    else:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=google_api_key)
            print("ChatGoogleGenerativeAI (Gemini) model initialized.")
        except Exception as e:
            print(f"Error initializing ChatGoogleGenerativeAI: {e}")
            llm = None

    if not llm:
        print("LLM not initialized. Agent creation will likely fail or be non-functional.")
        # Depending on Langchain version and agent type,
        # passing None LLM might raise error immediately or later.
        # For robust error handling, this should ideally be caught before creating agent_executor
        return None # Explicitly return None if LLM failed to initialize

    # Initialize tools
    tools = [DuckDuckGoSearchRun()]

    # Initialize memory
    # The ReAct prompt hwchase17/react expects 'chat_history' as an input variable
    memory = ConversationBufferMemory(memory_key="chat_history") # return_messages defaults to False

    # Create the ReAct agent
    # This might fail if llm is None, depending on Langchain's internal checks.
    try:
        agent = create_react_agent(llm, tools, prompt)
    except Exception as e:
        print(f"Error creating React agent with Gemini: {e}")
        return None

    # Create an agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

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
    # Example usage (for testing agent.py directly)
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == 'YOUR_GOOGLE_API_KEY_HERE':
        print("Google API key not found or is placeholder. Please set it in .env or as an environment variable.")
        print("Example .env content: GOOGLE_API_KEY='YOUR_ACTUAL_KEY'")
    else:
        print("Attempting to create Gemini agent and get a response (this requires network access and valid API key)...")
        try:
            agent_executor = create_agent()
            if agent_executor:
                test_query = "What's the latest version of Google Chrome using Gemini?"
                print(f"Testing agent with query: {test_query}")
                response = get_agent_response(test_query, agent_executor)
                print(f"Agent Response: {response}")
            else:
                print("Agent executor not created. Cannot run test query.")
        except Exception as e:
            print(f"Error during agent test: {e}")
