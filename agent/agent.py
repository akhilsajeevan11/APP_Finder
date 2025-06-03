import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder # Useful if we want to explicitly define where memory goes
)
# Note: ConversationalChatAgent.create_prompt already includes a MessagesPlaceholder for chat_history

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
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=google_api_key)
            print("ChatGoogleGenerativeAI (Gemini) model initialized.")
        except Exception as e:
            print(f"Error initializing ChatGoogleGenerativeAI: {e}")
            # llm remains None

    if not llm:
        print("LLM not initialized. Agent creation will fail.")
        return None, None # Return None for agent and tools

    # Initialize tools
    tools = [DuckDuckGoSearchRun()]

    # Agent Prompt Creation and Customization
    agent = None
    try:
        # Create the default prompt for the agent
        # This prompt already includes input_variables=['input', 'chat_history', 'agent_scratchpad']
        # and a MessagesPlaceholder(variable_name='chat_history')
        base_prompt = ConversationalChatAgent.create_prompt(tools=tools)

        # Define our custom instructions for downloads
        download_instructions = (
            "\n\nASSISTING WITH SOFTWARE DOWNLOADS:\n"
            "When a user asks to download software, an application, or a file:\n"
            "1. Your primary goal is to guide the user to the official public download page for that software.\n"
            "2. Use your search tool to identify the OFFICIAL website (e.g., 'ubuntu.com' for Ubuntu, 'python.org' for Python). If you are unsure, search for it.\n"
            "3. Once the official website is identified, use your search tool again to find the main public DOWNLOAD PAGE URL on that site. For example, search for 'official Ubuntu download page' or '[Software Name] download page'.\n"
            "4. Provide this full URL to the user directly in your response. For example: 'You can find downloads for [Software Name] on their official page: [URL]'.\n"
            "5. IMPORTANT: It is PERMISSIBLE, SAFE, and HELPFUL for you to provide links to official, publicly accessible download pages. This is a key part of assisting the user. Providing such links is NOT a violation of terms of service or copyright for publicly available software, as you are merely directing the user to the publicly provided resource.\n"
            "6. Do NOT invent links. If, after searching, you cannot find an official download page, inform the user clearly.\n"
            "7. AVOID making unsolicited statements about copyright, terms of service, or the 'risks' of downloading, unless you are directly asked about such topics or have specific, programmed knowledge about a particular piece of software's licensing. Focus on providing the link to the official public download page."
        )

        # Prepend our instructions to the existing system message in the prompt
        new_messages = []
        system_message_modified = False
        for msg_template in base_prompt.messages:
            if isinstance(msg_template, SystemMessagePromptTemplate):
                # Assuming msg_template.prompt is a PromptTemplate or similar with a 'template' attribute
                if hasattr(msg_template, 'prompt') and hasattr(msg_template.prompt, 'template'):
                    original_system_message = msg_template.prompt.template
                    custom_system_message_content = download_instructions + "\n\n" + original_system_message
                    new_messages.append(SystemMessagePromptTemplate.from_template(custom_system_message_content))
                    system_message_modified = True
                else: # Should not happen with standard create_prompt
                    new_messages.append(msg_template) # Keep original if structure is unexpected
            else:
                new_messages.append(msg_template)

        if not system_message_modified:
            print("WARNING: Could not find or modify SystemMessagePromptTemplate in default agent_prompt. Custom download instructions may not be applied.")
            # Potentially add the instructions as a new system message if none was found, though this is unlikely
            # For now, we'll proceed with new_messages which would be same as base_prompt.messages

        custom_agent_prompt = ChatPromptTemplate.from_messages(new_messages)

        # Instantiate the agent with the modified prompt
        agent = ConversationalChatAgent(llm=llm, tools=tools, prompt=custom_agent_prompt)
        print("ConversationalChatAgent initialized with custom download instructions.")

    except Exception as e:
        print(f"Error creating ConversationalChatAgent with custom prompt: {e}")
        # Fallback to default prompt agent if custom fails
        try:
            print("Attempting to fall back to default ConversationalChatAgent...")
            agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools) # Default prompt
            print("Fell back to default ConversationalChatAgent successfully.")
        except Exception as fallback_e:
            print(f"Error creating fallback default ConversationalChatAgent: {fallback_e}")
            return None, None # Critical failure

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
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = session_memories[session_id]

    # AgentExecutor Creation
    agent_executor = None
    try:
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )
    except Exception as e:
        print(f"Error creating AgentExecutor for session {session_id}: {e}")
        return f"Sorry, I encountered an error setting up the agent for your session: {e}"

    # Invoke Agent
    try:
        response = agent_executor.invoke({"input": query})
        return response.get("output", "Sorry, I could not process that.")
    except Exception as e:
        print(f"Agent Error during invoke for session {session_id}: {e}")
        if "Could not parse LLM output:" in str(e) or isinstance(e, Exception):
            return "Sorry, the agent's response was not in the expected format. Please try rephrasing your question or try again later."
        return f"Sorry, I encountered an error while processing your request with the agent: {e}"


if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == 'YOUR_GOOGLE_API_KEY_HERE':
        print("Google API key not found or is placeholder.")
    else:
        print("Attempting to create agent with custom download instructions...")
        test_agent, test_tools = create_agent()
        if test_agent and test_tools:
            print("Agent and tools created successfully.")

            test_session_id = "download_test_session"
            print(f"\n--- Test Query 1 (Download Request) for session {test_session_id} ---")
            # Example of how the agent might be invoked (directly calling get_agent_response)
            response1 = get_agent_response("I want to download Google Chrome.", test_agent, test_tools, test_session_id)
            print(f"Response 1: {response1}")

            print(f"\n--- Test Query 2 (Follow-up) for session {test_session_id} ---")
            response2 = get_agent_response("Is it available for Windows?", test_agent, test_tools, test_session_id)
            print(f"Response 2: {response2}")

            print(f"\n--- Test Query 3 (General Question) for session {test_session_id} ---")
            response3 = get_agent_response("What's the weather like?", test_agent, test_tools, test_session_id)
            print(f"Response 3: {response3}")

        else:
            print("Agent and/or tools creation failed in __main__.")
