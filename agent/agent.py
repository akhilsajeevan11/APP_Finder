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

    # Agent Prompt Creation and Customization
    agent = None
    try:
        # Create the default prompt for the agent
        # This prompt already includes input_variables=['input', 'chat_history', 'agent_scratchpad']
        # and a MessagesPlaceholder(variable_name='chat_history')
        base_prompt = ConversationalChatAgent.create_prompt(tools=tools)

        # Define our custom instructions for downloads
        download_instructions = (
            "\n\nSPECIFIC TASK: PROVIDING SOFTWARE DOWNLOAD LINKS\n"
            "When a user explicitly expresses a desire to download software, an application, or a file (e.g., 'I want to download X', 'download X', 'get X for download'):\n"
            "1. Your primary goal is to PROVIDE a relevant download link directly to the user in your response.\n"
            "2. First, identify the official or most reputable source website for the software (e.g., 'ubuntu.com' for Ubuntu, 'python.org' for Python).\n"
            "   - If you are confident about the official source website, proceed to step 3.\n"
            "   - If you are unsure of the official source, use your search tool ONCE to find this official source website. Prioritize results that look like official project pages or vendor sites.\n"
            "3. Once the official source website is identified, use your search tool to find the SPECIFIC DOWNLOAD PAGE URL on that website. For example, if the source is 'ubuntu.com', search for 'official Ubuntu download page link' or 'Ubuntu 24.04 download link'. Let's call this the 'download page URL'.\n"
            # Point 4 is about determining the best link from the information obtained.
            "4. From the information retrieved from the 'download page URL', determine the best link(s) to provide to the user based on the following prioritization:\n"
            # Aggressively Refined Point 5:
            "5. You MUST extensively scan the content of the 'download page URL' (from step 3) for any hyperlinks that appear to be DIRECT download links to installable files (e.g., URLs ending in .exe, .dmg, .pkg, .zip, .tar.gz, .msi). \n"
            "   a. If you identify one or more such potential direct links from this page, you MUST list up to three of the most relevant-looking ones. For each, state the full URL. Clearly label these as 'potential direct file download link(s)'.\n"
            "   b. If multiple direct links were found and you are unsure which is best for the user (e.g., different OS versions not specified by user), briefly state this uncertainty when listing them.\n"
            "   c. If, after extensively scanning the 'download page URL', you find no URLs that appear to be direct installable file links, you MUST explicitly state: 'I scanned the page at [URL of download page] but could not identify a clear, direct link to an installable file.'\n"
            # Point 6 is now a clear fallback if step 5c was reached.
            "6. If you stated in step 5c that you could not find direct installable file links, OR if the 'download page URL' inherently requires user interaction to choose from many critical options (like OS, architecture, or major software editions), then provide the link to this main 'download page URL'. When doing so, briefly mention that the user will need to select the appropriate option or navigate further on that page.\n"
            "7. If, after searching (steps 2 & 3), you cannot even find a clear download page or link from an official/reputable source, inform the user of this. Do not invent links.\n"
            "8. Do not just mention the website name (e.g., 'go to ubuntu.com'); your task is to provide the actual URL to the most direct and useful download resource you found, as per these instructions.\n"
            # Phrasing Rules (9-11) remain the same:
            "9. When you provide any download link or a link to a download page as per the steps above, present it clearly as the resource for the USER to initiate the download. "
            "For example, use phrasing like: 'You can download [Software Name] from the official page here: [link]' or 'Here is a potential direct download link for [Software Name] that you can use: [link]'.\n"
            "10. If you have successfully found and are providing a URL for downloading in your response, AVOID simultaneously stating 'I cannot download files' or similar self-limitations in that same response, as this can confuse the user. Your focus should be on successfully guiding the user to the link you've provided.\n"
            "11. If the user later asks *why* they have to do the download themselves (e.g., 'why can't you download it for me?'), then it is appropriate to explain your nature as an AI that can provide information and links but cannot perform actions like file downloads on their computer.\n"
            # New point #12
            "12. Once you have identified what you believe to be an official or reputable source website for the software (as per step 2) and are providing a link from that source "
            "(whether a direct file link as per step 5 or a download page link as per step 6), you should NOT then state that providing this link is 'risky' or 'unsafe'. "
            "Your prior steps have already involved selecting a trustworthy source. Your role is to provide the user with the means to access software from that official source. "
            "The user bears the ultimate responsibility for downloading and installing software."
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
