from langchain import hub
from langchain.prompts import ChatPromptTemplate, PromptTemplate # Ensure PromptTemplate is imported for older message types
import sys

# It seems like newer versions of hub.pull might return a ChatPromptTemplate,
# older versions might return a PromptTemplate directly or something else.
# The structure of prompt.messages[i].prompt.template is common.

try:
    prompt = hub.pull("hwchase17/react")
    print(f"Prompt object type: {type(prompt)}")

    if isinstance(prompt, ChatPromptTemplate):
        print("\n--- Prompt Template (as ChatPromptTemplate) ---")
        # For ChatPromptTemplate, the overall template isn't a single string but a list of message templates.
        # We will print each message template's content below.
        print("Note: ChatPromptTemplate consists of multiple message templates.")

        print("\n--- Input Variables ---")
        print(prompt.input_variables)

        print("\n--- Messages ---")
        for i, message_template in enumerate(prompt.messages):
            print(f"Message {i}:")
            print(f"  Type: {type(message_template)}")
            # Accessing the template string within each message object
            if hasattr(message_template, 'prompt') and hasattr(message_template.prompt, 'template'):
                 print(f"  Content Template: {message_template.prompt.template}")
            elif hasattr(message_template, 'template'): # For older/simpler message types
                 print(f"  Content Template: {message_template.template}")
            else:
                 print(f"  Content Template: Could not find template string in message: {message_template}")

    elif isinstance(prompt, PromptTemplate): # Handling if it's a basic PromptTemplate
        print("\n--- Prompt Template (as PromptTemplate) ---")
        print(prompt.template)
        print("\n--- Input Variables ---")
        print(prompt.input_variables)

    else: # Fallback for other types
        print("\n--- Prompt Details (Unknown Type) ---")
        print("Prompt template string (if available via .template):")
        if hasattr(prompt, 'template'):
            print(prompt.template)
        else:
            print("N/A")

        print("\nInput Variables (if available via .input_variables):")
        if hasattr(prompt, 'input_variables'):
            print(prompt.input_variables)
        else:
            print("N/A")

        print("\nFull prompt object string representation:")
        print(str(prompt))


except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)
    sys.exit(1)
