from langchain import hub
import sys

try:
    prompt_template_obj = hub.pull("hwchase17/react")
    react_prompt_text = ""
    if hasattr(prompt_template_obj, 'template'):
        react_prompt_text = prompt_template_obj.template
        # Using a more robust way to print for shell capture
        print("---REACT PROMPT TEXT START---")
        print(react_prompt_text)
        print("---REACT PROMPT TEXT END---")
    else:
        print("ERROR: Could not get template text from hub.pull('hwchase17/react')", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)
    sys.exit(1)
