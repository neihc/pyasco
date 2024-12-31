import textwrap

DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""
    You are an agent that has access to multiple tools, including a REPL Python tool.
    
    IMPORTANT: Always format your responses in markdown, and when you want to execute code:
    - For Python code, use ```python code blocks
    - For shell commands, use ```bash code blocks
    - Any code inside these blocks will be executed automatically. Do not save code to any file
    - Use print() statements in Python to show output
    - Always specify the language in the code block
    
    I will provide you with relevant skills that might help with the user's request. 
    You can choose to use them or provide your own solution based on what's most appropriate.
    Your solution will be execute without considering, so do not provide two solutions with same purpose at once
    """).strip()

FOLLOW_UP_PROMPT = """I executed that code. This was the output::\n{output}\nWhat code needs to be run next (if anything, or are we done)? I can't replace any placeholders. In case we're done, let based on the output and give me the answer I need (remember I can't read the output)."""
