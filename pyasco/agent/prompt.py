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

LEARN_SKILL_PROMPT = """Based on our conversation above, please consolidate the key functionality into a reusable skill. You MUST include ALL of these sections in your response:

SKILL NAME: <name of the skill>
USAGE: <brief description of what the skill does and how to use it>
REQUIREMENTS: <comma-separated list of pip packages, or 'none' if no requirements>
CODE:
```python
<the actual Python code for the skill, just define the function/class and do not contain example call here>
```{error_feedback}"""

IMPROVE_SKILL_PROMPT = """{conversation}
Based on our conversation above and the existing skill below, please improve the skill by updating its usage description and code as needed. You may need to put more information on usage to avoid mistake this time{error_info}

EXISTING SKILL:
NAME: {skill_name}
USAGE: {skill_usage}
Please provide the improved version in this format, Do not change SKILL NAME:
SKILL NAME: {skill_name}
USAGE: <improved description of what the skill does and how to use it>
REQUIREMENTS: <comma-separated list of pip packages, or 'none' if no requirements>
CODE:
```python
<improved Python code for the skill>
```
"""

IDENTIFY_SKILL_PROMPT = """{conversation}
Based on our conversation above, which of these existing skills would be most appropriate to improve? Reply with JUST the skill name, nothing else.

Available skills:
{available_skills}"""
