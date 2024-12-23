import re
import os
import platform
import psutil
import textwrap
from typing import List, Dict, Optional, Generator, Iterator, Any, Union
from dataclasses import dataclass

from .logger_config import setup_logger
from .config import Config
from .services.llm import get_openai_response
from .services.code_snippet_extractor import CodeSnippetExtractor
from .services.skill_manager import SkillManager, Skill
from .tools.code_execute import CodeExecutor

@dataclass
class AgentResponse:
    """Represents a response from the agent"""
    role: str = "assistant"
    content: str = ""
    tools: List[Dict] = None
    done: bool = True

    def __post_init__(self):
        if self.tools is None:
            self.tools = []

class Agent:
    """
    Agent that handles user interactions, tool determination, and execution.
    
    This class manages the conversation flow, tool execution, and maintains
    the chat history. It supports both streaming and non-streaming responses.
    """
    
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

    def __init__(self, config: Config):
        """Initialize the agent with configuration
        
        Args:
            config: Configuration object containing all settings
        """
        self.logger = setup_logger('agent')
        self.logger.info("Initializing Agent")
        self.messages: List[Dict] = []
        self.code_extractor = CodeSnippetExtractor()
        
        docker_options = None
        if config.docker.use_docker:
            docker_options = {
                'mem_limit': config.docker.mem_limit,
                'cpu_count': config.docker.cpu_count,
                'volumes': {}
            }
            
            # Add configured volumes
            if config.docker.volumes:
                docker_options['volumes'].update(config.docker.volumes)
            
            # Add skills directory mount if not already configured
            if config.skills_path not in docker_options['volumes']:
                docker_options['volumes'][config.skills_path] = {
                    'bind': '/skills',
                    'mode': 'ro'
                }
        
        self.python_executor = CodeExecutor(
            use_docker=config.docker.use_docker,
            docker_image=config.docker.image,
            docker_options=docker_options,
            bash_shell=config.docker.bash_command,
            python_command=config.docker.python_command,
            env_file=config.docker.env_file
        )
        
        self.custom_instructions = None  # Can be set later if needed
        self.model = config.llm.model
        # Configure LLM client
        from .services.llm import configure_client
        configure_client(api_key=config.llm.api_key, base_url=config.llm.base_url)
        self.skill_manager = SkillManager(config.skills_path)
        
        # Install requirements in Docker container during initialization
        if config.docker.use_docker:
            requirements = self.skill_manager.get_requirements()
            if requirements:
                docker_options.setdefault('command', [])
                docker_options['command'].extend([
                    "pip", "install", "-r", "/skills/requirements.txt"
                ])
                
        self._initialize_chat()

    def _format_skills_info(self) -> str:
        """Format available skills information for the prompt"""
        if not self.skill_manager.skills:
            return "No skills available yet."
            
        skill_info = []
        for name, skill in self.skill_manager.skills.items():
            skill_info.append(f"- {name}: {skill.usage}")
        return "\n".join(skill_info)

    def _get_system_info(self, use_docker: bool = False, docker_image: str = None) -> str:
        """Get current system information including environment variables"""
        if use_docker:
            # Get system info from inside Docker container
            code = """
import platform
import os

env_vars = '\\n'.join(f'- {k}' for k, v in sorted(os.environ.items()))
print(f'''System Information:
- OS: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- CPU Architecture: {platform.machine()}
- Execution Environment: Docker container using {docker_image}

Environment Variables:
{env_vars}''')
""".replace("{docker_image}", docker_image)
            stdout, _ = self.python_executor.execute(code)
            return stdout.strip()
        else:
            # Get host system info
            memory = psutil.virtual_memory()
            env_vars = '\n'.join(f'- {k}' for k, v in sorted(os.environ.items()))
            return f"""System Information:
- OS: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- CPU Architecture: {platform.machine()}
- Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available

Environment Variables:
{env_vars}"""

    def _initialize_chat(self) -> None:
        """Initialize chat with system prompt including available skills"""
        system_info = self._get_system_info(self.python_executor.use_docker,
                                        self.python_executor.docker_image)
        base_prompt = f"{self.DEFAULT_SYSTEM_PROMPT}\n\n{system_info}"
        system_content = f"{base_prompt}\n\n{self.custom_instructions}" if self.custom_instructions else base_prompt
        self.messages.append({
            "role": "system",
            "content": system_content
        })
    
    def _prepare_messages(self) -> List[Dict]:
        """Prepare messages for LLM by filtering out tool and skills information"""
        return [{k: v for k, v in msg.items() if k not in ["tools", "skills"]}
                for msg in self.messages]

    def _create_tool_response(self, content: str) -> List[Dict]:
        """Create tool response based on code snippets and commands"""
        tools = []
        
        # Handle code execution
        snippets = self.code_extractor.extract_snippets(content)
        if snippets:
            tools.append({
                "name": "python_executor",
                "parameters": {"snippets": snippets}
            })
        
        return tools

    def _handle_streaming_response(self, llm_response: Iterator[Any]) -> Generator[AgentResponse, None, None]:
        """Handle streaming response from LLM"""
        full_content = ""
        
        for chunk in llm_response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                yield AgentResponse(content=content, done=False)
        
        tools = self._create_tool_response(full_content)
        final_response = AgentResponse(content=full_content, tools=tools)
        self.messages.append(final_response.__dict__)
        yield final_response

    def get_response(self, user_input: str, stream: bool = False) -> Union[AgentResponse, Generator[AgentResponse, None, None]]:
        self.logger.info(f"Getting response for user input (stream={stream})")
        """
        Get assistant response with tools based on the user's input.
        
        Args:
            user_input: The user's message
            stream: Whether to stream the response
        
        Returns:
            Either a single AgentResponse or a generator of AgentResponses for streaming
        """
        # Get list of all available skills
        available_skills = list(self.skill_manager.skills.keys())
        skill_list = "\n".join(f"- {skill}" for skill in available_skills)
        
        # Ask LLM to select relevant skills by name
        skill_prompt = (
            "Based on the following user input, select which of these skills would be most relevant. "
            "Reply with ONLY the exact skill names, one per line, maximum 3 skills. "
            "If no skills are relevant, reply with 'none'.\n\n"
            f"User input: {user_input}\n\n"
            f"Available skills:\n{skill_list}\n\n"
            "Selected skills:"
        )
        
        skill_response = get_openai_response([{
            "role": "user",
            "content": skill_prompt
        }], model=self.model)
        
        # Parse selected skill names and get the actual skill objects
        selected_skill_names = [name.strip() for name in skill_response.split('\n') if name.strip()]
        relevant_skills = []
        for name in selected_skill_names:
            if name.lower() != 'none' and name in self.skill_manager.skills:
                relevant_skills.append(self.skill_manager.skills[name])
        
        # Check for existing skills in previous messages
        existing_skills = set()
        for msg in self.messages:
            if msg.get("skills"):
                existing_skills.update(skill["name"] for skill in msg["skills"])
        
        # Only append skills that haven't been mentioned before
        new_skills = [skill for skill in relevant_skills if skill.name not in existing_skills]
        
        if new_skills:
            # First install requirements and load the skills
            skills_info = "\n\nLoading and making available these relevant skills:\n\n"
            for skill in new_skills:
                skill_code = self.skill_manager.get_skill_code(skill)
                
                # Install requirements if any
                if skill.requirements:
                    req_install = f"pip install {' '.join(skill.requirements)}"
                    stdout, stderr = self.python_executor.execute(req_install, 'bash')
                    if stderr and "ERROR:" in stderr:
                        self.logger.error(f"Error installing requirements for {skill.name}: {stderr}")
                        continue
                
                # Execute the skill code to make functions available
                stdout, stderr = self.python_executor.execute(skill_code)
                if stderr:
                    self.logger.warning(f"Warning while loading skill {skill.name}: {stderr}")
                
                skills_info += f"### {skill.name}\n"
                skills_info += f"**Usage:** {skill.usage}\n"
                skills_info += f"**Note:** This skill's functions are now loaded and ready to use. "
                skills_info += f"Do not redefine them unless you need to modify their behavior.\n"
                skills_info += f"**Code:**\n```python\n{skill_code}\n```\n\n"
            user_input += skills_info

        self.messages.append({
            "role": "user", 
            "content": user_input,
            "skills": [skill.to_dict() for skill in new_skills] if new_skills else []
        })
        filtered_messages = self._prepare_messages()
        llm_response = get_openai_response(filtered_messages, model=self.model, stream=stream)
        
        if stream:
            return self._handle_streaming_response(llm_response)
        
        response = AgentResponse(content=llm_response)
        response.tools = self._create_tool_response(llm_response)
        message_dict = response.__dict__.copy()
        message_dict["skills"] = []  # Add empty skills list for assistant messages
        self.messages.append(message_dict)
        return response
    
    def call_tool(self, tool_name: str, params: Dict) -> List[str]:
        self.logger.info(f"Calling tool: {tool_name}")
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to invoke
            params: Parameters required for the tool
        
        Returns:
            List[str]: Results from tool execution (stdout/stderr)
        
        Raises:
            ValueError: If tool_name is not supported
        """
        if tool_name == "python_executor":
            execution_results = []
            for snippet in params.get("snippets", []):
                if snippet.language:
                    language = snippet.language.lower()
                    stdout = stderr = None
                    if 'python' in language or 'bash' in language:
                        stdout, stderr = self.python_executor.execute(snippet.content, language)
                    if stdout or stderr:
                        execution_results.append(f"Output:\n{stdout or ''}")
                        if stderr:
                            execution_results.append(f"Errors:\n{stderr}")
            return execution_results
        
        return []
    
    def ask(self, new_input: str, stream: bool = False) -> Dict:
        """
        Get response for new input and append to message history.
        Does not execute any tools.
        
        Args:
            new_input: The user's message
            stream: Whether to stream the response
            
        Returns:
            Dict: Assistant response with content and tools, and stream if enabled
        """
        response = self.get_response(new_input, stream=stream)
        
        if stream:
            return response
        
        # For non-streaming responses, append to history and print
        self.messages.append(response)
        return response
        
    def confirm(self) -> List | None:
        """
        Execute tools from the latest assistant response and get follow-up response.
        
        Returns:
            Dict: New assistant response after tool execution, or None if no tools to run
        """
        if not self.messages:
            return None
            
        last_response = self.messages[-1]
        if last_response["role"] != "assistant" or not last_response.get("tools"):
            return None
            
        execution_results = []
        for tool in last_response["tools"]:
            execution_results.extend(self.call_tool(tool["name"], tool["parameters"]))
            
        return execution_results
    
    def reset(self):
        """Reset the agent's state"""
        self.logger.info("Resetting agent state")
        self.messages = []
        self.python_executor.reset()
        self._initialize_chat()
    
    def cleanup(self):
        """Cleanup resources properly"""
        self.logger.info("Cleaning up agent resources")
        self.python_executor.cleanup()
        
    def parse_skill_response(self, response_content: str) -> Dict[str, Any]:
        """Parse a skill learning response into components.
        
        Args:
            response_content: The LLM response content
            
        Returns:
            Dict containing name, usage, code, and requirements
            
        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Extract skill name
            name_match = re.search(r"SKILL NAME:\s*(.+?)(?=\n|$)", response_content)
            name = name_match.group(1).strip() if name_match else None

            # Extract usage
            usage_match = re.search(r"USAGE:\s*(.+?)(?=\n|$)", response_content)
            usage = usage_match.group(1).strip() if usage_match else None

            # Extract requirements
            req_match = re.search(r"REQUIREMENTS:\s*(.+?)(?=\n|$)", response_content)
            requirements_str = req_match.group(1).strip() if req_match else "none"
            requirements = [r.strip() for r in requirements_str.split(",")] if requirements_str.lower() != "none" else []

            # Extract code
            code_snippets = self.code_extractor.extract_snippets(response_content)
            code = code_snippets[0].content if code_snippets else None

            if not all([name, usage, code]):
                raise ValueError("Missing required skill components")

            # Include all necessary fields including the code
            skill_data = {
                "name": name,
                "usage": usage,
                "file_path": f"{name.lower().replace(' ', '_')}.py",
                "requirements": requirements,
                "code": code
            }
            return skill_data
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Invalid skill response format: {str(e)}")

    def learn_that_skill(self) -> Skill:
        """Convert the current conversation into a reusable skill.
        Takes the conversation history and asks the LLM to consolidate it
        into a skill in the correct format.
        
        Returns:
            Skill: The newly learned skill
            
        Raises:
            ValueError: If skill response is invalid or no messages to learn from
        """
        if len(self.messages) < 2:  # Need at least system + 1 interaction
            raise ValueError("Not enough conversation history to learn from")
            
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Create prompt to consolidate conversation into skill
                conversation = "\n".join(f"{msg['role']}: {msg['content']}" 
                                    for msg in self.messages[1:])  # Skip system message
                
                error_feedback = ""
                if last_error:
                    error_feedback = (
                        f"\n\nPrevious attempt failed with error: {last_error}"
                        "\nPlease ensure your response includes ALL required sections:"
                        "\n- SKILL NAME (required)"
                        "\n- USAGE (required)"
                        "\n- REQUIREMENTS (required, use 'none' if no requirements)"
                        "\n- CODE section with ```python code block (required)"
                    )
                
                prompt = (
                    "Based on our conversation above, please consolidate the key functionality "
                    "into a reusable skill. You MUST include ALL of these sections in your response:"
                    "\n\nSKILL NAME: <name of the skill>"
                    "\nUSAGE: <brief description of what the skill does and how to use it>"
                    "\nREQUIREMENTS: <comma-separated list of pip packages, or 'none' if no requirements>"
                    "\nCODE:\n"
                    "```python\n"
                    "<the actual Python code for the skill>\n"
                    "```"
                    f"{error_feedback}"
                )
        
                response = self.get_response(prompt)
                if isinstance(response, Generator):
                    # Handle streaming response
                    content = ""
                    for chunk in response:
                        content += chunk.content
                    response_content = content
                else:
                    response_content = response.content
                    
                skill_data = self.parse_skill_response(response_content)
                
                return self.learn_skill(
                    name=skill_data["name"],
                    usage=skill_data["usage"],
                    code=skill_data["code"],
                    requirements=skill_data["requirements"]
                )
                
            except ValueError as e:
                last_error = str(e)
                retry_count += 1
                self.logger.warning(f"Attempt {retry_count} failed: {last_error}")
                
                if retry_count >= max_retries:
                    raise ValueError(
                        f"Failed to learn skill after {max_retries} attempts. "
                        f"Last error: {last_error}"
                    )
                
                # Continue to next retry
                continue
        
    def learn_skill(self, name: str, usage: str, code: str, requirements: List[str] = None) -> Skill:
        """Learn a new skill with optional requirements"""
        self.logger.info(f"Learning new skill: {name}")
        try:
            skill = self.skill_manager.learn(name, usage, code, requirements)
            self.logger.info(f"Successfully learned skill: {name}")
            return skill
        except Exception as e:
            self.logger.error(f"Failed to learn skill {name}: {str(e)}")
            raise
            
    def improve_that_skill(self, skill_name: Optional[str] = None) -> Optional[Skill]:
        """Improve an existing skill based on the current conversation.
        Takes the conversation history and asks the LLM to improve the specified skill.
        If no skill_name is provided, automatically determines which skill to improve.
        
        Args:
            skill_name: Optional name of the skill to improve. If None, auto-determines from conversation.
            
        Returns:
            Skill: The improved skill or None if skill not found
            
        Raises:
            ValueError: If skill response is invalid or no messages to learn from
        """
        if len(self.messages) < 2:
            raise ValueError("Not enough conversation history to learn from")
            
        error_context = []
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if skill_name is None:
                    # Create prompt to identify skill from conversation
                    conversation = "\n".join(f"{msg['role']}: {msg['content']}" 
                                        for msg in self.messages[1:])
                    
                    identify_prompt = (
                        f"{conversation}\n"
                        "Based on our conversation above, which of these existing skills would be most "
                        "appropriate to improve? Reply with JUST the skill name, nothing else.\n\n"
                        "Available skills:\n"
                    )
                    
                    # Add available skills to prompt
                    for name, skill in self.skill_manager.skills.items():
                        identify_prompt += f"- {name}: {skill.usage}\n"
                    
                    response = get_openai_response([{
                        "role": "user",
                        "content": identify_prompt
                    }], model=self.model)
                    
                    # Clean up response to get just the skill name
                    skill_name = response.strip()
                    
                # Get existing skill
                skill = self.skill_manager.get_skill(skill_name)
                if not skill:
                    raise ValueError(f"Skill '{skill_name}' not found")
                    
                # Get existing code
                existing_code = self.skill_manager.get_skill_code(skill)
                    
                # Create prompt to improve skill
                conversation = "\n".join(f"{msg['role']}: {msg['content']}" 
                                    for msg in self.messages[1:])
                
                # Add error context if any previous attempts failed
                error_info = ""
                if error_context:
                    error_info = "\nPrevious attempts failed with these errors:\n" + "\n".join(
                        f"Attempt {i+1}: {err}" for i, err in enumerate(error_context)
                    ) + "\nPlease address these issues in your improvement."
                
                prompt = (
                    f"{conversation}\n"
                    f"Based on our conversation above and the existing skill below, please improve "
                    f"the skill by updating its usage description and code as needed. You may need to put more information on usage to avoid mistake this time{error_info}\n\n"
                    f"EXISTING SKILL:\n"
                    f"NAME: {skill.name}\n"
                    f"USAGE: {skill.usage}\n"
                    f"Please provide the improved version in this format, Do not change SKILL NAME:\n"
                    f"SKILL NAME: {skill.name}\n"
                    f"USAGE: <improved description of what the skill does and how to use it>\n"
                    f"REQUIREMENTS: <comma-separated list of pip packages, or 'none' if no requirements>\n"
                    f"CODE:\n"
                    f"```python\n"
                    f"<improved Python code for the skill>\n"
                    f"```\n"
                )
                
                response = self.get_response(prompt)
                if isinstance(response, Generator):
                    content = ""
                    for chunk in response:
                        content += chunk.content
                    response_content = content
                else:
                    response_content = response.content
                    
                skill_data = self.parse_skill_response(response_content)
                
                # Verify the skill name matches
                if skill_data["name"] != skill_name:
                    raise ValueError(f"Skill name mismatch: expected {skill_name}, got {skill_data['name']}")
                
                return self.skill_manager.improve_skill(
                    name=skill_data["name"],
                    usage=skill_data["usage"],
                    code=skill_data["code"],
                    requirements=skill_data["requirements"]
                )
                
            except Exception as e:
                self.logger.error(f"Attempt {retry_count + 1} failed: {str(e)}")
                error_context.append(str(e))
                retry_count += 1
                
                if retry_count >= max_retries:
                    self.logger.error(f"Failed to improve skill after {max_retries} attempts")
                    raise ValueError(f"Failed to improve skill after {max_retries} attempts. Errors: {error_context}")
        
    def should_ask_user(self) -> bool:
        """Check if we should ask user for code execution confirmation"""
        if not self.messages:
            return False
        last_message = self.messages[-1]
        return bool(last_message.get("tools"))
    
    def get_follow_up(self, results: List[str]) -> str:
        """Generate follow-up message based on execution results"""
        return f"""I executed that code. This was the output::\n{chr(10).join(results)}\nWhat does this output mean (I can't understand it, please help) / what code needs to be run next (if anything, or are we done)? I can't replace any placeholders. In case we're done, let make sure your previous answer didn't have any mistake, if wer're done do not place code inside your answer"""
    
    def should_stop_follow_up(self, loop_count: int, max_loops: int = 5) -> bool:
        """Check if we should stop follow-up iterations"""
        return loop_count >= max_loops
