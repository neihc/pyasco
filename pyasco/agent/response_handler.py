from typing import List, Dict, Generator, Any, Union
from ..services.llm import get_openai_response
from ..services.code_snippet_extractor import CodeSnippetExtractor
from .base import AgentResponse

class ResponseHandler:
    def __init__(self, code_extractor: CodeSnippetExtractor):
        self.code_extractor = code_extractor

    def handle_response(
        self, 
        messages: List[Dict], 
        model: str, 
        stream: bool = False
    ) -> Union[AgentResponse, Generator[AgentResponse, None, None]]:
        """Handle LLM response and create appropriate AgentResponse"""
        filtered_messages = self._prepare_messages(messages)
        llm_response = get_openai_response(filtered_messages, model=model, stream=stream)

        if stream:
            return self._handle_streaming_response(llm_response, messages)

        response = AgentResponse(content=llm_response)
        response.tools = self._create_tool_response(llm_response)
        message_dict = response.__dict__.copy()
        message_dict["skills"] = []
        messages.append(message_dict)
        return response

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Remove tool and skills information from messages"""
        return [{k: v for k, v in msg.items() if k not in ["tools", "skills"]}
                for msg in messages]

    def _create_tool_response(self, content: str) -> List[Dict]:
        """Create tool response based on code snippets"""
        tools = []
        snippets = self.code_extractor.extract_snippets(content)
        if snippets:
            tools.append({
                "name": "python_executor",
                "parameters": {"snippets": snippets}
            })
        return tools

    def _handle_streaming_response(
        self, 
        llm_response: Generator[Any, None, None],
        messages: List[Dict]
    ) -> Generator[AgentResponse, None, None]:
        """Handle streaming response from LLM"""
        full_content = ""
        
        for chunk in llm_response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                yield AgentResponse(content=content, done=False)
        
        tools = self._create_tool_response(full_content)
        final_response = AgentResponse(content=full_content, tools=tools)
        messages.append(final_response.__dict__)
        yield final_response
