from typing import List, Dict, Generator, Any, Union
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor
from .types import Message
from .conversation import Conversation

class ResponseHandler:
    def __init__(self, code_extractor: CodeSnippetExtractor, llm_service: LLMService):
        self.code_extractor = code_extractor
        self.llm_service = llm_service

    def handle_response(
        self, 
        messages: List[Dict], 
        model: str,
        conversation: Conversation,
        stream: bool = False
    ) -> Union[Message, Generator[Message, None, None]]:
        """Handle LLM response and create appropriate Message"""
        llm_response = self.llm_service.get_response(messages, model=model, stream=stream)

        if stream:
            return self._handle_streaming_response(llm_response, conversation)

        tools = self._create_tool_response(llm_response)
        return conversation.add_message(
            role="assistant",
            content=llm_response,
            tools=tools
        )

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
        conversation: Conversation
    ) -> Generator[Message, None, None]:
        """Handle streaming response from LLM"""
        full_content = ""
        
        for chunk in llm_response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                yield Message(role="assistant", content=content)
        
        tools = self._create_tool_response(full_content)
        final_message = conversation.add_message(
            role="assistant",
            content=full_content,
            tools=tools
        )
        yield final_message
