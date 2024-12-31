from dataclasses import dataclass
from typing import List, Optional
from tree_sitter_languages import get_language, get_parser

@dataclass
class CodeSnippet:
    """Represents a code snippet extracted from markdown"""
    language: Optional[str]
    content: str

class CodeSnippetExtractor:
    """Extracts code snippets from markdown text using tree-sitter"""
    
    def __init__(self):
        # Initialize tree-sitter parser for markdown
        self.language = get_language('markdown')
        self.parser = get_parser('markdown')

    def extract_snippets(self, markdown_text: str) -> List[CodeSnippet]:
        """
        Extract code snippets from markdown text
        
        Args:
            markdown_text: The markdown text to parse
            
        Returns:
            List of CodeSnippet objects containing the language and content
        """
        tree = self.parser.parse(bytes(markdown_text, "utf8"))
        snippets = []
        
        # Traverse the syntax tree to find fenced code blocks
        cursor = tree.walk()
        
        reached_root = False
        while not reached_root:
            if cursor.node.type == "fenced_code_block":
                # Get the language (if specified)
                info_string = None
                content = ''
                for child in cursor.node.children:
                    if child.type == "info_string":
                        info_string = markdown_text[child.start_byte:child.end_byte]
                        break
                
                # Get the content
                for child in cursor.node.children:
                    if child.type == "code_fence_content":
                        content = child.text.decode()
                        break
            
                snippets.append(CodeSnippet(
                    language=info_string,
                    content=content.strip()
                ))
            if not cursor.goto_first_child():
                while not cursor.goto_next_sibling():
                    if not cursor.goto_parent():
                        reached_root = True
                        break
                        
        return snippets

    def omit_snippets(self, markdown_text: str) -> str:
        """
        Remove all code snippets from markdown text, leaving only the text content
        
        Args:
            markdown_text: The markdown text to process
            
        Returns:
            The markdown text with all code blocks removed
        """
        tree = self.parser.parse(bytes(markdown_text, "utf8"))
        result = markdown_text
        
        # Collect all code block ranges to remove
        ranges_to_remove = []
        cursor = tree.walk()
        
        reached_root = False
        while not reached_root:
            if cursor.node.type == "fenced_code_block":
                ranges_to_remove.append((cursor.node.start_byte, cursor.node.end_byte))
            
            if not cursor.goto_first_child():
                while not cursor.goto_next_sibling():
                    if not cursor.goto_parent():
                        reached_root = True
                        break
        
        # Remove code blocks from end to start to maintain correct positions
        for start, end in sorted(ranges_to_remove, reverse=True):
            result = result[:start] + result[end:]
            
        return result.strip()
        """
        Extract code snippets from markdown text
        
        Args:
            markdown_text: The markdown text to parse
            
        Returns:
            List of CodeSnippet objects containing the language and content
        """
        tree = self.parser.parse(bytes(markdown_text, "utf8"))
        snippets = []
        
        # Traverse the syntax tree to find fenced code blocks
        cursor = tree.walk()
        
        reached_root = False
        while not reached_root:
            if cursor.node.type == "fenced_code_block":
                # Get the language (if specified)
                info_string = None
                content = ''
                for child in cursor.node.children:
                    if child.type == "info_string":
                        info_string = markdown_text[child.start_byte:child.end_byte]
                        break
                
                # Get the content
                for child in cursor.node.children:
                    if child.type == "code_fence_content":
                        content = child.text.decode()
                        break
            
                snippets.append(CodeSnippet(
                    language=info_string,
                    content=content.strip()
                ))
            if not cursor.goto_first_child():
                while not cursor.goto_next_sibling():
                    if not cursor.goto_parent():
                        reached_root = True
                        break
                        
        return snippets
