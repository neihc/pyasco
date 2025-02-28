from typing import List, Dict, Optional, Tuple, Any
import reprlib
import ast
from ..tools.code_execute import CodeExecutor

class ToolHandler:
    def __init__(self, executor: CodeExecutor):
        self.executor = executor
        self.repr = reprlib.Repr()
        self.repr.maxstring = 1000  # Adjust max string length
        self.repr.maxother = 1000   # Adjust max length for other objects

    def execute_tools(self, tools: List[Dict]) -> List[str]:
        """Execute tools and return results"""
        execution_results = []
        
        for tool in tools:
            if tool["name"] == "python_executor":
                results = self._execute_python_tool(tool["parameters"])
                execution_results.extend(results)
                
        return execution_results

    def _execute_python_tool(self, params: Dict) -> List[str]:
        """Execute Python code snippets"""
        results = []
        for snippet in params.get("snippets", []):
            if snippet.language:
                language = snippet.language.lower()
                stdout = stderr = None
                
                if 'python' in language or 'bash' in language:
                    stdout, stderr, latest_value = self.executor.execute(snippet.content, language)
                    
                    # Apply compression to raw outputs
                    if stdout and len(stdout) > 1000:
                        stdout = self.repr.repr(stdout)
                    
                    if stderr and len(stderr) > 1000:
                        stderr = self.repr.repr(stderr)
                    
                    if latest_value:
                        # Try to parse the latest_value as a Python data structure
                        parsed_value = self._safe_eval(latest_value)
                        
                        # Handle based on the type
                        if isinstance(parsed_value, (dict, list)) and len(str(parsed_value)) > 1000:
                            latest_value = self.repr.repr(parsed_value)
                        elif isinstance(parsed_value, str) and len(parsed_value) > 1000:
                            latest_value = self.repr.repr(parsed_value)
                        elif not isinstance(parsed_value, str):
                            latest_value = str(parsed_value)
                    
                if stdout or stderr or latest_value:
                    results.append(f"Output:\n{stdout or ''}")
                    if stderr:
                        results.append(f"Errors:\n{stderr}")
                    if latest_value:
                        results.append(f"Value: {latest_value}")
                        
        return results
        
    def _safe_eval(self, value: str) -> Any:
        """
        Safely evaluate a string to see if it represents a Python data structure
        
        Args:
            value: String that might represent a Python data structure
            
        Returns:
            The evaluated object if successful, or the original string if not
        """
        try:
            # Only allow literals like lists, dicts, strings, numbers, etc.
            return ast.literal_eval(value.strip())
        except (SyntaxError, ValueError):
            return value

    def compress_results(self, results: List[str]) -> Tuple[str, bool]:
        """
        Join results and add compression notice if needed
        
        Args:
            results: List of string results from tool execution
            
        Returns:
            Tuple containing:
                - The joined output as a single string
                - Boolean indicating if any compression was applied
        """
        # Check if any result contains the repr format which indicates compression
        was_compressed = any('...' in result for result in results)
        
        output = chr(10).join(results)
        
        # Add compression notice if any output was compressed
        if was_compressed:
            output += "\n\n(This output was compressed due to its large size. Only essential parts are shown. If you need details please change your code to access directly the part you want)"
                
        return output, was_compressed
