from typing import List, Dict, Optional, Tuple
import reprlib
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
                    
                if stdout or stderr or latest_value:
                    results.append(f"Output:\n{stdout or ''}")
                    if stderr:
                        results.append(f"Errors:\n{stderr}")
                    if latest_value:
                        results.append(f"Value: {latest_value}")
                        
        return results
        
    def compress_results(self, results: List[str]) -> Tuple[str, bool]:
        """
        Compress large output results to a more manageable size
        
        Args:
            results: List of string results from tool execution
            
        Returns:
            Tuple containing:
                - The compressed output as a single string
                - Boolean indicating if any compression was applied
        """
        compact_results = []
        was_compressed = False
        
        for result in results:
            if len(result) > 1000:
                compact_results.append(self.repr.repr(result))
                was_compressed = True
            else:
                compact_results.append(result)
        
        output = chr(10).join(compact_results)
        
        # Add compression notice if any output was compressed
        if was_compressed:
            output += "\n\n(This output was compressed due to its large size. Only essential parts are shown. If you need details please change your code to access direcly to the part you want)"
                
        return output, was_compressed
