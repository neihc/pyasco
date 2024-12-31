from typing import List, Dict, Optional, Tuple
from ..tools.code_execute import CodeExecutor

class ToolHandler:
    def __init__(self, executor: CodeExecutor):
        self.executor = executor

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
                    stdout, stderr = self.executor.execute(snippet.content, language)
                    
                if stdout or stderr:
                    results.append(f"Output:\n{stdout or ''}")
                    if stderr:
                        results.append(f"Errors:\n{stderr}")
                        
        return results
