import time
import queue
import jupyter_client
import os
import subprocess
from typing import Optional, Tuple
from ..logger_config import setup_logger

class CodeExecutor:
    """A class to execute Python and Bash code using Jupyter kernel"""
    
    def __init__(self, python_version: str = 'python3', bash_shell: str = '/bin/bash'):
        self.logger = setup_logger('code_executor')
        """
        Initialize a new Jupyter kernel connection
        
        Args:
            python_version: The Python kernel to use (e.g. 'python3', 'python2', etc.)
            bash_shell: Path to bash shell executable
        """
        self.python_version = python_version
        self.bash_shell = bash_shell
        
        # Initialize kernel manager for local execution
        self.km = jupyter_client.KernelManager(kernel_name=self.python_version)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        # Wait for kernel to be ready
        self.kc.wait_for_ready()

    def reset(self):
        """Reset the current kernel"""
        self.cleanup()
        # Create fresh kernel manager and client
        self.km = jupyter_client.KernelManager(kernel_name=self.python_version)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready()

    def execute(self, code: str, language: str = 'python') -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Execute the given code and return stdout, stderr, and the latest value
        
        Args:
            code: The code to execute
            language: The language to execute ('python' or 'bash')
            
        Returns:
            Tuple of (stdout, stderr, latest_value)
        """
        if language.lower() == 'bash':
            result = self._execute_bash_local(code)
            return result[0], result[1], None  # No latest value for bash
        else:
            return self._execute_local(code)
            
    def _execute_local(self, code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Execute code using Jupyter kernel and capture latest value"""
        try:
            # Execute code
            msg_id = self.kc.execute(code)
            
            # Collect outputs
            stdout_parts = []
            stderr_parts = []
            latest_value = None
            
            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=1000)
                    msg_type = msg['msg_type']
                    content = msg['content']
                    
                    if msg_type == 'stream':
                        if content['name'] == 'stdout':
                            stdout_parts.append(content['text'])
                        elif content['name'] == 'stderr':
                            stderr_parts.append(content['text'])
                    elif msg_type == 'error':
                        stderr_parts.extend([
                            '\n'.join(content['traceback']),
                            f"{content['ename']}: {content['evalue']}"
                        ])
                    elif msg_type == 'execute_result':
                        # This captures the value of the last expression
                        if 'data' in content and 'text/plain' in content['data']:
                            latest_value = content['data']['text/plain']
                    elif msg_type == 'status' and content['execution_state'] == 'idle':
                        break
                        
                except queue.Empty:
                    break
            
            stdout = ''.join(stdout_parts) if stdout_parts else None
            stderr = ''.join(stderr_parts) if stderr_parts else None
            
            return stdout, stderr, latest_value
            
        except Exception as e:
            return None, str(e)

    def _execute_bash_local(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute bash code locally"""
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh') as f:
            f.write(code)
            f.flush()
            
            try:
                process = subprocess.Popen(
                    [self.bash_shell, f.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                return (stdout if stdout else None,
                        stderr if stderr else None)
            except Exception as e:
                return None, str(e)

    def cleanup(self):
        """Cleanup all resources properly"""
        if hasattr(self, 'kc'):
            try:
                self.kc.stop_channels()
                self.kc = None
            except Exception as e:
                print(f"Error stopping kernel channels: {str(e)}")
        
        if hasattr(self, 'km'):
            try:
                self.km.shutdown_kernel(now=True)
                self.km = None
            except Exception as e:
                print(f"Error shutting down kernel: {str(e)}")
