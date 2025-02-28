import time
import queue
import docker
import jupyter_client
import os
import json
from typing import Optional, Tuple, Dict
from ..logger_config import setup_logger

class CodeExecutor:
    """A class to execute Python and Bash code using Jupyter kernel or Docker"""
    
    def __init__(self, python_version: str = 'python3', use_docker: bool = False, 
                 docker_image: str = 'python:3.11-slim', 
                 docker_options: Optional[Dict] = None,
                 bash_shell: str = '/bin/bash',
                 python_command: str = 'python',
                 env_file: Optional[str] = None):
        self.logger = setup_logger('code_executor')
        """
        Initialize a new Jupyter kernel connection
        
        Args:
            python_version: The Python kernel to use (e.g. 'python3', 'python2', etc.)
            use_docker: Whether to run code inside a Docker container
            docker_image: Docker image to use if use_docker is True
            docker_options: Additional Docker container options (memory limits, etc.)
            bash_shell: Path to bash shell executable
            env_file: Path to environment file to load into Docker container
        """
        self.python_version = python_version
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.docker_options = docker_options or {}
        self.bash_shell = bash_shell
        self.python_command = python_command
        self.env_file = env_file
        
        if self.env_file and not os.path.exists(self.env_file):
            raise FileNotFoundError(f"Environment file not found: {self.env_file}")
        
        if self.use_docker:
            self.docker_client = docker.from_env()
            # Check if image exists
            try:
                self.docker_client.images.get(docker_image)
            except docker.errors.ImageNotFound:
                raise RuntimeError(f"Docker image not found: {docker_image}")
            
            self._start_container()
            
        if not self.use_docker:
            # Initialize kernel manager for local execution
            self.km = jupyter_client.KernelManager(kernel_name=self.python_version)
            self.km.start_kernel()
            self.kc = self.km.client()
            self.kc.start_channels()
            # Wait for kernel to be ready
            self.kc.wait_for_ready()

    def reset(self):
        """Reset the current kernel or container"""
        self.cleanup()
        if self.use_docker:
            self._start_container()
        else:
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
            if self.use_docker:
                result = self._execute_bash_in_docker(code)
                return result[0], result[1], None  # No latest value for bash
            else:
                result = self._execute_bash_local(code)
                return result[0], result[1], None  # No latest value for bash
        else:
            if self.use_docker:
                result = self._execute_in_docker(code)
                return result[0], result[1], None  # Docker doesn't capture latest value yet
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
                    msg = self.kc.get_iopub_msg(timeout=10)
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

    def _execute_bash_in_docker(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute bash code inside Docker container"""
        try:
            exit_code, (stdout, stderr) = self.container.exec_run(
                [self.bash_shell, '-c', code],
                demux=True
            )
            
            stdout = stdout.decode('utf-8') if stdout else None
            stderr = stderr.decode('utf-8') if stderr else None
            
            if exit_code != 0 and not stderr:
                stderr = f"Exit code: {exit_code}"
                    
            return stdout, stderr
        except Exception as e:
            return None, str(e)

    def _execute_in_docker(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute code inside Docker container using python -c"""
        try:
            self.logger.info("Starting Docker execution")
            self.logger.debug(f"Code to execute:\n{code}")
            
            # Execute code directly using python -c
            exit_code, (stdout, stderr) = self.container.exec_run(
                [self.python_command, '-c', code],
                demux=True
            )
            
            stdout = stdout.decode('utf-8') if stdout else None
            stderr = stderr.decode('utf-8') if stderr else None
            
            if exit_code != 0 and not stderr:
                stderr = f"Exit code: {exit_code}"
                    
            return stdout, stderr
        except Exception as e:
            self.logger.error(f"Docker execution error: {str(e)}", exc_info=True)
            return None, str(e)
                    
    def _start_container(self):
        """Initialize and start a new Docker container"""
        # Load environment variables if env_file exists
        environment = {}
        if self.env_file and os.path.exists(self.env_file):
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        environment[key.strip()] = value.strip()

        # Prepare container options
        container_options = {
            'command': ['tail', '-f', '/dev/null'],  # Keep container running
            'detach': True,
            'environment': environment
        }
        
        # Setup default volumes with expanded home directory path
        workspace_path = os.path.expanduser('~/.pyasco/workspace')
        os.makedirs(workspace_path, exist_ok=True)
        
        volumes = {
            workspace_path: {'bind': '/pyasco', 'mode': 'rw'}
        }
        
        # Add specific Docker options
        if 'mem_limit' in self.docker_options:
            container_options['mem_limit'] = self.docker_options['mem_limit']
        if 'cpu_count' in self.docker_options:
            container_options['cpu_count'] = self.docker_options['cpu_count']
        if 'volumes' in self.docker_options:
            volumes.update(self.docker_options['volumes'])
            
        container_options['volumes'] = volumes

        # Create container
        self.container = self.docker_client.containers.run(
            self.docker_image,
            **container_options
        )

    def cleanup(self):
        """Cleanup all resources properly"""
        # Clean up Jupyter kernel resources
        if not self.use_docker:
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
    
        # Clean up Docker resources
        if self.use_docker and hasattr(self, 'container'):
            try:
                # Reload container state
                self.container.reload()
                
                container_status = self.container.status
                print(f"Container status during cleanup: {container_status}")
                
                if container_status == 'running':
                    try:
                        self.container.commit(
                            repository=self.docker_image.split(':')[0],
                            tag='latest_state'
                        )
                    except Exception as e:
                        print(f"Error committing container state: {str(e)}")
                
                # Stop and remove container regardless of status
                try:
                    self.container.stop(timeout=2)
                except Exception as e:
                    print(f"Error stopping container: {str(e)}")
                    
                try:
                    self.container.remove(force=True)
                except Exception as e:
                    print(f"Error removing container: {str(e)}")
                    
                self.container = None
                    
            except Exception as e:
                print(f"Error during Docker cleanup: {str(e)}")
