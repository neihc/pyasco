import time
import queue
import docker
import jupyter_client
import os
import json
from typing import Optional, Tuple, Dict

class CodeExecutor:
    """A class to execute Python and Bash code using Jupyter kernel or Docker"""
    
    def __init__(self, python_version: str = 'python3', use_docker: bool = False, 
                 docker_image: str = 'python:3.11-slim', 
                 docker_options: Optional[Dict] = None,
                 bash_shell: str = '/bin/bash',
                 python_command: str = 'python',
                 env_file: Optional[str] = None):
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
            
            # Try to use saved state image if it exists
            saved_image = f"{self.docker_image.split(':')[0]}:latest_state"
            try:
                self.docker_client.images.get(saved_image)
                container_image = saved_image
            except docker.errors.ImageNotFound:
                container_image = self.docker_image

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
            
            # Add specific Docker options we support
            if 'mem_limit' in self.docker_options:
                container_options['mem_limit'] = self.docker_options['mem_limit']
            if 'cpu_count' in self.docker_options:
                container_options['cpu_count'] = self.docker_options['cpu_count']
            if 'volumes' in self.docker_options:
                container_options['volumes'] = self.docker_options['volumes']

            # Create persistent container
            self.container = self.docker_client.containers.run(
                container_image,
                **container_options
            )
            
            # Check if jupyter console is installed, if not install it
            exit_code, output = self.container.exec_run(['which', 'jupyter-console'])
            if exit_code != 0:
                print("Setting up Jupyter console in container...")
                setup_cmd = """
                pip install jupyter-console ipykernel > /dev/null 2>&1
                mkdir -p /root/.ipython/profile_default/
                """
                self.container.exec_run(['bash', '-c', setup_cmd])
                print("Jupyter console setup completed")
            
            # Create directories for communication
            self.container.exec_run(['mkdir', '-p', '/tmp/pyasco'])
            
            # Start persistent Python process
            python_server = """
import sys, os, time, json
from threading import Lock

# Global namespace for code execution
global_ns = {}
file_lock = Lock()

def execute_code(code):
    try:
        # Capture output
        from io import StringIO
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout = StringIO()
        stderr = StringIO()
        sys.stdout, sys.stderr = stdout, stderr
        
        try:
            # Execute in the global namespace
            exec(code, global_ns)
            return stdout.getvalue(), stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    except Exception as e:
        import traceback
        return None, traceback.format_exc()

# Main loop
while True:
    try:
        with file_lock:
            if os.path.exists('/tmp/pyasco/input.py'):
                with open('/tmp/pyasco/input.py', 'r') as f:
                    code = f.read()
                os.remove('/tmp/pyasco/input.py')
                
                stdout, stderr = execute_code(code)
                
                with open('/tmp/pyasco/output.json', 'w') as f:
                    json.dump({
                        'stdout': stdout,
                        'stderr': stderr
                    }, f)
                
                # Signal completion
                with open('/tmp/pyasco/done', 'w') as f:
                    f.write('1')
    except Exception as e:
        # Log any errors
        with open('/tmp/pyasco/error.log', 'a') as f:
            f.write(f"{str(e)}\\n")
    
    time.sleep(0.1)
"""
            # Write server code to container
            self.container.exec_run(['bash', '-c', f'cat > /tmp/server.py << EOL\n{python_server}\nEOL'])
            
            # Start Python server process
            self.container.exec_run(['python', '/tmp/server.py'], detach=True)
            
        if not self.use_docker:
            # Initialize kernel manager for local execution
            self.km = jupyter_client.KernelManager(kernel_name=self.python_version)
            self.km.start_kernel()
            self.kc = self.km.client()
            self.kc.start_channels()
            # Wait for kernel to be ready
            self.kc.wait_for_ready()

    def reset(self):
        """Reset the current kernel"""
        # Properly cleanup old client and kernel
        if hasattr(self, 'kc'):
            try:
                self.kc.stop_channels()
            except:
                pass

        if hasattr(self, 'km'):
            try:
                self.km.shutdown_kernel(now=True)
            except:
                pass

        # Create fresh kernel manager and client
        self.km = jupyter_client.KernelManager(kernel_name=self.python_version)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready()

    def execute(self, code: str, language: str = 'python') -> Tuple[Optional[str], Optional[str]]:
        """
        Execute the given code and return stdout and stderr
        
        Args:
            code: The code to execute
            language: The language to execute ('python' or 'bash')
            
        Returns:
            Tuple of (stdout, stderr)
        """
        if language.lower() == 'bash':
            if self.use_docker:
                return self._execute_bash_in_docker(code)
            else:
                return self._execute_bash_local(code)
        else:
            if self.use_docker:
                return self._execute_in_docker(code)
            else:
                return self._execute_local(code)
            
    def _execute_local(self, code: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute code in local Jupyter kernel"""
        msg_id = self.kc.execute(code)
        
        # Collect the outputs
        stdout = []
        stderr = []
        
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=1)
                msg_type = msg['header']['msg_type']
                content = msg['content']
                
                if msg_type == 'stream':
                    if content['name'] == 'stdout':
                        stdout.append(content['text'])
                    elif content['name'] == 'stderr':
                        stderr.append(content['text'])
                        
                elif msg_type == 'execute_result':
                    stdout.append(str(content['data'].get('text/plain', '')))
                    
                elif msg_type == 'error':
                    stderr.append('\n'.join(content['traceback']))
                    
            except queue.Empty:
                break

                
        return (''.join(stdout) if stdout else None, 
                ''.join(stderr) if stderr else None)

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
        """Execute code inside Docker container using persistent Python process"""
        try:
            # Clean up any old files
            self.container.exec_run(['rm', '-f', '/tmp/pyasco/output.json', '/tmp/pyasco/done'])
            
            # Write code to input file
            cmd = f'cat > /tmp/pyasco/input.py << EOL\n{code}\nEOL'
            self.container.exec_run(['bash', '-c', cmd])
            
            # Wait for execution to complete
            max_retries = 30  # 3 seconds
            for _ in range(max_retries):
                exit_code, _ = self.container.exec_run(['test', '-f', '/tmp/pyasco/done'])
                if exit_code == 0:
                    break
                time.sleep(0.1)
            else:
                return None, "Execution timeout"
            
            # Read results
            exit_code, output = self.container.exec_run(
                ['cat', '/tmp/pyasco/output.json']
            )
            
            if exit_code != 0:
                return None, "Failed to read output"
                
            result = json.loads(output.decode('utf-8'))
            return result['stdout'], result['stderr']
            
        except Exception as e:
            return None, str(e)
                    
    def cleanup(self):
        """Cleanup all resources properly"""
        # Clean up Jupyter kernel resources
        if not self.use_docker:
            if hasattr(self, 'kc'):
                try:
                    self.kc.stop_channels()
                    self.kc = None
                except:
                    pass
                
            if hasattr(self, 'km'):
                try:
                    self.km.shutdown_kernel(now=True)
                    self.km = None
                except:
                    pass
        
        # Clean up Docker resources
        if self.use_docker and hasattr(self, 'container'):
            try:
                # Kill the Python server process
                self.container.exec_run(
                    ["pkill", "-f", "python /tmp/server.py"]
                )
                
                # Stop the container
                self.container.stop(timeout=2)
                
                # Save container state
                self.container.commit(
                    repository=self.docker_image.split(':')[0],
                    tag='latest_state'
                )
                
                # Remove container
                self.container.remove(force=True)
                self.container = None
            except:
                pass
