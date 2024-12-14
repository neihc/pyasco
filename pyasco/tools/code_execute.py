import time
import queue
import docker
import jupyter_client
import os
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
            
            # Start IPython kernel in container and get connection file
            exit_code, output = self.container.exec_run(['pgrep', '-f', 'ipython.*kernel'])
            if exit_code != 0:
                # Start kernel in background and get connection file
                self.container.exec_run(
                    ['bash', '-c', 'ipython kernel > /tmp/kernel_output.txt 2>&1 &']
                )
                
                # Wait for kernel to start and connection file to be created
                max_retries = 10
                connection_file = None
                for _ in range(max_retries):
                    time.sleep(1)
                    # Read the kernel output file and extract the connection file path
                    _, (stdout, _) = self.container.exec_run(
                        ['bash', '-c', 'ls -1 /root/.local/share/jupyter/runtime/kernel-*.json | head -n 1'],
                        demux=True
                    )
                    if stdout:
                        connection_file = stdout.decode('utf-8').strip()
                    if connection_file:
                        break
                
                if not connection_file:
                    raise RuntimeError("Failed to get kernel connection file after multiple retries")
                
                # Store connection file for later use
                self.kernel_connection_file = connection_file
            
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
        """Execute code inside Docker container using existing IPython kernel"""
        try:
            # Create a temporary file for the code
            temp_file = '/tmp/code_to_run.py'
            cmd = f'cat > {temp_file} << EOL\n{code}\nEOL'
            self.container.exec_run(['bash', '-c', cmd])

            # Modified kernel execution script with better message handling
            run_cmd = """
from jupyter_client import BlockingKernelClient
import json, sys, time

# Load the connection info
with open(r'{}', 'r') as f:
    connection_info = json.load(f)
    
# Create a client and connect to the running kernel
kc = BlockingKernelClient()
kc.load_connection_info(connection_info)
kc.start_channels()

# Execute the code
with open('{}', 'r') as f:
    code = f.read()
    msg_id = kc.execute(code)
    
# Get the results
timeout = 300  # Five minute timeout
start_time = time.time()
stdout_parts = []
stderr_parts = []
execution_done = False

while time.time() - start_time < timeout and not execution_done:
    try:
        msg = kc.get_iopub_msg(timeout=1)
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
            
        msg_type = msg['header']['msg_type']
        content = msg['content']
        
        if msg_type == 'stream':
            if content['name'] == 'stdout':
                stdout_parts.append(content['text'])
                print(content['text'], end='', flush=True)
            elif content['name'] == 'stderr':
                stderr_parts.append(content['text'])
                print(content['text'], file=sys.stderr, end='', flush=True)
        elif msg_type == 'execute_result':
            result = content['data'].get('text/plain', '')
            stdout_parts.append(result)
            print(result, flush=True)
        elif msg_type == 'error':
            error_text = '\\n'.join(content['traceback'])
            stderr_parts.append(error_text)
            print(error_text, file=sys.stderr, flush=True)
        elif msg_type == 'status' and content['execution_state'] == 'idle':
            # Wait a bit more for any remaining messages
            time.sleep(0.5)
            execution_done = True
    except Exception as e:
        print(f"Error in message handling: {{str(e)}}", file=sys.stderr)
        break

# Ensure we capture all remaining messages
while True:
    try:
        msg = kc.get_iopub_msg(timeout=0.5)
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        
        content = msg['content']
        if msg['header']['msg_type'] == 'stream':
            if content['name'] == 'stdout':
                stdout_parts.append(content['text'])
            else:
                stderr_parts.append(content['text'])
    except:
        break

kc.stop_channels()

# Print final outputs for capture
if stdout_parts:
    print(''.join(stdout_parts), flush=True)
if stderr_parts:
    print(''.join(stderr_parts), file=sys.stderr, flush=True)
""".format(self.kernel_connection_file, temp_file)
            
            exit_code, (stdout, stderr) = self.container.exec_run(
                ['python', '-c', run_cmd],
                demux=True
            )

            # Clean up the temporary file
            self.container.exec_run(['rm', temp_file])
            
            stdout_content = stdout.decode('utf-8').strip() if stdout else None
            stderr_content = stderr.decode('utf-8').strip() if stderr else None
            
            return stdout_content, stderr_content
            
        except Exception as e:
            return None, str(e)
                    
    def cleanup(self):
        """Cleanup the kernel, client and Docker resources properly"""
        if hasattr(self, 'kc') and self.kc is not None:
            try:
                self.kc.stop_channels()
            except Exception as e:
                pass  # Ignore errors
            self.kc = None
            
        if hasattr(self, 'km') and self.km is not None:
            try:
                self.km.shutdown_kernel(now=True)
            except Exception as e:
                pass  # Ignore errors
            self.km = None
            
        if hasattr(self, 'container') and self.container is not None:
            try:
                # Stop the container first
                self.container.stop(timeout=5)
                # Commit container state to new image
                self.container.commit(
                    repository=self.docker_image.split(':')[0],
                    tag='latest_state'
                )
                self.container.remove(force=True)
            except Exception as e:
                pass  # Ignore errors
            self.container = None
