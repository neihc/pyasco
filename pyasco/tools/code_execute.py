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
        """Execute code in __main__ context"""
        import tempfile
        import subprocess
        import json
        import os

        runner_path = os.path.join(os.path.dirname(__file__), 'code_runner.py')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
            f.write(code)
            f.flush()
            
            try:
                result = subprocess.run(
                    [self.python_command, runner_path, f.name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    output = json.loads(result.stdout)
                    return (output.get('stdout'), output.get('stderr'))
                else:
                    return None, result.stderr
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
        """Execute code inside Docker container using persistent Python process"""
        try:
            # Generate unique execution ID
            exec_id = str(time.time())
            
            # Clean up old files and verify cleanup
            cleanup_result = self.container.exec_run(
                ['rm', '-f', '/tmp/pyasco/output.json', '/tmp/pyasco/done', '/tmp/pyasco/exec_id'])
            if cleanup_result.exit_code != 0:
                return None, "Failed to cleanup previous execution files"

            # Write execution ID and code
            write_result = self.container.exec_run(['bash', '-c', 
                f'echo "{exec_id}" > /tmp/pyasco/exec_id && '
                f'cat > /tmp/pyasco/input.py << EOL\n{code}\nEOL'])
            if write_result.exit_code != 0:
                return None, "Failed to write input files"
            
            # Wait for execution
            for _ in range(1200):
                if self.container.exec_run(['test', '-f', '/tmp/pyasco/done']).exit_code == 0:
                    # Verify it's our execution
                    id_check = self.container.exec_run(['cat', '/tmp/pyasco/exec_id'])
                    if id_check.exit_code == 0 and id_check.output.decode('utf-8').strip() == exec_id:
                        break
                time.sleep(0.1)
            else:
                return None, "Execution timeout"
            
            output = self.container.exec_run(['cat', '/tmp/pyasco/output.json'])
            if output.exit_code != 0:
                return None, "Failed to read output"
                
            result = json.loads(output.output.decode('utf-8'))
            return result['stdout'], result['stderr']
        except Exception as e:
            return None, str(e)
                    
    def _start_container(self):
        """Initialize and start a new Docker container with all required setup"""
        # Try to use saved state image if it exists
        container_image = self.docker_image
        try:
            saved_image = f"{self.docker_image.split(':')[0]}:latest_state"
            self.docker_client.images.get(saved_image)
            container_image = saved_image
        except docker.errors.ImageNotFound:
            pass

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
        
        # Add specific Docker options
        if 'mem_limit' in self.docker_options:
            container_options['mem_limit'] = self.docker_options['mem_limit']
        if 'cpu_count' in self.docker_options:
            container_options['cpu_count'] = self.docker_options['cpu_count']
        if 'volumes' in self.docker_options:
            container_options['volumes'] = self.docker_options['volumes']

        # Create container
        self.container = self.docker_client.containers.run(
            container_image,
            **container_options
        )
        
        # Install required packages
        print("Setting up IPython kernel in container...")
        setup_cmd = """
        pip install jupyter_client ipykernel > /dev/null 2>&1
        python -m ipykernel install --user
        """
        self.container.exec_run(['bash', '-c', setup_cmd])
        print("IPython kernel setup completed")
        
        # Create directories for communication
        self.container.exec_run(['mkdir', '-p', '/tmp/pyasco'])
        
        # Start Python server
        server_path = os.path.join(os.path.dirname(__file__), 'python_server.py')
        with open(server_path, 'r') as f:
            server_code = f.read()
        self.container.exec_run(['bash', '-c', f'cat > /tmp/server.py << EOL\n{server_code}\nEOL'])
        self.container.exec_run(['python', '/tmp/server.py'], detach=True)

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
                print("Starting Docker cleanup...")
                
                # Check if container is still running
                container_info = self.container.attrs
                print(f"Container status: {container_info['State']['Status']}")
                
                # Kill the Python server process
                try:
                    kill_result = self.container.exec_run(
                        ["pkill", "-f", "python /tmp/server.py"]
                    )
                    print(f"Kill server result: {kill_result.output.decode()}")
                except Exception as e:
                    print(f"Error killing Python server: {str(e)}")
                
                # Save container state with detailed logging
                try:
                    print("Attempting to save container state...")
                    # List installed packages before commit
                    pip_list = self.container.exec_run("pip list")
                    print(f"Installed packages before commit:\n{pip_list.output.decode()}")
                    
                    # Remove old state image if it exists
                    try:
                        old_image = self.docker_client.images.get(f"{self.docker_image.split(':')[0]}:latest_state")
                        self.docker_client.images.remove(old_image.id, force=True)
                        print("Removed old state image")
                    except docker.errors.ImageNotFound:
                        pass

                    # Commit new state
                    commit_result = self.container.commit(
                        repository=self.docker_image.split(':')[0],
                        tag='latest_state',
                        conf={
                            'Cmd': ['tail', '-f', '/dev/null'],
                            'WorkingDir': '/',
                            'Entrypoint': None
                        }
                    )
                    print(f"Container state saved successfully. New image ID: {commit_result.id}")
                    
                    # Verify the save worked
                    try:
                        saved_image = self.docker_client.images.get(f"{self.docker_image.split(':')[0]}:latest_state")
                        print(f"Verified saved image exists: {saved_image.id}")
                    except Exception as e:
                        print(f"Error verifying saved image: {str(e)}")
                        
                except Exception as e:
                    print(f"Failed to save container state: {str(e)}")

                # Stop the container
                try:
                    print("Stopping container...")
                    self.container.stop(timeout=2)
                    print("Container stopped successfully")
                except Exception as e:
                    print(f"Error stopping container: {str(e)}")
                
                # Remove container
                try:
                    print("Removing container...")
                    self.container.remove(force=True)
                    print("Container removed successfully")
                    self.container = None
                except Exception as e:
                    print(f"Error removing container: {str(e)}")
                    
            except Exception as e:
                print(f"Unexpected error during Docker cleanup: {str(e)}")
