import platform
import os
import psutil

def get_system_info(use_docker: bool = False, docker_image: str = None) -> str:
    """Get current system information including environment variables"""
    if use_docker:
        # Execute code inside Docker to get container system info
        docker_code = """
import platform
import os

env_vars = '\\n'.join(f'- {k}={v}' for k, v in sorted(os.environ.items()))
print(f'''System Information:
- OS: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- CPU Architecture: {platform.machine()}
- Execution Environment: Docker container using {os.environ.get('HOSTNAME', 'unknown')}

Environment Variables:
{env_vars}''')
"""
        # Execute the code in Docker and capture output
        from ..tools.code_execute import CodeExecutor
        executor = CodeExecutor(use_docker=True, docker_image=docker_image)
        output, error = executor.execute(docker_code)
        executor.cleanup()
        return output if output else "Failed to get Docker container system info"
    else:
        # Get host system info
        memory = psutil.virtual_memory()
        env_vars = '\n'.join(f'- {k}' for k, v in sorted(os.environ.items()))
        return f"""System Information:
- OS: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- CPU Architecture: {platform.machine()}
- Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available

Environment Variables:
{env_vars}"""
