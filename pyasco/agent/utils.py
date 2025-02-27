import platform
import os
import psutil
from datetime import datetime

def get_system_info(executor=None) -> str:
    """Get current system information including environment variables"""
    if executor and executor.use_docker:
        # Execute code inside Docker to get container system info
        docker_code = """
import platform
import os

print(f'''System Information:
- Date: {datetime.now().strftime('%Y-%m-%d')}
- OS: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- CPU Architecture: {platform.machine()}
- Execution Environment: Docker container using {os.environ.get('HOSTNAME', 'unknown')}
''')
"""
        output, error = executor.execute(docker_code)
        return output if output else "Failed to get Docker container system info"
    else:
        # Get host system info
        memory = psutil.virtual_memory()
        return f"""System Information:
- Date: {datetime.now().strftime('%Y-%m-%d')}
- OS: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- CPU Architecture: {platform.machine()}
- Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available
"""
