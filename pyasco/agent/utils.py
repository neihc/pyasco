import platform
import os
import psutil
from datetime import datetime

def get_system_info(executor=None) -> str:
    # Get host system info
    memory = psutil.virtual_memory()
    return f"""System Information:
- Date: {datetime.now().strftime('%Y-%m-%d')}
- OS: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- CPU Architecture: {platform.machine()}
- Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available
"""
