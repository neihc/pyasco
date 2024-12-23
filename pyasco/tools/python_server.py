import sys
import os
import time
import json
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
                
                # Ensure proper JSON encoding of special characters
                with open('/tmp/pyasco/output.json', 'w', encoding='utf-8') as f:
                    json.dump({
                        'stdout': stdout,
                        'stderr': stderr
                    }, f, ensure_ascii=False)
                
                # Signal completion
                with open('/tmp/pyasco/done', 'w') as f:
                    f.write('1')
    except Exception as e:
        # Log any errors
        with open('/tmp/pyasco/error.log', 'a') as f:
            f.write(f"{str(e)}\n")
    
    time.sleep(0.1)
