import os
import time
import json
import queue
import jupyter_client
from jupyter_client.manager import KernelManager

# Initialize kernel
km = KernelManager()
km.start_kernel()
kc = km.client()
kc.start_channels()
kc.wait_for_ready()

def execute_code(code):
    try:
        # Execute code
        kc.execute(code)
        stdout, stderr = [], []
        
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=1)
                content = msg['content']
                
                if msg['header']['msg_type'] == 'stream':
                    if content['name'] == 'stdout':
                        stdout.append(content['text'])
                    else:
                        stderr.append(content['text'])
                elif msg['header']['msg_type'] == 'execute_result':
                    stdout.append(str(content['data'].get('text/plain', '')))
                elif msg['header']['msg_type'] == 'error':
                    stderr.append('\n'.join(content['traceback']))
            except queue.Empty:
                break
                
        return ''.join(stdout) or None, ''.join(stderr) or None
    except Exception as e:
        import traceback
        return None, traceback.format_exc()

# Main loop
while True:
    try:
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
import os
import json
import time
from pathlib import Path

def run_code_in_main():
    """Monitor for input files and execute code in __main__ context"""
    base_dir = Path("/tmp/pyasco")
    input_file = base_dir / "input.py"
    output_file = base_dir / "output.json"
    done_file = base_dir / "done"
    
    while True:
        if input_file.exists():
            try:
                # Create a new module to run the code
                import types
                import sys
                module = types.ModuleType("__main__")
                module.__file__ = "__main__.py"
                sys.modules["__main__"] = module
                
                # Capture output
                import io
                import contextlib
                stdout = io.StringIO()
                stderr = io.StringIO()
                
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        code = input_file.read_text()
                        exec(code, module.__dict__)
                    except Exception as e:
                        import traceback
                        stderr.write(traceback.format_exc())
                
                # Write output
                output = {
                    "stdout": stdout.getvalue() or None,
                    "stderr": stderr.getvalue() or None
                }
                
                output_file.write_text(json.dumps(output))
                
            except Exception as e:
                output = {
                    "stdout": None,
                    "stderr": str(e)
                }
                output_file.write_text(json.dumps(output))
            
            finally:
                # Cleanup
                input_file.unlink(missing_ok=True)
                done_file.touch()
                
        time.sleep(0.1)

if __name__ == "__main__":
    run_code_in_main()
