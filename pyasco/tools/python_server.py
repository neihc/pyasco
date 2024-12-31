import os
import json
import time
from pathlib import Path
import traceback
import sys
from jupyter_client import KernelManager
import queue

def run_kernel_server():
    """Monitor for input files and execute code using a persistent Jupyter kernel"""
    base_dir = Path("/tmp/pyasco")
    input_file = base_dir / "input.py"
    output_file = base_dir / "output.json"
    done_file = base_dir / "done"
    error_log = base_dir / "error.log"
    
    # Initialize kernel
    km = KernelManager(kernel_name='python3')
    km.start_kernel()
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready()
    
    print("Python kernel server started and monitoring for input files...")
    sys.stdout.flush()
    
    while True:
        if input_file.exists():
            try:
                code = input_file.read_text()
                with open(error_log, 'a') as f:
                    f.write(f"\nExecuting code:\n{code}\n")
                
                # Execute code using kernel
                msg_id = kc.execute(code)
                
                # Collect outputs
                stdout_content = []
                stderr_content = []
                
                while True:
                    try:
                        msg = kc.get_iopub_msg(timeout=10)
                        msg_type = msg['msg_type']
                        content = msg['content']
                        
                        if msg_type == 'stream':
                            if content['name'] == 'stdout':
                                stdout_content.append(content['text'])
                            elif content['name'] == 'stderr':
                                stderr_content.append(content['text'])
                        elif msg_type == 'error':
                            stderr_content.extend([
                                '\n'.join(content['traceback']),
                                f"{content['ename']}: {content['evalue']}"
                            ])
                        elif msg_type == 'status' and content['execution_state'] == 'idle':
                            break
                            
                    except queue.Empty:
                        stderr_content.append("Execution timed out")
                        break
                
                # Get output values
                stdout_value = ''.join(stdout_content)
                stderr_value = ''.join(stderr_content)
                
                # Log captured output for debugging
                with open(error_log, 'a') as f:
                    f.write(f"Captured stdout: {stdout_value}\n")
                    f.write(f"Captured stderr: {stderr_value}\n")
                
                # Write output
                output = {
                    "stdout": stdout_value if stdout_value else None,
                    "stderr": stderr_value if stderr_value else None
                }
                
                output_file.write_text(json.dumps(output, ensure_ascii=False))
                
            except Exception as e:
                error_msg = f"Server error: {str(e)}\n{traceback.format_exc()}"
                with open(error_log, 'a') as f:
                    f.write(error_msg)
                output = {
                    "stdout": None,
                    "stderr": error_msg
                }
                output_file.write_text(json.dumps(output, ensure_ascii=False))
            
            finally:
                # Cleanup
                input_file.unlink(missing_ok=True)
                done_file.touch()
                
        time.sleep(0.1)

if __name__ == "__main__":
    run_kernel_server()
