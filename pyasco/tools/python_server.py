import os
import json
import time
from pathlib import Path
import io
import contextlib
import traceback
import sys

def run_code_in_main():
    """Monitor for input files and execute code in __main__ context"""
    base_dir = Path("/tmp/pyasco")
    input_file = base_dir / "input.py"
    output_file = base_dir / "output.json"
    done_file = base_dir / "done"
    error_log = base_dir / "error.log"
    
    print("Python server started and monitoring for input files...")
    sys.stdout.flush()
    
    while True:
        if input_file.exists():
            try:
                # Create a new module to run the code
                module = types.ModuleType("__main__")
                module.__file__ = "__main__.py"
                sys.modules["__main__"] = module
                
                # Capture output
                stdout = io.StringIO()
                stderr = io.StringIO()
                
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        code = input_file.read_text()
                        with open(error_log, 'a') as f:
                            f.write(f"\nExecuting code:\n{code}\n")
                        
                        exec(code, module.__dict__)
                        
                    except Exception as e:
                        with open(error_log, 'a') as f:
                            f.write(f"Exception occurred: {e}\n")
                        stderr.write(traceback.format_exc())
                
                # Get output values
                stdout_value = stdout.getvalue()
                stderr_value = stderr.getvalue()
                
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
    import types  # Import here to avoid potential circular imports
    run_code_in_main()
