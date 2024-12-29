import pytest
from ..tools.code_execute import CodeExecutor

@pytest.fixture
def executor():
    """Fixture to create and cleanup PythonExecutor instance"""
    exec = CodeExecutor()
    yield exec
    # Cleanup will happen automatically in __del__

def test_simple_execution(executor):
    """Test basic code execution"""
    stdout, stderr = executor.execute('print("Hello, World!")')
    assert stdout == "Hello, World!\n"
    assert stderr is None

def test_execution_with_error(executor):
    """Test execution of code that raises an error"""
    stdout, stderr = executor.execute('1/0')
    assert stdout is None
    assert "ZeroDivisionError" in stderr

def test_execution_with_result(executor):
    """Test execution that returns a value"""
    stdout, stderr = executor.execute('2 + 2')
    assert stdout == "4"
    assert stderr is None

def test_kernel_reset(executor):
    """Test kernel reset functionality"""
    # Define a variable
    executor.execute('x = 42')
    stdout, _ = executor.execute('print(x)')
    assert stdout == "42\n"
    
    # Reset kernel
    executor.reset()
    
    # Variable should no longer exist
    stdout, stderr = executor.execute('print(x)')
    assert stdout is None
    assert "NameError" in stderr

@pytest.mark.docker
def test_docker_kernel_state():
    """Test that Docker IPython kernel maintains state between executions"""
    executor = CodeExecutor(use_docker=True)
    try:
        # Define a variable
        stdout, stderr = executor.execute('x = 42')
        assert stderr == ''
        
        # Variable should persist in next execution
        stdout, stderr = executor.execute('print(x)')
        assert stdout == "42\n"
        assert stderr == ''
        
        # Complex state should also persist
        executor.execute('import math')
        executor.execute('y = math.pi')
        stdout, stderr = executor.execute('print(y)')
        assert stdout.startswith('3.14')
        assert stderr == ''
    finally:
        executor.cleanup()
