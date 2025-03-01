import pytest
from ..services.code_snippet_extractor import CodeSnippetExtractor, CodeSnippet

@pytest.fixture
def extractor():
    return CodeSnippetExtractor()

def test_extract_single_code_block_with_language(extractor):
    markdown = '''
Here’s how you can run the code to get your RabbitMQ stats:

1. **Install the `requests` library**, if you haven’t already:
   ```bash
   pip install requests
   ```

2. **Run the following Python code**, making sure to set the environment variables `RABBITMQ_API_URL`, `RABBITMQ_USER`, and `RABBITMQ_PASS` in your environment:
   ```python
   import os
   import requests
   from requests.auth import HTTPBasicAuth

   # Get environment variables
   RABBITMQ_API_URL = os.getenv('RABBITMQ_API_URL')
   RABBITMQ_USER = os.getenv('RABBITMQ_USER')
   RABBITMQ_PASS = os.getenv('RABBITMQ_PASS')

   if not RABBITMQ_API_URL or not RABBITMQ_USER or not RABBITMQ_PASS:
       print("Error: Required environment variables are not set.")
   else:
       # Construct the API URL for the overview
       api_url = f"{RABBITMQ_API_URL}/api/overview"
       
       # Send GET request with basic auth
       response = requests.get(api_url, auth=HTTPBasicAuth(RABBITMQ_USER, RABBITMQ_PASS))
       
       # Check if the request was successful
       if response.status_code == 200:
           stats = response.json()
           print("RabbitMQ Stats Overview:")
           print(stats)
       else:
           print(f"Error: Failed to fetch stats. Status code: {response.status_code}")
   ```

Make sure to define your environment variables correctly in your terminal/session before running this code. 

If you need any assistance or have questions about specific parts of the code, let me know!
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 1
    assert snippets[0] == CodeSnippet(
        language="python",
        content='def hello():\n    print("Hello")'
    )

def test_extract_code_block_without_language(extractor):
    markdown = '''
```
plain text code block
```
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 1
    assert snippets[0] == CodeSnippet(
        language=None,
        content="plain text code block"
    )

def test_extract_multiple_code_blocks(extractor):
    markdown = '''
```python
x = 1
```
Some text
```javascript
console.log("hi");
```
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 2
    assert snippets[0] == CodeSnippet(
        language="python",
        content="x = 1"
    )
    assert snippets[1] == CodeSnippet(
        language="javascript",
        content='console.log("hi");'
    )

def test_empty_code_block(extractor):
    markdown = '''
```python
```
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 1
    assert snippets[0] == CodeSnippet(
        language="python",
        content=""
    )

def test_no_code_blocks(extractor):
    markdown = "Just some regular markdown text"
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 0

def test_nested_code_blocks(extractor):
    markdown = '''
> Here's some code:
> ```python
> def nested():
>     pass
> ```
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 1
    assert snippets[0] == CodeSnippet(
        language="python",
        content="def nested():\n    pass"
    )

def test_unicode_content(extractor):
    markdown = '''
```python
print("Hello 世界")
```
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 1
    assert snippets[0] == CodeSnippet(
        language="python",
        content='print("Hello 世界")'
    )

def test_multiple_backticks(extractor):
    markdown = '''
````python
def test():
    pass
````
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 1
    assert snippets[0] == CodeSnippet(
        language="python",
        content='def test():\n    pass'
    )

def test_language_with_whitespace(extractor):
    markdown = '''
``` python 
x = 1
```
'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 1
    assert snippets[0] == CodeSnippet(
        language="python",
        content='x = 1'
    )

def test_consecutive_code_blocks(extractor):
    markdown = '''
```python
x = 1
```
```javascript
y = 2
```'''
    snippets = extractor.extract_snippets(markdown)
    assert len(snippets) == 2
    assert snippets[0] == CodeSnippet(language="python", content='x = 1')
    assert snippets[1] == CodeSnippet(language="javascript", content='y = 2')

# Tests for omit_snippets method
def test_omit_single_code_block(extractor):
    markdown = '''
Some text
```python
def hello():
    pass
```
More text
'''
    result = extractor.omit_snippets(markdown)
    assert 'Some text' in result
    assert 'More text' in result
    assert 'def hello()' not in result

def test_omit_multiple_code_blocks(extractor):
    markdown = '''
First text
```python
x = 1
```
Middle text
```javascript
y = 2
```
Last text
'''
    result = extractor.omit_snippets(markdown)
    assert 'First text' in result
    assert 'Middle text' in result
    assert 'Last text' in result
    assert 'x = 1' not in result
    assert 'y = 2' not in result

def test_omit_preserves_markdown(extractor):
    markdown = '''
# Heading
```python
code = 1
```
**Bold text**
'''
    result = extractor.omit_snippets(markdown)
    assert '# Heading' in result
    assert '**Bold text**' in result
    assert 'code = 1' not in result

def test_omit_no_code_blocks(extractor):
    markdown = "Just regular text\n**with formatting**"
    result = extractor.omit_snippets(markdown)
    assert result == markdown.strip()
