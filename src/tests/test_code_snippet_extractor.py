import pytest
from ..services.code_snippet_extractor import CodeSnippetExtractor, CodeSnippet

@pytest.fixture
def extractor():
    return CodeSnippetExtractor()

def test_extract_single_code_block_with_language(extractor):
    markdown = '''
Some text here
```python
def hello():
    print("Hello")
```
More text
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
