from code2image import Code2Image
from typing import Optional

class CodeToImage:
    """Converts code snippets to images with syntax highlighting"""
    
    def __init__(self):
        self.converter = Code2Image(
            padding=20,
            background_color="#2C2C2C",
            font_size=16,
            line_numbers=False
        )

    def convert(self, code: str, language: Optional[str] = None) -> bytes:
        """
        Convert code snippet to image
        
        Args:
            code: The code snippet to convert
            language: Optional programming language for syntax highlighting
            
        Returns:
            Image bytes in PNG format
        """
        if language is None:
            language = 'text'
            
        return self.converter.generate_image(code, language=language)
