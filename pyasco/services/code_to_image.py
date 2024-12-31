from code2image.cls import Code2Image
from typing import Optional
from io import BytesIO
from PIL import Image

class CodeToImage:
    """Converts code snippets to images with syntax highlighting"""
    
    def __init__(self):
        self.converter = Code2Image(
            font_size=16,
            line_pad=20,  # This is the padding
            code="#2C2C2C",  # Code background color
            line_numbers=False,
            font_family="Courier New"
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
            
        # Generate image using the correct API
        img = self.converter.highlight(code)
        
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
