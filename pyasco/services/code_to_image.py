from PIL import Image, ImageDraw, ImageFont
import io
from typing import Optional
import os

class CodeToImage:
    """Converts code snippets to images with syntax highlighting"""
    
    def __init__(self):
        # Default font settings
        font_path = os.path.join(os.path.dirname(__file__), "../../assets/fonts/JetBrainsMono-Regular.ttf")
        self.font_size = 14
        try:
            self.font = ImageFont.truetype(font_path, self.font_size)
        except:
            # Fallback to default font if custom font not found
            self.font = ImageFont.load_default()
            
        # Style settings
        self.padding = 20
        self.line_height = int(self.font_size * 1.5)
        self.bg_color = (40, 44, 52)  # Dark background
        self.text_color = (171, 178, 191)  # Light gray text

    def convert(self, code: str, language: Optional[str] = None) -> bytes:
        """
        Convert code snippet to image
        
        Args:
            code: The code snippet to convert
            language: Optional programming language for syntax highlighting
            
        Returns:
            Image bytes in PNG format
        """
        # Split code into lines
        lines = code.split('\n')
        
        # Calculate image dimensions
        max_line_width = max(self.font.getlength(line) for line in lines)
        width = int(max_line_width + 2 * self.padding)
        height = len(lines) * self.line_height + 2 * self.padding
        
        # Create image
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw code
        y = self.padding
        for line in lines:
            draw.text(
                (self.padding, y),
                line,
                font=self.font,
                fill=self.text_color
            )
            y += self.line_height
            
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
