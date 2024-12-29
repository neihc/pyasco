from PIL import Image, ImageDraw, ImageFont
import io
from typing import Optional
import os
from pygments import highlight
from pygments.lexers import PythonLexer, BashLexer, TextLexer
from pygments.token import Token

class CodeToImage:
    """Converts code snippets to images with syntax highlighting"""
    
    def __init__(self):
        # Default font settings
        font_path = os.path.join(os.path.dirname(__file__), "../../assets/fonts/JetBrainsMono-Regular.ttf")
        self.font_size = 16  # Increased base font size
        self.dpi = 300  # High DPI for better quality
        self.scale_factor = 2  # Scale factor for dimensions
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
        
        # Syntax highlighting colors
        self.syntax_colors = {
            Token.Keyword: (198, 120, 221),      # Purple for keywords
            Token.Keyword.Constant: (198, 120, 221),  # Purple for True, False, None
            Token.Keyword.Declaration: (198, 120, 221),  # Purple for def, class
            Token.String: (152, 195, 121),       # Green for strings
            Token.String.Doc: (152, 195, 121),   # Green for docstrings
            Token.Name.Function: (97, 175, 239),    # Blue for functions
            Token.Name.Class: (229, 192, 123),      # Yellow for classes
            Token.Name.Builtin: (229, 192, 123),    # Yellow for built-in functions
            Token.Name.Builtin.Pseudo: (198, 120, 221),  # Purple for self, cls
            Token.Number: (209, 154, 102),       # Orange for numbers
            Token.Number.Float: (209, 154, 102),  # Orange for floats
            Token.Number.Integer: (209, 154, 102),  # Orange for integers
            Token.Comment: (92, 99, 112),        # Gray for comments
            Token.Comment.Single: (92, 99, 112),  # Gray for single-line comments
            Token.Operator: (171, 178, 191),     # Light gray for operators
            Token.Punctuation: (171, 178, 191),  # Light gray for punctuation
            Token.Name.Variable: (171, 178, 191),  # Light gray for variables
        }

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
        
        # Calculate base image dimensions
        max_line_width = max(self.font.getlength(line) for line in lines)
        base_width = int(max_line_width + 2 * self.padding)
        # Adjust height calculation to account for actual text lines
        base_height = (len(lines) - 1) * self.line_height + self.font_size + 2 * self.padding
        
        # Scale dimensions
        width = base_width * self.scale_factor
        height = base_height * self.scale_factor
        
        # Create high-res image
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Get appropriate lexer
        if language == 'python':
            lexer = PythonLexer()
        elif language == 'bash':
            lexer = BashLexer()
        else:
            lexer = TextLexer()
            
        # Tokenize and draw code with syntax highlighting
        y = self.padding
        x = self.padding
        
        for token, text in lexer.get_tokens(code):
            # Try exact token match first, then parent token
            color = self.syntax_colors.get(token, None)
            if color is None:
                color = self.syntax_colors.get(token.parent, self.text_color)
            
            # Handle newlines
            if '\n' in text:
                text_parts = text.split('\n')
                for i, part in enumerate(text_parts):
                    if part:
                        draw.text((x, y), part, font=self.font, fill=color)
                        x += self.font.getlength(part)
                    if i < len(text_parts) - 1:  # Don't add line break after last part
                        y += self.line_height
                        x = self.padding
            else:
                draw.text((x, y), text, font=self.font, fill=color)
                x += self.font.getlength(text)
            
        # Convert to bytes with high DPI
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG', dpi=(self.dpi, self.dpi), optimize=True, quality=95)
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
