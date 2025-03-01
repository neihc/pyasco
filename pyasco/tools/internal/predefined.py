"""Predefined functions and utilities for the code executor environment"""

import os
import sys
import json
import base64
from typing import Any, Dict, List

def load_json( str) -> Dict:
    """Load JSON data safely"""
    return json.loads(data)

def save_json( Any) -> str:
    """Save data as JSON string"""
    return json.dumps(data, indent=2)

def encode_base64( str) -> str:
    """Encode string as base64"""
    return base64.b64encode(data.encode()).decode()

def decode_base64( str) -> str:
    """Decode base64 string"""
    return base64.b64decode(data.encode()).decode()

# Add more predefined functions as needed
