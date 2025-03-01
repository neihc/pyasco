import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name, log_file='agent.log', level=logging.INFO, verbose=True):
    """Set up logger with file and console handlers"""
    
    # Create ~/.pyasco/logs directory if it doesn't exist
    log_dir = Path.home() / '.pyasco' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = log_dir / log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else level)
    
    # Prevent adding handlers multiple times
    if not logger.handlers:
        # Create file handler for verbose logging
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG if verbose else level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    
    return logger
