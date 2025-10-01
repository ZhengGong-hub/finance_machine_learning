"""
Centralized logging configuration for the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = None) -> logging.Logger:
    """
    Set up and configure a logger with consistent formatting and output.
    
    Parameters
    ----------
    name : str, optional
        Name of the logger. If None, returns the root logger.
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added before
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler - daily rotating log file
        log_file = log_dir / f'factorlab_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Create default logger instance
logger = setup_logger('factorlab') 