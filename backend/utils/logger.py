"""
Logging Configuration for PartSelect RAG Pipeline
Provides structured logging with different levels and formatting.
"""

import logging
import sys
from datetime import datetime

# Define log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Color codes for terminal output
class LogColors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        logging.DEBUG: LogColors.CYAN,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.RED + LogColors.BOLD,
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if record.levelno in self.COLORS:
            levelname_color = f"{self.COLORS[record.levelno]}{levelname}{LogColors.RESET}"
            record.levelname = levelname_color
        
        return super().format(record)

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, DATE_FORMAT))
    
    logger.addHandler(console_handler)
    
    return logger

def log_pipeline_step(logger: logging.Logger, step: int, title: str):
    """Log a pipeline step header"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP {step}: {title}")
    logger.info(f"{'='*60}")

def log_success(logger: logging.Logger, message: str):
    """Log a success message with checkmark"""
    logger.info(f"✓ {message}")

def log_error(logger: logging.Logger, message: str):
    """Log an error message with X mark"""
    logger.error(f"✗ {message}")

def log_warning(logger: logging.Logger, message: str):
    """Log a warning message with warning symbol"""
    logger.warning(f"⚠️  {message}")

def log_metric(logger: logging.Logger, label: str, value):
    """Log a metric or statistic"""
    logger.info(f"📊 {label}: {value}")

