"""
Comprehensive logging framework for SVN to Git migration.
Provides structured logging to both file and console.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog


class LoggerManager:
    """Manages logging configuration and instances."""

    _loggers = {}

    @classmethod
    def setup_logging(cls, log_dir: Path, log_level: str = 'INFO',
                     run_id: Optional[str] = None) -> None:
        """
        Set up the global logging configuration.

        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            run_id: Optional run identifier for log file naming
        """
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        log_dir.mkdir(parents=True, exist_ok=True)

        # Create formatters
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)-8s]%(reset)s %(asctime)s - %(name)s - %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )

        file_formatter = logging.Formatter(
            '[%(levelname)-8s] %(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level.upper())

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level.upper())
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler
        log_file = log_dir / f'migration_{run_id}.log'
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        cls._run_id = run_id
        cls._log_file = log_file

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger instance.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]

    @classmethod
    def get_log_file(cls) -> Optional[Path]:
        """Get the current log file path."""
        return getattr(cls, '_log_file', None)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return LoggerManager.get_logger(name)

