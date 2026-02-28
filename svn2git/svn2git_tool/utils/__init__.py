"""
Utilities module for logging and error handling.

Contains:
- Logging framework
- Error handling and recovery
"""

from .logger import LoggerManager, get_logger
from .error_handler import (
    MigrationError,
    SVNConnectionError,
    SVNFetchError,
    GitAuthenticationError,
    GitPushError,
    TimeoutError as MigrationTimeoutError,
    FailureMode,
    classify_error,
    retry_with_backoff,
    handle_transient_error,
    handle_permanent_error,
    handle_timeout_error,
)

__all__ = [
    "LoggerManager",
    "get_logger",
    "MigrationError",
    "SVNConnectionError",
    "SVNFetchError",
    "GitAuthenticationError",
    "GitPushError",
    "MigrationTimeoutError",
    "FailureMode",
    "classify_error",
    "retry_with_backoff",
    "handle_transient_error",
    "handle_permanent_error",
    "handle_timeout_error",
]

