"""
Error handling and recovery strategies for SVN to Git migration.
"""

import time
from typing import Callable, TypeVar, Any, Optional
from svn2git_tool.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class MigrationError(Exception):
    """Base exception for migration errors."""
    pass


class SVNConnectionError(MigrationError):
    """Raised when SVN connection fails."""
    pass


class SVNFetchError(MigrationError):
    """Raised when fetching from SVN fails."""
    pass


class GitAuthenticationError(MigrationError):
    """Raised when Git authentication fails."""
    pass


class GitPushError(MigrationError):
    """Raised when pushing to Git fails."""
    pass


class TimeoutError(MigrationError):
    """Raised when operation exceeds timeout."""
    pass


class StateManagementError(MigrationError):
    """Raised when state management operations fail."""
    pass


class FailureMode:
    """Enum-like class for different failure modes."""
    TRANSIENT = 'transient'  # Temporary error, retry recommended
    PERMANENT = 'permanent'  # Permanent error, manual intervention needed
    TIMEOUT = 'timeout'      # Operation timed out, resume from checkpoint
    VALIDATION = 'validation'  # Data validation failed


def classify_error(exception: Exception) -> str:
    """
    Classify an exception to determine if it's transient or permanent.

    Args:
        exception: The exception to classify

    Returns:
        FailureMode constant
    """
    if isinstance(exception, TimeoutError):
        return FailureMode.TIMEOUT

    if isinstance(exception, (SVNConnectionError, GitPushError)):
        # Network errors are typically transient
        return FailureMode.TRANSIENT

    if isinstance(exception, (SVNFetchError, GitAuthenticationError)):
        # Auth and data errors are typically permanent
        return FailureMode.PERMANENT

    return FailureMode.PERMANENT


def retry_with_backoff(
    func: Callable[..., T],
    max_attempts: int = 3,
    base_delay: int = 5,
    max_delay: int = 300,
    backoff_multiplier: float = 3.0,
    on_retry: Optional[Callable[[int, Exception], None]] = None
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        on_retry: Optional callback function called on retry (attempt_num, exception)

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"Attempt {attempt}/{max_attempts}: Executing {func.__name__}")
            return func()
        except Exception as e:
            last_exception = e

            if attempt >= max_attempts:
                logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (backoff_multiplier ** (attempt - 1)), max_delay)

            # Call retry callback if provided
            if on_retry:
                on_retry(attempt, e)

            logger.warning(
                f"Attempt {attempt} failed: {e}. "
                f"Retrying in {delay:.1f}s... ({max_attempts - attempt} retries left)"
            )

            time.sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


def handle_transient_error(
    exception: Exception,
    attempt: int,
    max_attempts: int,
    operation_name: str
) -> None:
    """
    Log and handle a transient error.

    Args:
        exception: The exception that occurred
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed
        operation_name: Name of the operation being attempted
    """
    remaining = max_attempts - attempt
    logger.warning(
        f"Transient error in {operation_name} (attempt {attempt}/{max_attempts}): {exception}. "
        f"Will retry {remaining} more time(s)."
    )


def handle_permanent_error(
    exception: Exception,
    operation_name: str,
    recovery_hint: Optional[str] = None
) -> None:
    """
    Log and handle a permanent error.

    Args:
        exception: The exception that occurred
        operation_name: Name of the operation that failed
        recovery_hint: Hint for manual recovery
    """
    error_msg = f"Permanent error in {operation_name}: {exception}"
    if recovery_hint:
        error_msg += f"\nRecovery hint: {recovery_hint}"

    logger.error(error_msg)


def handle_timeout_error(
    operation_name: str,
    last_checkpoint: Optional[Any] = None
) -> None:
    """
    Log and handle a timeout error with recovery information.

    Args:
        operation_name: Name of the operation that timed out
        last_checkpoint: Last known good state/checkpoint
    """
    msg = f"Timeout occurred during {operation_name}"
    if last_checkpoint:
        msg += f". Last checkpoint: {last_checkpoint}"
    msg += ". Migration will resume from the last checkpoint on next run."

    logger.warning(msg)

