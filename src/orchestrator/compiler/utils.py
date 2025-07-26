"""Utility functions for the compiler module."""

import asyncio
import functools
import logging
from typing import Callable, Type, Tuple, Any

logger = logging.getLogger(__name__)


def async_retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
) -> Callable:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        exceptions: Tuple of exceptions to catch and retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        max_delay: Maximum delay between retries

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    else:
                        # Log and retry
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * backoff, max_delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def is_transient_error(error: Exception) -> bool:
    """
    Determine if an error is transient and should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error is transient
    """
    error_str = str(error).lower()
    transient_indicators = [
        "timeout",
        "connection",
        "network",
        "rate limit",
        "throttl",
        "unavailable",
        "temporary",
        "retry",
        "too many requests",
        "429",  # HTTP 429 Too Many Requests
        "503",  # HTTP 503 Service Unavailable
        "504",  # HTTP 504 Gateway Timeout
    ]

    return any(indicator in error_str for indicator in transient_indicators)
