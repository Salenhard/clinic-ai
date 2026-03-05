"""
Token-bucket rate limiter for the free Claude API tier.

Free tier limits (as of 2025):
  - 5 requests per minute
  - Shared across all pipeline stages

Usage:
    limiter = RateLimiter(requests_per_minute=5)
    limiter.acquire()   # blocks until a slot is available
"""
import threading
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 5):
        self.rpm = requests_per_minute
        # Minimum gap between requests in seconds (with small safety margin)
        self._min_interval = 60.0 / requests_per_minute + 0.5
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0
        self._call_count: int = 0

    def acquire(self) -> None:
        """Block until it is safe to make the next API request."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            wait = self._min_interval - elapsed

            if wait > 0:
                logger.info(
                    f"⏳ Rate limit: waiting {wait:.1f}s "
                    f"(free tier — {self.rpm} req/min)"
                )
                time.sleep(wait)

            self._last_call_time = time.monotonic()
            self._call_count += 1
            logger.debug(f"Rate limiter: request #{self._call_count} allowed")

    @property
    def total_requests(self) -> int:
        return self._call_count


# Singleton shared across all stages
_global_limiter: RateLimiter | None = None


def get_limiter(requests_per_minute: int = 5) -> RateLimiter:
    """Return the global rate limiter, creating it if needed."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter(requests_per_minute)
    return _global_limiter


def configure_limiter(requests_per_minute: int) -> RateLimiter:
    """(Re)create the global limiter with a specific RPM."""
    global _global_limiter
    _global_limiter = RateLimiter(requests_per_minute)
    logger.info(f"Rate limiter configured: {requests_per_minute} req/min")
    return _global_limiter
