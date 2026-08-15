"""Retry policy + exponential backoff (Phase 11).

Differentiates RETRYABLE from NON_RETRYABLE failures and applies bounded
exponential backoff with jitter for retryable provider failures. No tight
retry loops. Every attempt is recorded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..core.errors import TypedErrorCode, VideoError
from .models import Job, is_retryable, is_non_retryable


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff config."""

    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0
    max_retries: int = 3
    jitter: float = 0.2  # +/- fraction of computed delay

    def compute_delay(self, attempt: int) -> float:
        """Bounded exponential backoff with jitter for attempt N (0-based)."""
        if attempt < 0:
            attempt = 0
        delay = self.base_delay * (2 ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter > 0:
            delta = delay * self.jitter
            delay = delay + random.uniform(-delta, delta)
        return max(0.0, delay)


DEFAULT_POLICY = RetryPolicy()


def classify_error(code: Optional[str]) -> str:
    """Return 'RETRYABLE', 'NON_RETRYABLE', or 'UNKNOWN'."""
    if is_retryable(code):
        return "RETRYABLE"
    if is_non_retryable(code):
        return "NON_RETRYABLE"
    return "UNKNOWN"


def should_retry(job: Job, policy: RetryPolicy = DEFAULT_POLICY) -> bool:
    """True if the job's last failure is retryable and retries remain."""
    if not is_retryable(job.error_code):
        return False
    return job.retry_count < min(job.max_retries, policy.max_retries)


def next_delay(job: Job, policy: RetryPolicy = DEFAULT_POLICY) -> float:
    """Compute the next backoff delay for a failed retryable job."""
    return policy.compute_delay(job.retry_count)


def record_retry(job: Job, policy: RetryPolicy = DEFAULT_POLICY) -> float:
    """Record a retry attempt on the job. Returns the backoff delay used.

    Raises NON_RETRYABLE if the error is deterministic and must not be retried.
    """
    if is_non_retryable(job.error_code):
        raise VideoError(
            TypedErrorCode.NON_RETRYABLE,
            f"Error '{job.error_code}' is non-retryable for job {job.job_id}.",
            context={"job_id": job.job_id, "error_code": job.error_code,
                     "error_detail": job.error_detail},
        )
    if not is_retryable(job.error_code):
        raise VideoError(
            TypedErrorCode.NON_RETRYABLE,
            f"Error '{job.error_code}' is not classified retryable for job {job.job_id}.",
            context={"job_id": job.job_id, "error_code": job.error_code},
        )
    if job.retry_count >= min(job.max_retries, policy.max_retries):
        raise VideoError(
            TypedErrorCode.RETRY_EXHAUSTED,
            f"Job {job.job_id} exhausted retries ({job.retry_count}/{policy.max_retries}).",
            context={"job_id": job.job_id, "retry_count": job.retry_count,
                     "max_retries": policy.max_retries},
        )
    delay = next_delay(job, policy)
    job.retry_count += 1
    job.last_backoff = delay
    return delay
