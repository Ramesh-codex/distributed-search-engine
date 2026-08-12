import threading
import time
from enum import Enum


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised by a caller that chooses to treat allow_request()==False as
    an exception rather than branching on the bool. Coordinator uses this
    so a skipped shard fails the same way a slow/error shard does."""


class CircuitBreaker:
    """Per-dependency failure tracker: closed (normal) -> open (skip
    entirely) -> half-open (one probe) -> closed or open again.

    Exists because a timeout only bounds the cost of one request. A
    dependency that is actually down still gets hit, and still costs a full
    timeout, on every single subsequent call. After enough consecutive
    failures, stop calling it at all for a cooldown window instead of
    paying for a doomed request every time.

    Thread-safe: allow_request() both reads and, when the cooldown has
    elapsed, transitions OPEN -> HALF_OPEN, all under one lock. That
    transition granting exactly one caller permission to probe is what
    stops every concurrent caller from independently deciding the cooldown
    is over and each sending their own probe the instant it expires.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 10.0) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")

        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def allow_request(self) -> bool:
        """Call before attempting the guarded operation. False means: don't
        bother, the circuit is open and the cooldown hasn't elapsed yet."""
        with self._lock:
            if self._state is BreakerState.CLOSED:
                return True

            if self._state is BreakerState.OPEN:
                assert self._opened_at is not None
                if time.monotonic() - self._opened_at < self._cooldown_seconds:
                    return False
                # Cooldown elapsed: this caller gets the one half-open probe.
                self._state = BreakerState.HALF_OPEN
                return True

            # HALF_OPEN: a probe is already in flight; no second one until
            # record_success()/record_failure() resolves it.
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = BreakerState.CLOSED
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            if self._state is BreakerState.HALF_OPEN:
                # The probe failed: back to open for a fresh cooldown.
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()
                return

            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state.value

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures
