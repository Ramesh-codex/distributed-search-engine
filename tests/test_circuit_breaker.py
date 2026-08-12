import time

import pytest

from search.distribution.circuit_breaker import BreakerState, CircuitBreaker

COOLDOWN = 0.05  # real sleeps in these tests; kept tiny to stay fast


def test_starts_closed():
    breaker = CircuitBreaker()
    assert breaker.state == BreakerState.CLOSED.value
    assert breaker.allow_request() is True


def test_stays_closed_below_the_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=COOLDOWN)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == BreakerState.CLOSED.value
    assert breaker.allow_request() is True


def test_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=COOLDOWN)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == BreakerState.CLOSED.value  # not yet: 2 < 3
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN.value  # 3rd consecutive failure


def test_a_success_resets_the_consecutive_failure_count():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=COOLDOWN)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == BreakerState.CLOSED.value  # count restarted after the success


def test_skips_requests_while_open():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=10.0)
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN.value
    assert breaker.allow_request() is False
    assert breaker.allow_request() is False  # still within the 10s cooldown


def test_half_open_probe_allowed_after_cooldown():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=COOLDOWN)
    breaker.record_failure()
    assert breaker.allow_request() is False

    time.sleep(COOLDOWN * 2)
    assert breaker.allow_request() is True
    assert breaker.state == BreakerState.HALF_OPEN.value


def test_half_open_allows_only_one_probe_at_a_time():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=COOLDOWN)
    breaker.record_failure()
    time.sleep(COOLDOWN * 2)

    assert breaker.allow_request() is True  # the one probe
    assert breaker.allow_request() is False  # a concurrent caller: no second probe
    assert breaker.allow_request() is False


def test_half_open_closes_on_success():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=COOLDOWN)
    breaker.record_failure()
    time.sleep(COOLDOWN * 2)
    assert breaker.allow_request() is True  # now half-open

    breaker.record_success()
    assert breaker.state == BreakerState.CLOSED.value
    assert breaker.consecutive_failures == 0
    assert breaker.allow_request() is True


def test_half_open_reopens_on_a_failed_probe():
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=COOLDOWN)
    breaker.record_failure()
    time.sleep(COOLDOWN * 2)
    assert breaker.allow_request() is True  # now half-open

    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN.value
    assert breaker.allow_request() is False  # fresh cooldown, doesn't immediately reopen

    time.sleep(COOLDOWN * 2)
    assert breaker.allow_request() is True  # cooldown elapsed again: another probe


@pytest.mark.parametrize("failure_threshold", [0, -1])
def test_rejects_a_nonpositive_failure_threshold(failure_threshold):
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=failure_threshold)


@pytest.mark.parametrize("cooldown_seconds", [0, -1.0])
def test_rejects_a_nonpositive_cooldown(cooldown_seconds):
    with pytest.raises(ValueError):
        CircuitBreaker(cooldown_seconds=cooldown_seconds)
