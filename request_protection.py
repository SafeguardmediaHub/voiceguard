"""
request_protection.py — VoiceGuard V8 rate limiting and anomaly detection

Wraps the /detect endpoint to slow down model probing and flag suspicious
query patterns. Two layers:

  1. RATE LIMITING — token-bucket per source IP. Default: 30 requests per
     minute, burst of 10. Prevents bulk model querying.

  2. ANOMALY DETECTION — three lightweight rules:
     - Exact-duplicate detection: same file SHA-256 submitted multiple times
       within a window (signal: gradient estimation against a fixed input)
     - Burst detection: many requests from one source in a short window
       (signal: automated probing)
     - Cumulative volume: source has exceeded a daily soft limit
       (signal: bulk extraction attempt)

Honest scope notes:
  - This is NOT comprehensive attack detection. A patient attacker can stay
    under the thresholds. The goal is to make casual/automated probing
    expensive enough to deter, not to stop a determined adversary.
  - All thresholds are configurable below. Tune to your traffic patterns —
    legitimate batch users may need higher rate limits.
  - State is in-memory only; restart wipes counts. For production with
    multiple workers, you'd want Redis-backed state.

Behavior summary:
  - Hard block: rate limit exceeded → 429 response with retry-after header
  - Soft flag: anomaly detected → request is allowed, but result gets a
    'suspicious' flag in the response metadata; operator can log/alert
"""
import time
import hashlib
import threading
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional


# ── Configuration ───────────────────────────────────────────────────────────
# All values are conservative defaults. Tune as needed.

# Rate limiting (token bucket)
RATE_LIMIT_PER_MINUTE = 30      # sustainable rate
RATE_LIMIT_BURST = 10           # short-burst capacity above sustainable rate

# Anomaly: duplicate detection
DUPLICATE_WINDOW_SEC = 300      # 5 minutes — same file hash within this counts
DUPLICATE_THRESHOLD = 3         # 3 repeats triggers a flag

# Anomaly: burst detection (separate from rate limiting; tracks bursts shorter
# than the rate-limit window)
BURST_WINDOW_SEC = 10
BURST_THRESHOLD = 8

# Anomaly: cumulative daily volume
DAILY_VOLUME_WINDOW_SEC = 86400
DAILY_VOLUME_SOFT_LIMIT = 500


class TokenBucket:
    """Standard token-bucket rate limiter. Thread-safe."""

    def __init__(self, rate_per_minute: float, burst: int):
        self.rate = rate_per_minute / 60.0   # tokens per second
        self.burst = burst
        self.tokens = float(burst)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def allow(self) -> Tuple[bool, float]:
        """Try to consume one token.

        Returns:
            (allowed, retry_after_sec). retry_after_sec is 0 if allowed.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            else:
                # How long until the next token?
                deficit = 1.0 - self.tokens
                wait = deficit / self.rate
                return False, wait


class RequestProtection:
    """Per-source rate limiting + anomaly detection.

    Identifies sources by a string key (typically client IP). Maintains:
      - A token bucket per source
      - Sliding window of (timestamp, file_hash) for anomaly detection
    """

    def __init__(self,
                 rate_per_minute: int = RATE_LIMIT_PER_MINUTE,
                 burst: int = RATE_LIMIT_BURST):
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self._buckets: Dict[str, TokenBucket] = {}
        # Deque of (timestamp, file_hash) per source
        self._history: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _get_bucket(self, source: str) -> TokenBucket:
        with self._lock:
            if source not in self._buckets:
                self._buckets[source] = TokenBucket(self.rate_per_minute, self.burst)
            return self._buckets[source]

    def _prune_history(self, source: str, now: float):
        """Remove entries older than the largest window we care about."""
        h = self._history[source]
        max_window = max(DUPLICATE_WINDOW_SEC, DAILY_VOLUME_WINDOW_SEC,
                         BURST_WINDOW_SEC)
        cutoff = now - max_window
        while h and h[0][0] < cutoff:
            h.popleft()

    def check_request(self, source: str, file_hash: Optional[str] = None
                       ) -> Tuple[bool, float, Dict]:
        """Check whether a request should be allowed and whether it's anomalous.

        Args:
            source: Source identifier (typically client IP address).
            file_hash: Optional SHA-256 hash of the uploaded file, used for
                       duplicate detection. Pass None to skip duplicate check.

        Returns:
            (allowed, retry_after_sec, info_dict)
              - allowed: False if rate-limited (caller should return 429)
              - retry_after_sec: seconds until next token available (when blocked)
              - info_dict: contains 'anomalies' list with any flags raised
                           and 'flagged' bool summary

        Behavior:
            - Hard block on rate limit (allowed=False)
            - Soft flag on anomaly (allowed=True, but info has flags)
        """
        bucket = self._get_bucket(source)
        allowed, retry_after = bucket.allow()

        if not allowed:
            return allowed, retry_after, {
                'flagged': True,
                'anomalies': ['rate_limit_exceeded'],
                'rate_limited': True,
            }

        # Rate-limit passed — record this request and run anomaly checks
        now = time.time()
        anomalies = []

        with self._lock:
            self._prune_history(source, now)
            h = self._history[source]
            h.append((now, file_hash))

            # Duplicate detection (only if we have a hash to compare)
            if file_hash is not None:
                cutoff = now - DUPLICATE_WINDOW_SEC
                dup_count = sum(1 for ts, fh in h
                                if ts >= cutoff and fh == file_hash)
                if dup_count >= DUPLICATE_THRESHOLD:
                    anomalies.append('duplicate_submissions')

            # Burst detection
            burst_cutoff = now - BURST_WINDOW_SEC
            burst_count = sum(1 for ts, _ in h if ts >= burst_cutoff)
            if burst_count >= BURST_THRESHOLD:
                anomalies.append('burst_pattern')

            # Cumulative daily volume
            daily_count = len(h)  # already pruned above
            if daily_count >= DAILY_VOLUME_SOFT_LIMIT:
                anomalies.append('high_daily_volume')

        return True, 0.0, {
            'flagged': bool(anomalies),
            'anomalies': anomalies,
            'rate_limited': False,
        }

    def stats(self) -> Dict:
        """Snapshot of current state, for logging or admin endpoint."""
        with self._lock:
            now = time.time()
            return {
                'active_sources': len(self._buckets),
                'sources_with_history': len(self._history),
                'config': {
                    'rate_per_minute': self.rate_per_minute,
                    'burst': self.burst,
                    'duplicate_window_sec': DUPLICATE_WINDOW_SEC,
                    'duplicate_threshold': DUPLICATE_THRESHOLD,
                    'burst_window_sec': BURST_WINDOW_SEC,
                    'burst_threshold': BURST_THRESHOLD,
                },
                'timestamp': now,
            }


# ── Helper for computing SHA-256 from a file-like object ───────────────────
def hash_file_content(file_path: str) -> str:
    """Return the full SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ── Module-level singleton (use this from the Flask app) ────────────────────
_global_protection: Optional[RequestProtection] = None


def get_protection() -> RequestProtection:
    """Lazy initialization of the module-level protection instance."""
    global _global_protection
    if _global_protection is None:
        _global_protection = RequestProtection()
    return _global_protection
