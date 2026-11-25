"""Protect the free LLM quota when real people use the app.

TTLCache: the same question again -> return the saved answer, no LLM call.
          (Many people ask "Can I withdraw consent?" in the same words.)
RateLimiter: each user gets N questions per minute and per day, so one person
          (or a bot) can't use up everyone's free quota.

Both live in memory: simple, fast, and fine for one server. With several servers
you would move them to Redis (shared memory for all servers).
"""
import re
import threading
import time
from collections import OrderedDict, deque


def normalize(text: str) -> str:
    """'Can I withdraw  my consent?' and 'can i withdraw my consent' -> same key."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


class TTLCache:
    def __init__(self, max_items: int, ttl_seconds: float):
        self.max_items, self.ttl = max_items, ttl_seconds
        self.data: OrderedDict = OrderedDict()  # key -> (saved_at, value), oldest first
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            item = self.data.get(key)
            if item is None:
                return None
            saved_at, value = item
            if time.time() - saved_at > self.ttl:
                del self.data[key]
                return None
            self.data.move_to_end(key)  # recently used -> keep longer (LRU)
            return value

    def set(self, key, value) -> None:
        with self.lock:
            self.data[key] = (time.time(), value)
            self.data.move_to_end(key)
            while len(self.data) > self.max_items:
                self.data.popitem(last=False)  # drop the least recently used


class RateLimiter:
    """Sliding window: remember when each user asked; count the last minute / day."""

    def __init__(self, per_minute: int, per_day: int):
        self.per_minute, self.per_day = per_minute, per_day
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def check(self, client: str) -> int:
        """Returns 0 if allowed (and records it), else seconds to wait."""
        now = time.time()
        with self.lock:
            q = self.hits.setdefault(client, deque())
            while q and now - q[0] > 86400:
                q.popleft()
            if len(q) >= self.per_day:
                return int(86400 - (now - q[0])) + 1
            last_minute = [t for t in q if now - t < 60]
            if len(last_minute) >= self.per_minute:
                return int(60 - (now - last_minute[0])) + 1
            q.append(now)
            return 0
