"""
Phase 2 support: rate limiting, cache, and performance monitoring in one module.
"""

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

from app.phases.phase2.query_optimizer import normalize_query
from app.phases.phase2.schemas import SearchResult

# ----- Rate limiter -----


@dataclass
class RateLimitConfig:
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    backoff_base_seconds: float = 1.0
    max_backoff_seconds: float = 60.0


@dataclass
class RateLimitState:
    requests_per_minute: list[float] = field(default_factory=list)
    requests_per_hour: list[float] = field(default_factory=list)
    backoff_until: float = 0.0
    consecutive_errors: int = 0
    lock: Lock = field(default_factory=Lock)


class RateLimiter:
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._states: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = Lock()

    def _get_state(self, api_key: str) -> RateLimitState:
        with self._lock:
            if api_key not in self._states:
                self._states[api_key] = RateLimitState()
            return self._states[api_key]

    def _cleanup_old_requests(self, state: RateLimitState, current_time: float):
        state.requests_per_minute = [ts for ts in state.requests_per_minute if current_time - ts < 60.0]
        state.requests_per_hour = [ts for ts in state.requests_per_hour if current_time - ts < 3600.0]

    def wait_if_needed(self, api_key: str) -> float:
        state = self._get_state(api_key)
        current_time = time.time()
        wait_time = 0.0
        with state.lock:
            if current_time < state.backoff_until:
                wait_time = state.backoff_until - current_time
                time.sleep(wait_time)
                current_time = time.time()
            self._cleanup_old_requests(state, current_time)
            if len(state.requests_per_minute) >= self.config.max_requests_per_minute:
                oldest = min(state.requests_per_minute)
                wait_time = 60.0 - (current_time - oldest) + 0.1
                if wait_time > 0:
                    time.sleep(wait_time)
                    current_time = time.time()
                    self._cleanup_old_requests(state, current_time)
            if len(state.requests_per_hour) >= self.config.max_requests_per_hour:
                oldest = min(state.requests_per_hour)
                wait_time = 3600.0 - (current_time - oldest) + 0.1
                if wait_time > 0:
                    time.sleep(wait_time)
                    current_time = time.time()
                    self._cleanup_old_requests(state, current_time)
            state.requests_per_minute.append(current_time)
            state.requests_per_hour.append(current_time)
        return wait_time

    def record_success(self, api_key: str):
        state = self._get_state(api_key)
        with state.lock:
            state.consecutive_errors = 0
            state.backoff_until = 0.0

    def record_error(self, api_key: str, is_rate_limit_error: bool = False):
        state = self._get_state(api_key)
        with state.lock:
            state.consecutive_errors += 1
            if is_rate_limit_error or state.consecutive_errors >= 3:
                backoff_time = min(
                    self.config.backoff_base_seconds * (2 ** state.consecutive_errors),
                    self.config.max_backoff_seconds,
                )
                state.backoff_until = time.time() + backoff_time


_tavily_limiter = RateLimiter(RateLimitConfig(max_requests_per_minute=30, max_requests_per_hour=500))
_serper_limiter = RateLimiter(RateLimitConfig(max_requests_per_minute=50, max_requests_per_hour=1000))


def get_rate_limiter(api_name: str) -> RateLimiter:
    if api_name.lower() == "tavily":
        return _tavily_limiter
    if api_name.lower() == "serper":
        return _serper_limiter
    return RateLimiter()


# ----- Cache -----


@dataclass
class CachedSearchResult:
    query: str
    normalized_query: str
    results: list[dict]
    timestamp: float
    source_api: str
    expires_at: float


class SearchCache:
    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: int = 3600):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._memory_cache: dict[str, CachedSearchResult] = {}

    def _get_cache_key(self, query: str, api_name: str) -> str:
        key_string = f"{api_name}:{normalize_query(query)}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_file(self, cache_key: str) -> Optional[CachedSearchResult]:
        cache_file = self._get_cache_file(cache_key)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() > data.get("expires_at", 0):
                cache_file.unlink()
                return None
            return CachedSearchResult(
                query=data["query"],
                normalized_query=data["normalized_query"],
                results=data["results"],
                timestamp=data["timestamp"],
                source_api=data["source_api"],
                expires_at=data["expires_at"],
            )
        except Exception as e:
            print(f"Error loading cache file {cache_file}: {e}")
            return None

    def _save_to_file(self, cache_key: str, cached: CachedSearchResult):
        try:
            with open(self._get_cache_file(cache_key), "w", encoding="utf-8") as f:
                json.dump(asdict(cached), f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def get(self, query: str, api_name: str) -> Optional[list[SearchResult]]:
        cache_key = self._get_cache_key(query, api_name)
        if cache_key in self._memory_cache:
            cached = self._memory_cache[cache_key]
            if time.time() < cached.expires_at:
                return [SearchResult(**r) for r in cached.results]
            del self._memory_cache[cache_key]
        cached = self._load_from_file(cache_key)
        if cached:
            self._memory_cache[cache_key] = cached
            return [SearchResult(**r) for r in cached.results]
        return None

    def set(self, query: str, api_name: str, results: list[SearchResult]):
        cache_key = self._get_cache_key(query, api_name)
        current_time = time.time()
        cached = CachedSearchResult(
            query=query,
            normalized_query=normalize_query(query),
            results=[r.model_dump() for r in results],
            timestamp=current_time,
            source_api=api_name,
            expires_at=current_time + self.ttl_seconds,
        )
        self._memory_cache[cache_key] = cached
        self._save_to_file(cache_key, cached)


_search_cache = SearchCache()


def get_search_cache() -> SearchCache:
    return _search_cache


# ----- Performance monitor -----


@dataclass
class SearchMetrics:
    query: str
    api_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    result_count: int = 0
    error_message: Optional[str] = None
    cache_hit: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.end_time or 0.0) - self.start_time


@dataclass
class PerformanceStats:
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    cache_hits: int = 0
    total_duration_seconds: float = 0.0
    avg_duration_seconds: float = 0.0
    min_duration_seconds: float = float("inf")
    max_duration_seconds: float = 0.0
    total_results: int = 0
    avg_results_per_search: float = 0.0
    api_stats: dict[str, "PerformanceStats"] = field(default_factory=dict)


class PerformanceMonitor:
    def __init__(self):
        self._metrics: list[SearchMetrics] = []
        self._lock = Lock()
        self._max_metrics = 1000

    def start_search(self, query: str, api_name: str) -> SearchMetrics:
        return SearchMetrics(query=query, api_name=api_name, start_time=time.time())

    def record_success(self, metric: SearchMetrics, result_count: int, cache_hit: bool = False):
        metric.end_time = time.time()
        metric.success = True
        metric.result_count = result_count
        metric.cache_hit = cache_hit
        with self._lock:
            self._metrics.append(metric)
            if len(self._metrics) > self._max_metrics:
                self._metrics = self._metrics[-self._max_metrics :]

    def record_error(self, metric: SearchMetrics, error_message: str):
        metric.end_time = time.time()
        metric.success = False
        metric.error_message = error_message
        with self._lock:
            self._metrics.append(metric)
            if len(self._metrics) > self._max_metrics:
                self._metrics = self._metrics[-self._max_metrics :]

    def get_stats(self, api_name: Optional[str] = None) -> PerformanceStats:
        with self._lock:
            metrics = self._metrics.copy()
        if api_name:
            metrics = [m for m in metrics if m.api_name == api_name]
        if not metrics:
            return PerformanceStats()
        successful = [m for m in metrics if m.success]
        durations = [m.duration_seconds for m in metrics if m.end_time]
        total_results = sum(m.result_count for m in successful)
        return PerformanceStats(
            total_searches=len(metrics),
            successful_searches=len(successful),
            failed_searches=len(metrics) - len(successful),
            cache_hits=sum(1 for m in metrics if m.cache_hit),
            total_duration_seconds=sum(durations),
            avg_duration_seconds=sum(durations) / len(durations) if durations else 0.0,
            min_duration_seconds=min(durations) if durations else 0.0,
            max_duration_seconds=max(durations) if durations else 0.0,
            total_results=total_results,
            avg_results_per_search=total_results / len(successful) if successful else 0.0,
        )


_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    return _performance_monitor
