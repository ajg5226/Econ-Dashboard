"""
ALFRED Vintage Cache
====================

Caches "as-of" vintage snapshots of FRED series pulled via the ALFRED endpoint
so that repeat runs of the A2 vintage replay (and future vintage-based
experiments) do not re-pay the API cost.

Cache layout
------------
    data/alfred_cache/{feature_col}_{asof_date}.parquet

where ``feature_col`` matches the column name used inside the recession
engine (e.g. ``coincident_PAYEMS``), and ``asof_date`` is an ISO date
(``YYYY-MM-DD``). Each parquet contains two columns (``date``, ``value``)
and represents the historical series **as it was known on ``asof_date``** —
i.e. the ALFRED vintage snapshot.

Writes are atomic (temp file + rename). Corrupted parquets are deleted on
read failure so the next call transparently re-fetches.

Series that are not available on ALFRED (common for market data and a
handful of real-activity series) are handled gracefully: the fetcher logs
a debug message and returns ``None`` so the caller can fall back to its
simulated-lag path.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

# Default cache root lives alongside the project's data directory; callers
# can override via env var or by passing an explicit ``cache_dir``.
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "ALFRED_CACHE_DIR",
        str(Path(__file__).parent.parent / "data" / "alfred_cache"),
    )
)

# Polite delay between ALFRED hits to stay well within FRED's quota.
DEFAULT_REQUEST_DELAY = 0.5


def _feature_to_series_id(feature_col: str) -> str:
    """Convert an internal feature column name to its underlying FRED id."""
    return feature_col.split("_", 1)[1] if "_" in feature_col else feature_col


def _cache_path(feature_col: str, asof_date: str, cache_dir: Path) -> Path:
    """Return the parquet path for a ``(feature, asof)`` pair."""
    asof = pd.Timestamp(asof_date).strftime("%Y-%m-%d")
    return cache_dir / f"{feature_col}_{asof}.parquet"


def load_cached_vintage(
    feature_col: str,
    asof_date: str,
    cache_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Return cached vintage as a ``(date, value)`` DataFrame or ``None``."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    path = _cache_path(feature_col, asof_date, cache_dir)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Corrupted vintage cache %s (%s); deleting so it will be re-fetched.",
            path,
            exc,
        )
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if df is None:
        return None
    # Empty frames are legitimate sentinels ("series unavailable on ALFRED")
    # and are returned so callers can distinguish them from cache misses.
    return df


def fetch_and_cache_vintage(
    feature_col: str,
    asof_date: str,
    api_key: str,
    *,
    start_date: str = "1948-01-01",
    cache_dir: Optional[Path] = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    series_id: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Pull an ALFRED vintage for ``feature_col`` as of ``asof_date``.

    Parameters
    ----------
    feature_col:
        Engine-namespaced column name, e.g. ``coincident_PAYEMS``.
    asof_date:
        ISO date (``YYYY-MM-DD``). Both ``realtime_start`` and ``realtime_end``
        are set to this value so we receive the single vintage snapshot.
    api_key:
        FRED API key. Returns ``None`` if missing or empty.
    start_date:
        Earliest observation date to request. Defaults to 1948-01-01.
    cache_dir:
        Override cache root. Defaults to ``data/alfred_cache``.
    request_delay:
        Seconds to sleep *after* a network call (rate limiting).
    series_id:
        Optional explicit FRED id if it does not follow the
        ``prefix_SERIES`` convention.
    """
    if not api_key:
        logger.debug("No FRED API key available — cannot fetch %s", feature_col)
        return None

    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp(asof_date).strftime("%Y-%m-%d")
    path = _cache_path(feature_col, asof, cache_dir)
    series_id = series_id or _feature_to_series_id(feature_col)

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": asof,
        "realtime_end": asof,
        "observation_start": start_date,
        "observation_end": asof,
    }
    url = "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params)

    try:
        with urlopen(url) as response:
            payload = json.load(response)
    except HTTPError as exc:
        # FRED returns HTTP 400 with an informative message when a series
        # does not exist on ALFRED. Treat as an expected miss and cache
        # an empty sentinel to avoid re-querying on the next invocation.
        body = ""
        try:
            body = exc.read().decode()
        except Exception:
            pass
        logger.info(
            "ALFRED unavailable for %s as of %s (HTTP %s): %s",
            feature_col,
            asof,
            exc.code,
            body[:160].replace("\n", " "),
        )
        _write_sentinel(path)
        time.sleep(max(0.0, request_delay))
        return None
    except (URLError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "ALFRED fetch error for %s as of %s: %s",
            feature_col,
            asof,
            exc,
        )
        time.sleep(max(0.0, request_delay))
        return None

    observations = payload.get("observations", [])
    if not observations:
        logger.info("ALFRED returned zero observations for %s as of %s", feature_col, asof)
        _write_sentinel(path)
        time.sleep(max(0.0, request_delay))
        return None

    df = pd.DataFrame(observations)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        _write_sentinel(path)
        time.sleep(max(0.0, request_delay))
        return None

    # Atomic write: temp file + os.replace.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write vintage cache %s: %s", path, exc)
        try:
            tmp_path.unlink()
        except OSError:
            pass
    finally:
        time.sleep(max(0.0, request_delay))

    return df


def _write_sentinel(path: Path) -> None:
    """Persist an empty sentinel parquet so callers skip repeat misses."""
    try:
        empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                              "value": pd.Series(dtype="float64")})
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        empty.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:  # pragma: no cover - best-effort only
        pass


def get_vintage(
    feature_col: str,
    asof_date: str,
    api_key: Optional[str] = None,
    *,
    cache_dir: Optional[Path] = None,
    refetch_empty: bool = False,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    series_id: Optional[str] = None,
    start_date: str = "1948-01-01",
) -> Optional[pd.DataFrame]:
    """Cache-first lookup: return the vintage DataFrame or ``None``.

    ``None`` is returned both when the series is absent on ALFRED and when
    we are offline / missing an API key. Callers should treat both cases the
    same (fall back to simulated lag).

    If ``refetch_empty`` is ``True`` a cached empty sentinel is ignored and
    the request will go to ALFRED again — useful when new vintages have
    been published.
    """
    cached = load_cached_vintage(feature_col, asof_date, cache_dir=cache_dir)
    if cached is not None and not cached.empty:
        return cached
    if cached is not None and cached.empty and not refetch_empty:
        return None

    return fetch_and_cache_vintage(
        feature_col,
        asof_date,
        api_key or os.environ.get("FRED_API_KEY", ""),
        cache_dir=cache_dir,
        request_delay=request_delay,
        series_id=series_id,
        start_date=start_date,
    )


def vintage_series_to_monthly(df: pd.DataFrame) -> pd.Series:
    """Resample a raw (date, value) vintage frame to month-end.

    The A2 pipeline stores engine features at monthly frequency, so we
    downsample the ALFRED output using the last observation in each month
    and drop the sentinel empty frames transparently.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    series = (
        df.set_index("date")["value"]
        .sort_index()
        .resample("ME")
        .last()
    )
    series.name = "value"
    return series
