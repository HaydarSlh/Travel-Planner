"""live_conditions tool — weather from Open-Meteo and a flight availability stub.

Open-Meteo is free and needs no API key. We geocode the destination name via
their free geocoding API, then fetch monthly climate normals for the requested
travel month. Results are TTL-cached (10 minutes) so repeated calls for the
same city+month don't burn network round trips.

Amadeus flight data is optional — if AMADEUS_API_KEY is blank the tool returns
flight_available=False rather than raising. This lets the agent run in
development without credentials.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog
from cachetools import TTLCache

from core.config import get_settings
from schemas.tools import LiveConditionsInput, LiveConditionsOutput, ToolError

log = structlog.get_logger()

_TOOL_NAME = "live_conditions"

# TTL cache: up to 128 city+month pairs, each valid for 10 minutes.
# A lock prevents the thundering-herd on simultaneous cache misses.
_cache: TTLCache = TTLCache(maxsize=128, ttl=600)
_cache_lock = asyncio.Lock()

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"

# WMO weather-interpretation codes → short human label
_WMO_LABELS: dict[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "rain showers", 81: "heavy showers", 82: "violent showers",
    95: "thunderstorm", 96: "thunderstorm with hail",
}

# Month index → name, for the weather summary string
_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


async def _geocode(name: str, http: httpx.AsyncClient) -> tuple[float, float] | None:
    """Return (latitude, longitude) for a destination name, or None on failure.

    Fetches up to 5 results so we can pick the best match when the top hit is a
    false positive (e.g. "Bali" → Bali, India instead of Bali, Indonesia).
    When the name contains a country hint (comma-separated) we prefer results
    whose country_code or admin1 match that hint.
    """
    parts = [p.strip() for p in name.split(",")]
    city = parts[0]
    country_hint = parts[-1].lower() if len(parts) > 1 else ""

    for query in _geocode_candidates(name):
        resp = await http.get(
            _GEO_URL,
            params={"name": query, "count": 5, "language": "en", "format": "json"},
            timeout=8.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            continue

        # If we have a country hint, try to find a result whose country name or
        # country_code matches — this prevents "Bali, Indonesia" → Bali, India.
        if country_hint:
            for r in results:
                country_name = (r.get("country") or "").lower()
                country_code = (r.get("country_code") or "").lower()
                admin1 = (r.get("admin1") or "").lower()
                if (
                    country_hint in country_name
                    or country_hint in country_code
                    or country_hint in admin1
                    or country_name in country_hint
                ):
                    return r["latitude"], r["longitude"]

        # No country hint or no match found — fall back to top result
        return results[0]["latitude"], results[0]["longitude"]

    return None


def _geocode_candidates(name: str) -> list[str]:
    """Return the name as-is, then progressively shorter comma-split variants."""
    parts = [p.strip() for p in name.split(",")]
    candidates = []
    for i in range(len(parts), 0, -1):
        candidates.append(", ".join(parts[:i]))
    seen: set[str] = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


_CLIMATE_MODELS = ["MRI_AGCM3_2_S", "NICAM16_8S", "EC_Earth3P_HR"]
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


async def _fetch_climate(lat: float, lon: float, month: int, http: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch 30-year climate normals for a lat/lon and extract the given month's data.

    Tries each climate model in order; falls back to the forecast API (current conditions)
    if all climate models return 400 for this coordinate.
    """
    for model in _CLIMATE_MODELS:
        try:
            resp = await http.get(
                _CLIMATE_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": "1991-01-01",
                    "end_date": "2020-12-31",
                    "models": model,
                    "monthly": "temperature_2m_mean,precipitation_sum",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            monthly = data.get("monthly", {})
            temps: list[float] = monthly.get("temperature_2m_mean", [None] * 12)
            precip: list[float] = monthly.get("precipitation_sum", [None] * 12)
            idx = month - 1
            avg_temp = temps[idx] if idx < len(temps) else None
            precip_mm = precip[idx] if idx < len(precip) else None
            if avg_temp is not None:
                return {"avg_temp_c": avg_temp, "precipitation_mm": precip_mm}
        except httpx.HTTPStatusError:
            log.debug("live_conditions.climate_model_unavailable", model=model, lat=lat, lon=lon)
            continue

    # Last resort: Open-Meteo historical reanalysis — global coverage, works everywhere.
    # We average the same calendar month across the last 5 years.
    _HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    import statistics
    try:
        year_now = time.localtime().tm_year
        all_temps: list[float] = []
        all_precip: list[float] = []
        for yr in range(year_now - 5, year_now):
            start = f"{yr}-{month:02d}-01"
            # last day of month (safe approximation)
            end_day = 28 if month == 2 else (30 if month in {4, 6, 9, 11} else 31)
            end = f"{yr}-{month:02d}-{end_day:02d}"
            r = await http.get(
                _HISTORICAL_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date": end,
                    "daily": "temperature_2m_mean,precipitation_sum",
                    "timezone": "auto",
                },
                timeout=10.0,
            )
            if r.status_code != 200:
                continue
            daily = r.json().get("daily", {})
            temps_d = [v for v in (daily.get("temperature_2m_mean") or []) if v is not None]
            precip_d = [v for v in (daily.get("precipitation_sum") or []) if v is not None]
            if temps_d:
                all_temps.append(statistics.mean(temps_d))
            if precip_d:
                all_precip.append(sum(precip_d))
        avg_temp = round(statistics.mean(all_temps), 1) if all_temps else None
        precip_mm = round(statistics.mean(all_precip), 0) if all_precip else None
        if avg_temp is not None:
            log.debug("live_conditions.historical_fallback_used", lat=lat, lon=lon)
            return {"avg_temp_c": avg_temp, "precipitation_mm": precip_mm}
    except Exception:  # noqa: BLE001
        pass

    return {"avg_temp_c": None, "precipitation_mm": None}


def _weather_summary(avg_temp: float | None, precip: float | None, month: int) -> str:
    if avg_temp is None:
        return "Weather data unavailable."
    month_name = _MONTH_NAMES[month]
    parts = [f"In {month_name}, expect average temperatures around {avg_temp:.1f}°C"]
    if precip is not None:
        if precip < 20:
            parts.append("with very little rainfall")
        elif precip < 80:
            parts.append(f"with moderate rainfall (~{precip:.0f} mm)")
        else:
            parts.append(f"with significant rainfall (~{precip:.0f} mm) — consider a rain jacket")
    return ". ".join(parts) + "."


async def _fetch_live(inp: LiveConditionsInput) -> LiveConditionsOutput:
    """Do the actual HTTP work — called only on cache miss."""
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as http:
        coords = await _geocode(inp.destination_name, http)
        if coords is None:
            return LiveConditionsOutput(
                weather_summary=f"Could not geocode '{inp.destination_name}'.",
            )

        lat, lon = coords
        climate = await _fetch_climate(lat, lon, inp.travel_month, http)

        avg_temp = climate["avg_temp_c"]
        precip = climate["precipitation_mm"]
        summary = _weather_summary(avg_temp, precip, inp.travel_month)

        # Amadeus flight check — optional, degrades gracefully when no key
        flight_available = False
        flight_cost: float | None = None
        if settings.amadeus_api_key:
            try:
                flight_available, flight_cost = await _check_flights(
                    inp.destination_name, inp.travel_month, settings, http
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("live_conditions.flights.error", error=str(exc))

        return LiveConditionsOutput(
            avg_temp_c=avg_temp,
            precipitation_mm=precip,
            weather_summary=summary,
            flight_available=flight_available,
            estimated_flight_cost_usd=flight_cost,
        )


async def _check_flights(
    destination: str,
    month: int,
    settings,
    http: httpx.AsyncClient,
) -> tuple[bool, float | None]:
    """Very lightweight Amadeus check — get an OAuth token then search one route."""
    token_resp = await http.post(
        "https://test.api.amadeus.com/v1/security/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": settings.amadeus_api_key,
            "client_secret": settings.amadeus_api_secret,
        },
        timeout=8.0,
    )
    token_resp.raise_for_status()
    token = token_resp.json()["access_token"]

    # Use the current year for the travel date estimate
    year = time.localtime().tm_year
    travel_date = f"{year}-{month:02d}-15"

    search_resp = await http.get(
        "https://test.api.amadeus.com/v2/shopping/flight-offers",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "originLocationCode": "JFK",
            "destinationLocationCode": destination[:3].upper(),
            "departureDate": travel_date,
            "adults": 1,
            "max": 1,
            "currencyCode": "USD",
        },
        timeout=10.0,
    )
    if search_resp.status_code != 200:
        return False, None

    offers = search_resp.json().get("data", [])
    if not offers:
        return False, None

    price = float(offers[0]["price"]["total"])
    return True, price


async def live_conditions(inp: LiveConditionsInput) -> LiveConditionsOutput | ToolError:
    """Return weather normals (and optional flight data) for a destination + month.

    Results are cached for 10 minutes. Returns ToolError on network failure.
    """
    cache_key = (inp.destination_name.lower(), inp.travel_month)

    async with _cache_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            log.info("live_conditions.cache_hit", destination=inp.destination_name)
            return cached

    try:
        result = await _fetch_live(inp)
        async with _cache_lock:
            _cache[cache_key] = result
        log.info(
            "live_conditions.fetched",
            destination=inp.destination_name,
            month=inp.travel_month,
            temp=result.avg_temp_c,
        )
        return result
    except httpx.TimeoutException as exc:
        log.warning("live_conditions.timeout", error=str(exc))
        return ToolError(tool=_TOOL_NAME, error=f"weather API timeout: {exc}")
    except httpx.NetworkError as exc:
        log.warning("live_conditions.network_error", error=str(exc))
        return ToolError(tool=_TOOL_NAME, error=f"weather API unreachable: {exc}")
    except Exception as exc:  # noqa: BLE001
        log.warning("live_conditions.error", error=str(exc))
        return ToolError(tool=_TOOL_NAME, error=str(exc))
