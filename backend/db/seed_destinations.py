"""Seed the destinations table from ml/destinations.csv.

Run with:
    $env:PYTHONPATH = "."; uv run python -m db.seed_destinations

Skips rows that already exist (by name) so it is safe to re-run.
"""

from __future__ import annotations

import asyncio
import csv
import uuid
from pathlib import Path

import structlog
from sqlalchemy import select

from core.config import get_settings
from db.engine import create_engine, make_session_factory
from db.models import Destination

log = structlog.get_logger()

CSV_PATH = Path(__file__).parent.parent / "ml" / "destinations.csv"

# CSV column → Destination field, with type coercion
_INT_FIELDS = {"peak_season_length_months", "unesco_sites_count", "outdoor_activity_score",
               "daily_cost_bucket", "visa_difficulty", "english_prevalence"}
_FLOAT_FIELDS = {"avg_temp_peak_season_c"}
_BOOL_FIELDS = {"coastal_access"}


def _coerce(field: str, value: str):
    if field in _INT_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    if field in _BOOL_FIELDS:
        return value.strip() in ("1", "true", "True", "yes")
    return value.strip()


async def seed() -> None:
    settings = get_settings()
    engine = create_engine(settings)
    session_factory = make_session_factory(engine)

    inserted = 0
    skipped = 0

    async with session_factory() as session:
        with CSV_PATH.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["destination"].strip()

                # Skip if already exists
                exists = await session.execute(select(Destination).where(Destination.name == name))
                if exists.scalar_one_or_none():
                    skipped += 1
                    continue

                # Extract country from "City, Country" format if present
                parts = name.split(",", 1)
                country = parts[1].strip() if len(parts) == 2 else None

                dest = Destination(
                    id=uuid.uuid4(),
                    name=name,
                    country=country,
                    climate_zone=_coerce("climate_zone", row["climate_zone"]),
                    avg_temp_peak_season_c=_coerce("avg_temp_peak_season_c", row["avg_temp_peak_season_c"]),
                    peak_season_length_months=_coerce("peak_season_length_months", row["peak_season_length_months"]),
                    terrain_primary=_coerce("terrain_primary", row["terrain_primary"]),
                    coastal_access=_coerce("coastal_access", row["coastal_access"]),
                    unesco_sites_count=_coerce("unesco_sites_count", row["unesco_sites_count"]),
                    outdoor_activity_score=_coerce("outdoor_activity_score", row["outdoor_activity_score"]),
                    daily_cost_bucket=_coerce("daily_cost_bucket", row["daily_cost_bucket"]),
                    accommodation_range=_coerce("accommodation_range", row["accommodation_range"]),
                    visa_difficulty=_coerce("visa_difficulty", row["visa_difficulty"]),
                    english_prevalence=_coerce("english_prevalence", row["english_prevalence"]),
                    tourism_maturity=_coerce("tourism_maturity", row["tourism_maturity"]),
                    hand_label=row["travel_style"].strip(),
                )
                session.add(dest)
                inserted += 1

        await session.commit()

    await engine.dispose()
    log.info("seed.done", inserted=inserted, skipped=skipped)
    print(f"Done — {inserted} inserted, {skipped} already existed.")


if __name__ == "__main__":
    asyncio.run(seed())
