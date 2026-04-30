# Tool: live_conditions
# Input validated by schemas.tools.LiveConditionsInput before execution.
# Fetches current weather from Open-Meteo (free, no key) and flight estimates from Amadeus.
# All HTTP calls use httpx.AsyncClient with a timeout — no blocking I/O.
# Retries transient failures with exponential backoff (tenacity).
# On exhausted retries, returns a structured error dict — never raises into the agent loop.
