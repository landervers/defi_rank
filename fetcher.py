"""
Async data fetching layer — retrieves the full pool dataset from the DefiLlama Yields API.
Includes timeout control, exponential-backoff retry logic, and response validation.
"""

import asyncio
import logging
from typing import List, Dict, Any

import httpx

logger = logging.getLogger(__name__)

YIELDS_API_URL = "https://yields.llama.fi/pools"
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds between retries (multiplied by attempt number)


async def fetch_pools() -> List[Dict[str, Any]]:
    """
    Asynchronously fetch all yield pool records from the DefiLlama API.

    Returns:
        List[Dict]: Raw pool objects corresponding to the ``data`` array in the API response.

    Raises:
        RuntimeError: When all retry attempts fail or the response format is unexpected.
    """
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info("Requesting DefiLlama Yields API (attempt %d)...", attempt)
                response = await client.get(YIELDS_API_URL)
                response.raise_for_status()

                payload = response.json()

                if not isinstance(payload, dict) or "data" not in payload:
                    raise ValueError(
                        f"Unexpected response format — expected object with 'data' key, "
                        f"got keys: {list(payload.keys())}"
                    )

                pools: List[Dict[str, Any]] = payload["data"]
                logger.info("Successfully fetched %d raw pool records.", len(pools))
                return pools

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF * attempt
                    logger.warning(
                        "Request failed (%s), retrying in %.1fs (%d/%d)...",
                        type(exc).__name__,
                        wait,
                        attempt,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("Max retries reached, giving up.")

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.error("HTTP status error: %s", exc)
                break

            except (ValueError, KeyError) as exc:
                last_exc = exc
                logger.error("Response parsing failed: %s", exc)
                break

    raise RuntimeError(f"Failed to fetch DefiLlama data: {last_exc}") from last_exc
