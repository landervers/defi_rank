"""
DeFi Yield Pool Scoring Service — FastAPI entry point.

Routes:
  GET /health     — Service health check
  GET /top-pools  — Return the top N yield pools by composite score

Start:
  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fetcher import fetch_pools
from scorer import MIN_TVL_USD, clean_data, get_top_pools, score_pools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeFi Yield Pool Scoring Service",
    description=(
        "Fetches the full DefiLlama yield pool dataset and ranks pools using a "
        "four-dimensional scoring model: TVL depth, APY sustainability, "
        "prediction confidence, and risk penalty."
    ),
    version="1.0.0",
)


# ── Response Models ──────────────────────────────────────────────────────────

class PoolItem(BaseModel):
    rank: int = Field(description="Composite rank (1 = highest score)")
    pool_id: str = Field(description="DefiLlama pool UUID")
    pool_name: str = Field(description="Token pair symbol, e.g. USDC-USDT")
    project: str = Field(description="Protocol name, e.g. curve")
    chain: str = Field(description="Blockchain, e.g. Ethereum")
    tvl: float = Field(description="Current TVL in USD")
    apy: float = Field(description="Current APY (%)")
    apy_mean_30d: Optional[float] = Field(None, description="30-day mean APY (%)")
    score: float = Field(description="Composite score (0–100)")
    risk_level: str = Field(description="Risk level: Low / Medium / High")
    predicted_class: str = Field(description="DefiLlama prediction trend: Stable / Up / Down")


class TopPoolsResponse(BaseModel):
    generated_at: str = Field(description="UTC timestamp when this response was generated")
    total_pools_analyzed: int = Field(description="Number of valid pools after cleaning")
    filters_applied: dict = Field(description="Active query filters for this request")
    top_pools: list[PoolItem]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", summary="Service health check")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get(
    "/top-pools",
    response_model=TopPoolsResponse,
    summary="Get top-ranked yield pools by composite score",
    description=(
        "Fetches the full DefiLlama yield pool dataset, applies cleaning and "
        "multi-dimensional scoring, then returns the top N results sorted by "
        "composite score descending. Supports optional chain and TVL filtering."
    ),
)
async def top_pools(
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of results to return (default 50, max 200)",
    ),
    chain: Optional[str] = Query(
        default=None,
        description="Filter by blockchain (case-insensitive), e.g. Ethereum, BSC, Arbitrum",
    ),
    min_tvl: float = Query(
        default=MIN_TVL_USD,
        ge=0,
        description="Minimum TVL threshold in USD (default 1,000,000)",
    ),
):
    try:
        raw = await fetch_pools()
    except RuntimeError as exc:
        logger.error("Data fetch failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream data source error: {exc}") from exc

    try:
        df_clean = clean_data(raw)
    except ValueError as exc:
        logger.error("Data cleaning failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Data cleaning error: {exc}") from exc

    total_analyzed = len(df_clean)

    df_scored = score_pools(df_clean)
    df_top = get_top_pools(df_scored, limit=limit, chain=chain, min_tvl=min_tvl)

    if df_top.empty:
        return TopPoolsResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_pools_analyzed=total_analyzed,
            filters_applied={"chain": chain, "min_tvl": min_tvl, "limit": limit},
            top_pools=[],
        )

    pools_out: list[PoolItem] = []
    for rank, row in df_top.iterrows():
        apy_mean = row.get("apyMean30d")
        pools_out.append(
            PoolItem(
                rank=int(rank),
                pool_id=str(row.get("pool", "")),
                pool_name=str(row.get("symbol", "-")),
                project=str(row.get("project", "-")),
                chain=str(row.get("chain", "-")),
                tvl=round(float(row["tvlUsd"]), 2),
                apy=round(float(row["apy"]), 4),
                apy_mean_30d=round(float(apy_mean), 4) if apy_mean is not None and not __import__("math").isnan(float(apy_mean)) else None,
                score=float(row["score"]),
                risk_level=str(row["risk_level"]),
                predicted_class=str(row.get("predictedClass", "Stable")),
            )
        )

    return TopPoolsResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_pools_analyzed=total_analyzed,
        filters_applied={"chain": chain, "min_tvl": min_tvl, "limit": limit},
        top_pools=pools_out,
    )


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
