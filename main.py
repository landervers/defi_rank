"""
DeFi Yield Pool Scoring Service — FastAPI entry point.

Routes:
  GET /health              — Service health check
  GET /top-pools           — Top N yield pools by composite score, with profile/tier filters
  GET /pool/{pool_id}      — Full detail for a single pool by DefiLlama UUID

Start:
  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import math
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
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
        "Fetches the full DefiLlama yield pool dataset and ranks pools using an "
        "8-dimension scoring model: TVL depth, APY quality, risk-adjusted return, "
        "prediction confidence, asset safety, data maturity, plus 6 classification "
        "dimension labels for user-profile matching."
    ),
    version="2.0.0",
)


# ── Response Models ────────────────────────────────────────────────────────────

class PoolItem(BaseModel):
    """Standard pool entry returned by /top-pools."""

    rank: int = Field(description="Composite rank (1 = highest score)")
    pool_id: str = Field(description="DefiLlama pool UUID")
    pool_name: str = Field(description="Token pair / asset symbol, e.g. USDC-USDT")
    project: str = Field(description="Protocol name, e.g. curve")
    chain: str = Field(description="Blockchain, e.g. Ethereum")
    tvl: float = Field(description="Current TVL in USD")
    apy: float = Field(description="Current APY (%)")
    apy_mean_30d: Optional[float] = Field(None, description="30-day mean APY (%)")

    # Composite score & grade
    score: float = Field(description="Composite score (0–100)")
    grade_label: str = Field(description="Grade: S / A / B / C / D")
    risk_level: str = Field(description="Risk level: Low / Medium / High")
    pool_tier: str = Field(description="Exclusivity tier: free / premium / elite")

    # DefiLlama prediction
    predicted_class: str = Field(description="ML trend prediction, e.g. Stable/Up")

    # 8-Dimension classification labels
    volatility_tier: str = Field(description="D1 sigma-based: low / medium / high / unknown")
    apy_tier: str = Field(description="D2 APY absolute value: low / medium / high")
    strategy_complexity: str = Field(description="D3 exposure type: simple / medium / complex")
    capital_tier: str = Field(description="D4 TVL gate: retail / standard / whale")
    asset_tier: str = Field(description="D6 asset type: conservative / balanced / aggressive")
    security_tier: str = Field(description="D7 security proxy: high / medium / experimental")
    recommended_automation: str = Field(description="D5 AI execution depth (reserved)")
    profile_tags: List[str] = Field(
        description="Compatible user profiles: conservative / balanced / aggressive / whale"
    )


class PoolDetail(PoolItem):
    """Extended pool entry returned by /pool/{pool_id} — includes all raw metrics."""

    # Auxiliary computed metrics
    sustainability_ratio: Optional[float] = Field(
        None, description="apyBase / apy — organic yield share (null if apyBase unavailable)"
    )
    sharpe_score: Optional[float] = Field(
        None, description="Normalized Sharpe-like score: mu/(sigma+0.01)"
    )
    data_maturity: Optional[float] = Field(
        None, description="min(count, 365) / 365 — historical data completeness [0, 1]"
    )
    predicted_probability: Optional[float] = Field(
        None, description="ML model prediction probability (0–100)"
    )

    # Internal scoring components
    tvl_score: Optional[float] = Field(None, description="TVL dimension score [0, 1]")
    apy_score: Optional[float] = Field(None, description="APY quality dimension score [0, 1]")
    risk_adjusted_score: Optional[float] = Field(
        None, description="Risk-adjusted (Sharpe) dimension score [0, 1]"
    )
    confidence_score: Optional[float] = Field(
        None, description="Confidence dimension score [0, 1]"
    )
    asset_safety: Optional[float] = Field(
        None, description="Asset safety dimension score [0, 1]"
    )
    risk_penalty: Optional[float] = Field(
        None, description="Risk penalty factor [0, 1] (higher = more penalised)"
    )

    # Raw API fields surfaced for transparency
    sigma: Optional[float] = Field(None, description="Historical APY standard deviation")
    mu: Optional[float] = Field(None, description="Historical APY mean")
    apy_base: Optional[float] = Field(None, description="Base APY from protocol (no incentives)")
    apy_reward: Optional[float] = Field(None, description="Reward token APY")
    volume_usd_7d: Optional[float] = Field(None, description="7-day trading volume in USD")
    stablecoin: Optional[bool] = Field(None, description="Whether underlying asset is a stablecoin")
    exposure: Optional[str] = Field(None, description="single / multi asset exposure")
    il_risk: Optional[str] = Field(None, description="Impermanent loss risk level")
    count: Optional[int] = Field(None, description="Number of historical data points")


class TopPoolsResponse(BaseModel):
    generated_at: str = Field(description="UTC timestamp when this response was generated")
    total_pools_analyzed: int = Field(description="Number of valid pools after cleaning")
    filters_applied: dict = Field(description="Active query filters for this request")
    top_pools: List[PoolItem]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _nan_to_none(value) -> Optional[float]:
    """Convert NaN / inf to None for JSON-safe serialization."""
    try:
        if value is None:
            return None
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _build_pool_item(rank: int, row) -> PoolItem:
    return PoolItem(
        rank=rank,
        pool_id=str(row.get("pool", "")),
        pool_name=str(row.get("symbol", "-")),
        project=str(row.get("project", "-")),
        chain=str(row.get("chain", "-")),
        tvl=round(float(row["tvlUsd"]), 2),
        apy=round(float(row["apy"]), 4),
        apy_mean_30d=_nan_to_none(row.get("apyMean30d")),
        score=float(row["score"]),
        grade_label=str(row["grade_label"]),
        risk_level=str(row["risk_level"]),
        pool_tier=str(row["pool_tier"]),
        predicted_class=str(row.get("predictedClass", "Stable")),
        volatility_tier=str(row["volatility_tier"]),
        apy_tier=str(row["apy_tier"]),
        strategy_complexity=str(row["strategy_complexity"]),
        capital_tier=str(row["capital_tier"]),
        asset_tier=str(row["asset_tier"]),
        security_tier=str(row["security_tier"]),
        recommended_automation=str(row["recommended_automation"]),
        profile_tags=list(row["profile_tags"]),
    )


def _build_pool_detail(rank: int, row) -> PoolDetail:
    count_val = _nan_to_none(row.get("count"))
    return PoolDetail(
        **_build_pool_item(rank, row).model_dump(),
        sustainability_ratio=_nan_to_none(row.get("sustainability_ratio")),
        sharpe_score=_nan_to_none(row.get("sharpe_score")),
        data_maturity=_nan_to_none(row.get("data_maturity")),
        predicted_probability=_nan_to_none(row.get("predictedProbability")),
        tvl_score=_nan_to_none(row.get("tvl_score")),
        apy_score=_nan_to_none(row.get("apy_score")),
        risk_adjusted_score=_nan_to_none(row.get("risk_adjusted_score")),
        confidence_score=_nan_to_none(row.get("confidence_score")),
        asset_safety=_nan_to_none(row.get("asset_safety")),
        risk_penalty=_nan_to_none(row.get("risk_penalty")),
        sigma=_nan_to_none(row.get("sigma")),
        mu=_nan_to_none(row.get("mu")),
        apy_base=_nan_to_none(row.get("apyBase")),
        apy_reward=_nan_to_none(row.get("apyReward")),
        volume_usd_7d=_nan_to_none(row.get("volumeUsd7d")),
        stablecoin=bool(row.get("stablecoin", False)),
        exposure=str(row.get("exposure", "single")),
        il_risk=str(row.get("ilRisk", "no")),
        count=int(count_val) if count_val is not None else None,
    )


async def _fetch_and_score():
    """Shared pipeline: fetch → clean → score. Raises HTTPException on failure."""
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

    return score_pools(df_clean)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Service health check")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get(
    "/top-pools",
    response_model=TopPoolsResponse,
    summary="Get top-ranked yield pools by composite score",
    description=(
        "Fetches the full DefiLlama yield pool dataset, applies 8-dimension scoring, "
        "then returns the top N results.  Supports filtering by grade level, all 6 "
        "dimension labels, user profile, pool tier, chain, and TVL.  "
        "Use `sort_by` to rank by a specific metric instead of composite score."
    ),
)
async def top_pools(
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of results (default 50, max 200)",
    ),
    chain: Optional[str] = Query(
        default=None,
        description="Filter by blockchain (case-insensitive), e.g. Ethereum, Arbitrum",
    ),
    min_tvl: float = Query(
        default=MIN_TVL_USD,
        ge=0,
        description="Minimum TVL threshold in USD (default 1,000,000)",
    ),
    profile: Optional[str] = Query(
        default=None,
        description="User profile filter: conservative | balanced | aggressive | whale",
    ),
    tier: Optional[str] = Query(
        default=None,
        description="Pool exclusivity filter: free | premium | elite",
    ),
    grade: Optional[str] = Query(
        default=None,
        description=(
            "Grade label filter: S | A | B | C | D.  "
            "Accepts comma-separated values, e.g. S,A"
        ),
    ),
    volatility_tier: Optional[str] = Query(
        default=None,
        description="D1 volatility filter: low | medium | high | unknown",
    ),
    apy_tier: Optional[str] = Query(
        default=None,
        description="D2 APY range filter: low (< 8%) | medium (8–25%) | high (> 25%)",
    ),
    strategy_complexity: Optional[str] = Query(
        default=None,
        description="D3 strategy filter: simple | medium | complex",
    ),
    capital_tier: Optional[str] = Query(
        default=None,
        description="D4 capital threshold filter: retail (< $50M) | standard | whale (> $500M)",
    ),
    asset_tier: Optional[str] = Query(
        default=None,
        description="D6 asset type filter: conservative | balanced | aggressive",
    ),
    security_tier: Optional[str] = Query(
        default=None,
        description="D7 security proxy filter: high | medium | experimental",
    ),
    sort_by: str = Query(
        default="score",
        description=(
            "Sort results by: score | apy | tvl | "
            "sustainability_ratio | sharpe_score | data_maturity"
        ),
    ),
):
    df_scored = await _fetch_and_score()
    total_analyzed = len(df_scored)

    df_top = get_top_pools(
        df_scored,
        limit=limit,
        chain=chain,
        min_tvl=min_tvl,
        profile=profile,
        tier=tier,
        grade=grade,
        volatility_tier=volatility_tier,
        apy_tier=apy_tier,
        strategy_complexity=strategy_complexity,
        capital_tier=capital_tier,
        asset_tier=asset_tier,
        security_tier=security_tier,
        sort_by=sort_by,
    )

    pools_out: List[PoolItem] = [
        _build_pool_item(int(rank), row) for rank, row in df_top.iterrows()
    ]

    return TopPoolsResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_pools_analyzed=total_analyzed,
        filters_applied={
            "chain": chain,
            "min_tvl": min_tvl,
            "limit": limit,
            "profile": profile,
            "tier": tier,
            "grade": grade,
            "volatility_tier": volatility_tier,
            "apy_tier": apy_tier,
            "strategy_complexity": strategy_complexity,
            "capital_tier": capital_tier,
            "asset_tier": asset_tier,
            "security_tier": security_tier,
            "sort_by": sort_by,
        },
        top_pools=pools_out,
    )


@app.get(
    "/pool/{pool_id}",
    response_model=PoolDetail,
    summary="Get full detail for a single pool",
    description=(
        "Returns all scoring components, 8-dimension labels, auxiliary metrics, "
        "and raw API fields for the pool identified by its DefiLlama UUID.  "
        "The rank shown is relative to the full scored dataset."
    ),
)
async def get_pool_detail(pool_id: str):
    df_scored = await _fetch_and_score()

    match = df_scored[df_scored["pool"].astype(str) == pool_id]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Pool '{pool_id}' not found or does not meet minimum TVL / APY criteria.",
        )

    # Rank within the full scored dataset (score descending)
    df_ranked = df_scored.sort_values("score", ascending=False).reset_index(drop=True)
    df_ranked.index += 1
    pool_rank_series = df_ranked[df_ranked["pool"].astype(str) == pool_id].index
    pool_rank = int(pool_rank_series[0]) if not pool_rank_series.empty else 0

    row = match.iloc[0]
    return _build_pool_detail(pool_rank, row)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
