"""
Core scoring engine — multi-dimensional quantitative scoring for DefiLlama yield pools.

Scoring dimensions:
  1. TVL Score        (37.5%) — log10(tvlUsd) with p5-p95 percentile normalization
  2. APY Stability    (37.5%) — exponential saturation curve, half-life 4%, rewards sustainable yield
  3. Confidence Score (25%)   — based on predictions.binnedConfidence / predictedClass
  4. Risk Factor      (up to -25%) — outlier flag + ilRisk level + 7-day APY volatility

Composite formula (positive weights sum to 1.0; risk applied as a multiplicative factor):
  positive = 0.375 × TVL + 0.375 × APY_Stable + 0.25 × Confidence
  S = positive × (1 − 0.25 × Risk) × 100
  S = clip(S, 0, 100)

Design notes:
  - TVL uses p5-p95 interval normalization: pools above the 95th percentile all score 1.0,
    preventing giants like Lido from suppressing mid-tier protocols.
  - APY uses an exponential saturation curve: 5% → 0.71, 10% → 0.92, rewarding real
    sustainable DeFi yields rather than unsustainable farm incentives.
  - Confidence baseline is 0.5 (neutral); missing prediction data is not a negative signal.
  - Risk acts as a multiplier, reducing the score by at most 25%.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_TVL_USD = 1_000_000
MIN_APY = 0.0

IL_RISK_PENALTIES: Dict[str, float] = {
    "no": 0.0,
    "low": 0.1,
    "medium": 0.2,
    "high": 0.35,
    "very high": 0.5,
}

PRED_CLASS_BONUS: Dict[str, float] = {
    "Stable": 0.15,
    "Up": 0.05,
    "Down": -0.25,
}


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def clean_data(raw_pools: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the raw pool list to a DataFrame and apply data cleaning:
      - Drop pools with tvlUsd < 1,000,000 (insufficient liquidity)
      - Drop pools with apy <= 0 (invalid / inactive)
      - Flatten the nested ``predictions`` object into separate columns
      - Fill missing values for key numeric fields
    """
    if not raw_pools:
        raise ValueError("Raw pool data is empty; cannot proceed with cleaning.")

    df = pd.DataFrame(raw_pools)
    total_before = len(df)

    df["tvlUsd"] = pd.to_numeric(df.get("tvlUsd", 0), errors="coerce").fillna(0)
    df["apy"] = pd.to_numeric(df.get("apy", 0), errors="coerce").fillna(0)

    df = df[df["tvlUsd"] >= MIN_TVL_USD].copy()
    df = df[df["apy"] > MIN_APY].copy()

    # Flatten the nested predictions object
    df["binnedConfidence"] = df["predictions"].apply(
        lambda p: _safe_get(p, "binnedConfidence", 1)
    )
    df["predictedClass"] = df["predictions"].apply(
        lambda p: _safe_get(p, "predictedClass", "Stable")
    )
    df["outlier"] = df["predictions"].apply(
        lambda p: bool(_safe_get(p, "outlier", False))
    )

    # Fill optional numeric fields
    df["apyMean30d"] = pd.to_numeric(df.get("apyMean30d", np.nan), errors="coerce")
    df["apyPct7D"] = pd.to_numeric(df.get("apyPct7D", 0), errors="coerce").fillna(0)
    df["ilRisk"] = df.get("ilRisk", "no").fillna("no").str.lower()

    logger.info(
        "Data cleaning complete: %d → %d records (%d dropped).",
        total_before,
        len(df),
        total_before - len(df),
    )
    return df.reset_index(drop=True)


def _normalize_p95(series: pd.Series) -> pd.Series:
    """
    Percentile normalization using the p5-p95 interval, clipped to [0, 1].
    Values above p95 are all mapped to 1.0, preventing a single extreme outlier
    from compressing the rest of the distribution toward zero.
    """
    p5 = series.quantile(0.05)
    p95 = series.quantile(0.95)
    if p95 - p5 < 1e-9:
        return pd.Series(0.5, index=series.index)
    return ((series - p5) / (p95 - p5)).clip(0, 1)


def _compute_tvl_score(df: pd.DataFrame) -> pd.Series:
    """
    TVL Score (weight 37.5%):
      tvl_log   = log10(tvlUsd)
      tvl_score = p5-p95 normalization of tvl_log

    Using p5-p95 percentile normalization instead of global Min-Max:
    - All pools above the 95th-percentile TVL (e.g. Lido at $10B+) receive a perfect 1.0.
    - Eliminates score suppression of solid mid-cap protocols ($100M–$1B) by outlier giants.
    - Pools near the minimum threshold ($1M TVL) receive ~0.
    """
    tvl_log = np.log10(df["tvlUsd"].clip(lower=1))
    return _normalize_p95(tvl_log)


APY_HALF_LIFE = 4.0  # Half-life of the APY saturation curve (%); 4% APY maps to ~0.632


def _compute_apy_stability_score(df: pd.DataFrame) -> pd.Series:
    """
    APY Stability Score (weight 37.5%):

      Step 1 — Sustainability cap:
        apy_cap = min(apy, apyMean30d × 2.0)
        If the current APY exceeds twice its 30-day mean, it is considered
        unsustainable and is capped at that upper limit.

      Step 2 — Exponential saturation curve (replaces log-MinMax):
        score = 1 − exp(−apy_cap / APY_HALF_LIFE)
        Score table: 1%→0.22  3%→0.53  5%→0.71  8%→0.86  10%→0.92  15%→0.98
        This naturally rewards the real DeFi sustainable yield range (3-15%) and
        eliminates cross-batch score drift caused by Min-Max normalization.

      Step 3 — Unsustainability penalty multiplier (works with Step 1):
        ratio     = apy / apyMean30d
        multiplier = 1.0                          if ratio ≤ 2
                   = max(0.5, 1 − (ratio−2) / 24) if 2 < ratio ≤ 14
                   = 0.5                          if ratio > 14
    """
    has_mean = df["apyMean30d"].notna() & (df["apyMean30d"] > 0)

    apy_cap = df["apy"].copy()
    apy_cap[has_mean] = df.loc[has_mean, "apy"].clip(
        upper=df.loc[has_mean, "apyMean30d"] * 2.0
    )

    apy_score = 1.0 - np.exp(-apy_cap / APY_HALF_LIFE)

    ratio = pd.Series(1.0, index=df.index)
    ratio[has_mean] = df.loc[has_mean, "apy"] / df.loc[has_mean, "apyMean30d"]

    multiplier = ratio.apply(
        lambda r: 1.0 if r <= 2 else max(0.5, 1.0 - (r - 2) / 24)
    )

    return (apy_score * multiplier).clip(0, 1)


def _compute_confidence_score(df: pd.DataFrame) -> pd.Series:
    """
    Confidence Score (weight 25%):
      - binnedConfidence 1/2/3 → base score 0.5 / 0.75 / 1.0
        (Level-1 maps to 0.5 neutral baseline rather than 0.0 to avoid
         penalizing pools not yet covered by the DefiLlama ML model.)
      - predictedClass bonus: Stable +0.15,  Up +0.05,  Down -0.25
      - Result clipped to [0, 1]
    """
    conf_map = {1: 0.5, 2: 0.75, 3: 1.0}
    base = df["binnedConfidence"].map(conf_map).fillna(0.5)
    bonus = df["predictedClass"].map(PRED_CLASS_BONUS).fillna(0.0)
    return (base + bonus).clip(0, 1)


def _compute_risk_penalty(df: pd.DataFrame) -> pd.Series:
    """
    Risk Penalty (applied as a multiplicative factor, reducing score by at most 25%):
      - outlier == True  : +0.4  (statistical anomaly detected by DefiLlama ML)
      - ilRisk mapping   : no=0, low=0.1, medium=0.2, high=0.35, very high=0.5
      - |apyPct7D| > 50% : +0.30 (severe APY swing)
      - |apyPct7D| > 20% : +0.15 (moderate APY swing)
      - Result clipped to [0, 1]
    """
    risk = pd.Series(0.0, index=df.index)

    risk += df["outlier"].astype(float) * 0.4

    risk += df["ilRisk"].map(IL_RISK_PENALTIES).fillna(0.0)

    apy_vol = df["apyPct7D"].abs()
    risk += (apy_vol > 50).astype(float) * 0.3
    risk += ((apy_vol > 20) & (apy_vol <= 50)).astype(float) * 0.15

    return risk.clip(0, 1)


def _classify_risk_level(risk_score: float) -> str:
    if risk_score < 0.2:
        return "Low"
    if risk_score < 0.4:
        return "Medium"
    return "High"


def score_pools(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the four-dimensional scoring model on a cleaned DataFrame.
    Adds the following columns: tvl_score, apy_score, confidence_score,
    risk_penalty, score (0-100), risk_level.

    Composite formula:
      positive    = 0.375×TVL + 0.375×APY + 0.25×Confidence   ∈ [0, 1]
      risk_factor = 1 − 0.25×Risk                              ∈ [0.75, 1.0]
      S           = positive × risk_factor × 100               ∈ [0, 100]
    """
    df = df.copy()

    df["tvl_score"] = _compute_tvl_score(df)
    df["apy_score"] = _compute_apy_stability_score(df)
    df["confidence_score"] = _compute_confidence_score(df)
    df["risk_penalty"] = _compute_risk_penalty(df)

    positive = (
        0.375 * df["tvl_score"]
        + 0.375 * df["apy_score"]
        + 0.25 * df["confidence_score"]
    )
    risk_factor = (1.0 - 0.25 * df["risk_penalty"]).clip(0.75, 1.0)

    df["score"] = (positive * risk_factor * 100).clip(0, 100).round(2)
    df["risk_level"] = df["risk_penalty"].apply(_classify_risk_level)

    logger.info("Scoring complete. Score range: [%.2f, %.2f]", df["score"].min(), df["score"].max())
    return df


def get_top_pools(
    df: pd.DataFrame,
    limit: int = 50,
    chain: str | None = None,
    min_tvl: float = MIN_TVL_USD,
) -> pd.DataFrame:
    """
    Filter and rank the scored DataFrame, returning the top N pools.

    Args:
        df:      DataFrame already processed by ``score_pools()``.
        limit:   Maximum number of results to return (default 50).
        chain:   If provided, keep only pools on this chain (case-insensitive).
        min_tvl: Minimum TVL threshold in USD (default $1M).
    """
    result = df.copy()

    result = result[result["tvlUsd"] >= min_tvl]

    if chain:
        result = result[result["chain"].str.lower() == chain.lower()]

    result = result.sort_values("score", ascending=False).head(limit)
    result = result.reset_index(drop=True)
    result.index += 1
    result.index.name = "rank"

    return result
