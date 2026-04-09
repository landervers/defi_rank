"""
Core scoring engine — multi-dimensional quantitative scoring for DefiLlama yield pools.

Scoring dimensions (6 weighted components):
  1. TVL Score          (25%) — log10(tvlUsd) with p5-p95 percentile normalization
  2. APY Quality        (25%) — exponential saturation × sustainability ratio (apyBase/apy)
  3. Risk-Adjusted      (20%) — Sharpe-like score: mu/(sigma+0.01), p5-p95 normalized
  4. Confidence         (15%) — binnedConfidence tier blended with predictedProbability
  5. Asset Safety       (10%) — stablecoin + exposure + ilRisk composite
  6. Data Maturity      ( 5%) — min(count, 365) / 365

Composite formula:
  positive    = Σ(weight_i × dim_i)              ∈ [0, 1]
  risk_factor = clip(1 − 0.25 × risk_penalty, 0.75, 1.0)
  score       = clip(positive × risk_factor × 100, 0, 100)

8-Dimension classification labels (per plan):
  D1  volatility_tier       ∈ {low, medium, high, unknown}    — sigma thresholds
  D2  apy_tier              ∈ {low, medium, high}             — APY absolute value
  D3  strategy_complexity   ∈ {simple, medium, complex}       — exposure + rewardTokens
  D4  capital_tier          ∈ {retail, standard, whale}       — tvlUsd thresholds
  D5  recommended_automation = "reserved"                     — placeholder, not yet implemented
  D6  asset_tier            ∈ {conservative, balanced, aggressive}
  D7  security_tier         ∈ {high, medium, experimental}    — tvlUsd + count + outlier proxy
  D8  pool_tier             ∈ {free, premium, elite}          — mapped from grade_label

Grade labels: S (≥85) / A (70-85) / B (55-70) / C (40-55) / D (<40)
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

# Expanded to cover compound class strings returned by the DefiLlama ML model
PRED_CLASS_BONUS: Dict[str, float] = {
    "Stable": 0.15,
    "Stable/Up": 0.10,
    "Up": 0.05,
    "Stable/Down": -0.10,
    "Down": -0.25,
}

# Component weights for asset_safety score (sum to 1.0 when all maxed)
ASSET_SAFETY_IL_SCORES: Dict[str, float] = {
    "no": 0.3,
    "low": 0.2,
    "medium": 0.1,
    "high": 0.0,
    "very high": -0.1,
}


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _col(df: pd.DataFrame, name: str, default) -> pd.Series:
    """Return df[name] if the column exists, otherwise a Series filled with default."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def clean_data(raw_pools: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the raw pool list to a DataFrame and apply data cleaning:
      - Drop pools with tvlUsd < 1,000,000 (insufficient liquidity)
      - Drop pools with apy <= 0 (invalid / inactive)
      - Flatten the nested ``predictions`` object
      - Parse extended fields: sigma, mu, count, stablecoin, exposure,
        has_reward_tokens, predictedProbability, apyBase, apyReward, volumeUsd7d
    """
    if not raw_pools:
        raise ValueError("Raw pool data is empty; cannot proceed with cleaning.")

    df = pd.DataFrame(raw_pools)
    total_before = len(df)

    df["tvlUsd"] = pd.to_numeric(_col(df, "tvlUsd", 0), errors="coerce").fillna(0)
    df["apy"] = pd.to_numeric(_col(df, "apy", 0), errors="coerce").fillna(0)

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
    df["predictedProbability"] = pd.to_numeric(
        df["predictions"].apply(lambda p: _safe_get(p, "predictedProbability", None)),
        errors="coerce",
    )

    # Original optional numeric fields
    df["apyMean30d"] = pd.to_numeric(_col(df, "apyMean30d", np.nan), errors="coerce")
    df["apyPct7D"] = pd.to_numeric(_col(df, "apyPct7D", 0), errors="coerce").fillna(0)
    df["ilRisk"] = _col(df, "ilRisk", "no").fillna("no").str.lower()

    # Extended numeric fields for new scoring dimensions
    df["sigma"] = pd.to_numeric(_col(df, "sigma", np.nan), errors="coerce")
    df["mu"] = pd.to_numeric(_col(df, "mu", np.nan), errors="coerce")
    df["count"] = pd.to_numeric(_col(df, "count", np.nan), errors="coerce")
    df["apyBase"] = pd.to_numeric(_col(df, "apyBase", np.nan), errors="coerce")
    df["apyReward"] = pd.to_numeric(_col(df, "apyReward", np.nan), errors="coerce")
    df["volumeUsd7d"] = pd.to_numeric(_col(df, "volumeUsd7d", np.nan), errors="coerce")

    # Boolean / categorical fields
    df["stablecoin"] = _col(df, "stablecoin", False).fillna(False).astype(bool)
    df["exposure"] = _col(df, "exposure", "single").fillna("single").str.lower()

    # rewardTokens: treat None / empty list as "no rewards"
    df["has_reward_tokens"] = _col(df, "rewardTokens", None).apply(
        lambda x: bool(x) if isinstance(x, list) else False
    )

    logger.info(
        "Data cleaning complete: %d → %d records (%d dropped).",
        total_before,
        len(df),
        total_before - len(df),
    )
    return df.reset_index(drop=True)


# ── Normalization ─────────────────────────────────────────────────────────────

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


# ── Scoring Dimensions ────────────────────────────────────────────────────────

def _compute_tvl_score(df: pd.DataFrame) -> pd.Series:
    """
    TVL Score (weight 25%):
      tvl_log   = log10(tvlUsd)
      tvl_score = p5-p95 normalization of tvl_log

    All pools above the 95th-percentile TVL receive 1.0; pools near the $1M
    minimum receive ~0.  Prevents giant protocols from suppressing mid-tier ones.
    """
    tvl_log = np.log10(df["tvlUsd"].clip(lower=1))
    return _normalize_p95(tvl_log)


APY_HALF_LIFE = 4.0  # Half-life of the APY saturation curve (%); 4% APY maps to ~0.632


def _compute_apy_stability_score(df: pd.DataFrame) -> pd.Series:
    """
    Base APY stability score (used internally by _compute_apy_quality_score):

      Step 1 — Sustainability cap:
        apy_cap = min(apy, apyMean30d × 2.0)
        Only applied when count >= 30 so the 30-day mean is based on sufficient
        history.  For newer pools (count < 30) the current APY is used as-is,
        preventing a stale or sparse mean from misclassifying recently repriced
        institutional products (e.g. a fund that just raised its yield rate).

      Step 2 — Exponential saturation curve:
        score = 1 − exp(−apy_cap / APY_HALF_LIFE)
        1%→0.22  3%→0.53  5%→0.71  8%→0.86  10%→0.92  15%→0.98

      Step 3 — Unsustainability penalty multiplier:
        ratio     = apy / apyMean30d
        multiplier = 1.0                          if ratio ≤ 2
                   = max(0.5, 1 − (ratio−2) / 24) if 2 < ratio ≤ 14
                   = 0.5                          if ratio > 14
    """
    # Require at least 30 data points before trusting apyMean30d as a signal
    has_mean = (
        df["apyMean30d"].notna()
        & (df["apyMean30d"] > 0)
        & (df["count"].fillna(0) >= 30)
    )

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


def _compute_apy_quality_score(df: pd.DataFrame) -> pd.Series:
    """
    APY Quality Score (weight 25%):
    Extends the stability score with a sustainability factor derived from
    the apyBase/apy ratio (organic-yield share).

      sustainability_factor = 0.7 + 0.3 × sustainability_ratio
        - fully organic (ratio = 1.0): factor = 1.0 (no penalty)
        - pure incentive (ratio = 0.0): factor = 0.7 (30% reduction)
        - ratio unavailable (apyBase null): factor = 1.0 (neutral)

    Requires df["sustainability_ratio"] to be pre-computed in score_pools().
    """
    base = _compute_apy_stability_score(df)
    has_ratio = df["sustainability_ratio"].notna()
    factor = pd.Series(1.0, index=df.index)
    factor[has_ratio] = 0.7 + 0.3 * df.loc[has_ratio, "sustainability_ratio"]
    return (base * factor).clip(0, 1)


def _compute_risk_adjusted_score(df: pd.DataFrame) -> pd.Series:
    """
    Risk-Adjusted Score (weight 20%): Sharpe-like metric mu/(sigma+0.01).

    Uses the DefiLlama historical mean (mu) and standard deviation (sigma)
    to capture risk-adjusted return quality.  Normalized with p5-p95 across
    valid rows; pools missing mu or sigma receive a neutral 0.5.
    """
    has_both = df["mu"].notna() & df["sigma"].notna()
    result = pd.Series(0.5, index=df.index)

    if has_both.sum() < 2:
        return result

    sharpe_raw = df.loc[has_both, "mu"] / (df.loc[has_both, "sigma"] + 0.01)
    result[has_both] = _normalize_p95(sharpe_raw)
    return result.clip(0, 1)


def _compute_confidence_score(df: pd.DataFrame) -> pd.Series:
    """
    Confidence Score (weight 15%):
      - binnedConfidence 1/2/3 → tier base 0.5 / 0.75 / 1.0
      - When predictedProbability is available, blend:
          conf_base = 0.5 × tier_base + 0.5 × (predictedProbability / 100)
      - predictedClass bonus: Stable +0.15, Stable/Up +0.10, Up +0.05,
                              Stable/Down −0.10, Down −0.25
      - Result clipped to [0, 1]
    """
    conf_map = {1: 0.5, 2: 0.75, 3: 1.0}
    tier_base = df["binnedConfidence"].map(conf_map).fillna(0.5)

    has_prob = df["predictedProbability"].notna()
    prob_norm = (df["predictedProbability"] / 100.0).clip(0, 1)

    conf_base = tier_base.copy()
    conf_base[has_prob] = 0.5 * tier_base[has_prob] + 0.5 * prob_norm[has_prob]

    bonus = df["predictedClass"].map(PRED_CLASS_BONUS).fillna(0.0)
    return (conf_base + bonus).clip(0, 1)


def _compute_asset_safety_score(df: pd.DataFrame) -> pd.Series:
    """
    Asset Safety Score (weight 10%):
      stablecoin = True  → +0.4  (price-stable underlying)
      exposure   = single → +0.3  (no impermanent loss exposure)
      ilRisk component   → [−0.1, +0.3] per ASSET_SAFETY_IL_SCORES
      Result clipped to [0, 1].
    """
    stablecoin_score = df["stablecoin"].astype(float) * 0.4
    exposure_score = (df["exposure"] == "single").astype(float) * 0.3
    il_score = df["ilRisk"].map(ASSET_SAFETY_IL_SCORES).fillna(0.0)
    return (stablecoin_score + exposure_score + il_score).clip(0, 1)


def _compute_risk_penalty(df: pd.DataFrame) -> pd.Series:
    """
    Risk Penalty (multiplicative factor, reducing score by at most 25%):
      - outlier == True  : +0.4
      - ilRisk mapping   : no=0, low=0.1, medium=0.2, high=0.35, very high=0.5
      - |apyPct7D| > 50% : +0.30
      - |apyPct7D| > 20% : +0.15
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


# ── 8-Dimension Label Functions ───────────────────────────────────────────────

def _label_volatility_tier(sigma) -> str:
    """D1: Risk level based on historical APY standard deviation (sigma)."""
    if pd.isna(sigma):
        return "unknown"
    if sigma < 2:
        return "low"
    if sigma < 10:
        return "medium"
    return "high"


def _label_apy_tier(apy: float) -> str:
    """D2: Target yield tier based on absolute APY value."""
    if apy < 8:
        return "low"
    if apy < 25:
        return "medium"
    return "high"


def _label_strategy_complexity(row: pd.Series) -> str:
    """
    D3: Strategy complexity inferred from exposure type and reward tokens.
      simple  — single-asset lending/staking, no token incentives
      medium  — single-asset with reward token incentives (yield farming)
      complex — multi-asset LP or high IL-risk strategy
    """
    exposure = str(row.get("exposure", "single")).lower()
    il_risk = str(row.get("ilRisk", "no")).lower()
    has_rewards = bool(row.get("has_reward_tokens", False))

    if exposure == "multi" or il_risk in ("high", "very high"):
        return "complex"
    if has_rewards:
        return "medium"
    return "simple"


def _label_capital_tier(tvl: float) -> str:
    """D4: Capital threshold based on pool TVL."""
    if tvl >= 500_000_000:
        return "whale"
    if tvl >= 50_000_000:
        return "standard"
    return "retail"


def _label_asset_tier(row: pd.Series) -> str:
    """
    D6: Asset type coverage.
      conservative — stablecoin underlying
      balanced     — single-asset volatile with low/no IL risk
      aggressive   — multi-asset or medium/high IL risk
    """
    stablecoin = bool(row.get("stablecoin", False))
    exposure = str(row.get("exposure", "single")).lower()
    il_risk = str(row.get("ilRisk", "no")).lower()

    if stablecoin:
        return "conservative"
    if exposure == "single" and il_risk in ("no", "low"):
        return "balanced"
    return "aggressive"


def _label_security_tier(tvl: float, count, outlier: bool) -> str:
    """
    D7: Security & transparency proxy (audit data unavailable from API).
      high         — TVL > $100M AND count > 300 AND not outlier
      experimental — TVL < $10M OR count < 100 OR outlier
      medium       — everything else
    """
    if outlier:
        return "experimental"
    if tvl < 10_000_000:
        return "experimental"
    count_val = float(count) if not pd.isna(count) else 0.0
    if count_val < 100:
        return "experimental"
    if tvl >= 100_000_000 and count_val > 300:
        return "high"
    return "medium"


def _label_grade(score: float) -> str:
    """
    Composite grade: S / A / B / C / D.

    S threshold is 80 (not 85) because several scoring dimensions have practical
    ceilings below 1.0 (e.g. apy_quality for low-APY stablecoin pools tops out
    around 0.65-0.70), meaning the empirical maximum achievable score for
    conservative institutional pools is typically 82-88 rather than 100.
    Setting S at 80 targets approximately the top 1-3% of all qualified pools.
    """
    if score >= 80:
        return "S"
    if score >= 67:
        return "A"
    if score >= 53:
        return "B"
    if score >= 38:
        return "C"
    return "D"


def _label_pool_tier(grade: str) -> str:
    """D8: Pool exclusivity tier mapped from grade label."""
    if grade == "S":
        return "elite"
    if grade == "A":
        return "premium"
    return "free"


# ── Profile Matching ──────────────────────────────────────────────────────────

def _compute_profile_tags(row: pd.Series) -> List[str]:
    """
    Return the list of user profiles this pool is suitable for.
    All pools qualify for 'aggressive'; stricter profiles require meeting
    additional criteria across the 8 dimensions.

      aggressive  — all pools
      balanced    — moderate volatility, no outlier, medium/high security
      conservative— low volatility, stable asset, high security, simple strategy
      whale       — capital_tier = whale (TVL > $500M)
    """
    tags: List[str] = ["aggressive"]

    if (
        row.get("volatility_tier") in ("low", "medium")
        and row.get("security_tier") in ("high", "medium")
        and not row.get("outlier", False)
    ):
        tags.append("balanced")

    if (
        row.get("volatility_tier") == "low"
        and row.get("asset_tier") == "conservative"
        and row.get("security_tier") == "high"
        and row.get("strategy_complexity") == "simple"
    ):
        tags.append("conservative")

    if row.get("capital_tier") == "whale":
        tags.append("whale")

    return tags


# ── Main Scoring Pipeline ─────────────────────────────────────────────────────

def score_pools(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full scoring model on a cleaned DataFrame.

    Adds columns:
      Auxiliary metrics : sustainability_ratio, sharpe_score, data_maturity,
                          predicted_probability_norm
      Scoring components: tvl_score, apy_score, risk_adjusted_score,
                          confidence_score, asset_safety, risk_penalty
      Composite output  : score (0-100), risk_level, grade_label
      8-D labels        : volatility_tier, apy_tier, strategy_complexity,
                          capital_tier, asset_tier, security_tier,
                          recommended_automation (reserved), pool_tier
      Profile matching  : profile_tags (list of compatible user profiles)
    """
    df = df.copy()

    # ── Auxiliary metrics (computed first; some are used in scoring dims) ──────
    has_base = df["apyBase"].notna() & (df["apy"] > 0)
    df["sustainability_ratio"] = pd.Series(np.nan, index=df.index)
    df.loc[has_base, "sustainability_ratio"] = (
        df.loc[has_base, "apyBase"] / df.loc[has_base, "apy"]
    ).clip(0, 1)

    # 90 days is sufficient to consider a pool "data-mature"; using 365 heavily
    # penalised recently-listed institutional products (e.g. BlackRock BUIDL at
    # count=12 on DefiLlama) that are genuinely low-risk but newly tracked.
    df["data_maturity"] = (
        df["count"].clip(upper=90) / 90.0
    ).fillna(0.0).clip(0, 1)

    df["predicted_probability_norm"] = (
        df["predictedProbability"] / 100.0
    ).clip(0, 1)

    # ── Scoring dimensions ────────────────────────────────────────────────────
    df["tvl_score"] = _compute_tvl_score(df)
    df["apy_score"] = _compute_apy_quality_score(df)
    df["risk_adjusted_score"] = _compute_risk_adjusted_score(df)
    df["sharpe_score"] = df["risk_adjusted_score"]
    df["confidence_score"] = _compute_confidence_score(df)
    df["asset_safety"] = _compute_asset_safety_score(df)
    df["risk_penalty"] = _compute_risk_penalty(df)

    # ── Composite score ───────────────────────────────────────────────────────
    positive = (
        0.25 * df["tvl_score"]
        + 0.25 * df["apy_score"]
        + 0.20 * df["risk_adjusted_score"]
        + 0.15 * df["confidence_score"]
        + 0.10 * df["asset_safety"]
        + 0.05 * df["data_maturity"]
    )
    risk_factor = (1.0 - 0.25 * df["risk_penalty"]).clip(0.75, 1.0)
    df["score"] = (positive * risk_factor * 100).clip(0, 100).round(2)
    df["risk_level"] = df["risk_penalty"].apply(_classify_risk_level)
    df["grade_label"] = df["score"].apply(_label_grade)

    # ── 8-Dimension labels (computed after score so grade_label is available) ──
    df["volatility_tier"] = df["sigma"].apply(_label_volatility_tier)
    df["apy_tier"] = df["apy"].apply(_label_apy_tier)
    df["strategy_complexity"] = df.apply(_label_strategy_complexity, axis=1)
    df["capital_tier"] = df["tvlUsd"].apply(_label_capital_tier)
    df["asset_tier"] = df.apply(_label_asset_tier, axis=1)
    df["security_tier"] = df.apply(
        lambda r: _label_security_tier(r["tvlUsd"], r["count"], r["outlier"]), axis=1
    )
    df["pool_tier"] = df["grade_label"].apply(_label_pool_tier)
    df["recommended_automation"] = "reserved"  # D5 placeholder — not yet implemented

    # ── Profile tags (requires all dimension labels to be ready) ─────────────
    df["profile_tags"] = df.apply(_compute_profile_tags, axis=1)

    logger.info(
        "Scoring complete. Score range: [%.2f, %.2f]", df["score"].min(), df["score"].max()
    )
    return df


# ── Filtering & Ranking ───────────────────────────────────────────────────────

def get_top_pools(
    df: pd.DataFrame,
    limit: int = 50,
    chain: str | None = None,
    min_tvl: float = MIN_TVL_USD,
    profile: str | None = None,
    tier: str | None = None,
    grade: str | None = None,
    volatility_tier: str | None = None,
    apy_tier: str | None = None,
    strategy_complexity: str | None = None,
    capital_tier: str | None = None,
    asset_tier: str | None = None,
    security_tier: str | None = None,
    sort_by: str = "score",
) -> pd.DataFrame:
    """
    Filter and rank the scored DataFrame, returning the top N pools.

    Args:
        df:                  DataFrame already processed by ``score_pools()``.
        limit:               Maximum number of results to return (default 50).
        chain:               Filter by blockchain name (case-insensitive).
        min_tvl:             Minimum TVL threshold in USD (default $1M).
        profile:             Filter by user profile: conservative | balanced |
                             aggressive | whale.
        tier:                Filter by pool exclusivity: free | premium | elite.
        grade:               Filter by grade label: S | A | B | C | D.
                             Accepts comma-separated values, e.g. "S,A".
        volatility_tier:     D1 filter: low | medium | high | unknown.
        apy_tier:            D2 filter: low | medium | high.
        strategy_complexity: D3 filter: simple | medium | complex.
        capital_tier:        D4 filter: retail | standard | whale.
        asset_tier:          D6 filter: conservative | balanced | aggressive.
        security_tier:       D7 filter: high | medium | experimental.
        sort_by:             Column to sort by descending (default: score).
                             Supported: score | apy | tvl | sustainability_ratio |
                             sharpe_score | data_maturity.
    """
    _SORT_COLUMN_MAP = {
        "score": "score",
        "apy": "apy",
        "tvl": "tvlUsd",
        "sustainability_ratio": "sustainability_ratio",
        "sharpe_score": "sharpe_score",
        "data_maturity": "data_maturity",
    }

    result = df.copy()
    result = result[result["tvlUsd"] >= min_tvl]

    if chain:
        result = result[result["chain"].str.lower() == chain.lower()]

    if profile:
        result = result[
            result["profile_tags"].apply(lambda tags: profile.lower() in tags)
        ]

    if tier:
        result = result[result["pool_tier"] == tier.lower()]

    if grade:
        grades = {g.strip().upper() for g in grade.split(",")}
        result = result[result["grade_label"].isin(grades)]

    if volatility_tier:
        result = result[result["volatility_tier"] == volatility_tier.lower()]

    if apy_tier:
        result = result[result["apy_tier"] == apy_tier.lower()]

    if strategy_complexity:
        result = result[result["strategy_complexity"] == strategy_complexity.lower()]

    if capital_tier:
        result = result[result["capital_tier"] == capital_tier.lower()]

    if asset_tier:
        result = result[result["asset_tier"] == asset_tier.lower()]

    if security_tier:
        result = result[result["security_tier"] == security_tier.lower()]

    sort_col = _SORT_COLUMN_MAP.get(sort_by.lower(), "score")
    result = result.sort_values(sort_col, ascending=False, na_position="last").head(limit)
    result = result.reset_index(drop=True)
    result.index += 1
    result.index.name = "rank"

    return result
