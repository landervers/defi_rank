# DeFi Yield Pool Scoring Service

A multi-dimensional yield pool ranking system built on the [DefiLlama Yields API](https://yields.llama.fi/pools), powered by FastAPI + Pandas. Each request fetches live data and returns the top 50 pools ranked by composite score.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the service

```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Available endpoints

| Purpose | URL |
|---------|-----|
| Health check | `GET http://localhost:8000/health` |
| Top 50 pools | `GET http://localhost:8000/top-pools` |
| Filter by chain | `GET http://localhost:8000/top-pools?chain=Ethereum` |
| Custom filters | `GET http://localhost:8000/top-pools?chain=Arbitrum&min_tvl=5000000&limit=20` |
| Interactive docs | `http://localhost:8000/docs` |

---

## Project Structure

```
defi_level/
├── main.py           # FastAPI application — routes and response models
├── scorer.py         # Core scoring engine (clean_data / score_pools / get_top_pools)
├── fetcher.py        # Async HTTP fetching layer with retry and timeout
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Scoring System

### Composite Formula

$$positive = 0.375 \times TVL_{score} + 0.375 \times APY_{stable} + 0.25 \times Confidence$$

$$S = positive \times \bigl(1 - 0.25 \times Risk\bigr) \times 100$$

$$S = \text{clip}(S,\ 0,\ 100)$$

> Positive weights sum to 1.0, giving a theoretical maximum of **100 points**. Risk acts as a multiplicative factor that reduces the score by at most 25%.

---

### Dimension 1 — TVL Score (37.5%)

**Purpose**: Measure liquidity depth without letting mega-protocols dominate the ranking.

**Calculation**:

```
tvl_log   = log10(tvlUsd)
tvl_score = p5-p95 percentile normalization of tvl_log  →  [0, 1]
```

**Why p5-p95 normalization?**
DeFi TVL follows a power-law distribution. Protocols like Lido hold $10B+, which compresses
all other pools toward zero under plain Min-Max normalization. By using the 5th–95th percentile
interval, every pool above the 95th percentile earns a perfect 1.0, and solid mid-cap protocols
($100M–$1B) receive fair, competitive scores.

---

### Dimension 2 — APY Stability Score (37.5%)

**Purpose**: Measure real, sustainable yield while penalizing inflated short-term incentives.

**Calculation**:

```
# Step 1: Sustainability cap
apy_cap = min(apy, apyMean30d × 2.0)     # use raw apy if no 30d mean available

# Step 2: Exponential saturation curve
apy_score = 1 − exp(−apy_cap / 4.0)
# Score table: 1%→0.22  3%→0.53  5%→0.71  8%→0.86  10%→0.92  15%→0.98

# Step 3: Unsustainability penalty multiplier
ratio      = apy / apyMean30d
multiplier = 1.0                           if ratio ≤ 2
           = max(0.5, 1 − (ratio−2) / 24) if ratio > 2

apy_score = apy_score × multiplier
```

**Key design decisions**:

- **2× cap**: An APY more than twice its 30-day mean is likely driven by short-term liquidity
  mining incentives with strong mean-reversion tendencies.
- **Exponential curve**: Unlike log-MinMax, this is an absolute scale independent of the
  current dataset distribution, so scores remain consistent across daily fetches.

---

### Dimension 3 — Confidence Score (25%)

**Purpose**: Leverage DefiLlama's ML prediction signals to reward stable, high-confidence pools.

**Data source**: `predictions` object — fields `binnedConfidence` and `predictedClass`.

**Calculation**:

```
binnedConfidence  1  →  base = 0.50  (low confidence — neutral baseline)
                  2  →  base = 0.75  (medium confidence)
                  3  →  base = 1.00  (high confidence)

predictedClass  "Stable"  →  bonus = +0.15
                "Up"      →  bonus = +0.05
                "Down"    →  bonus = −0.25

confidence_score = clip(base + bonus, 0, 1)
```

**Why baseline 0.5 instead of 0.0?**
Many newer pools are not yet covered by the DefiLlama ML model. Absence of prediction data
is not a negative signal — setting the baseline to 0.5 (neutral) prevents "missing data" from
being misread as "poor quality."

**Why "Stable" ranks above "Up"?**
For long-term yield positions, predictable stable returns are more valuable than anticipated
upward movements, which typically carry higher uncertainty.

---

### Dimension 4 — Risk Factor (reduces score by up to 25%)

**Purpose**: Lower the ranking of pools with extreme volatility, high impermanent loss risk,
or statistical anomalies.

**How it works**: Applied as `risk_factor = 1 − 0.25 × risk_penalty`. Even at maximum risk
(risk_penalty = 1.0), the pool retains 75% of its base score — preventing a single risk flag
from completely disqualifying an otherwise strong pool.

**Calculation**:

```
risk = 0.0

# Statistical outlier (detected by DefiLlama ML)
if outlier == True:
    risk += 0.4

# Impermanent loss risk level
ilRisk:  "no"=0   "low"=0.1   "medium"=0.2   "high"=0.35   "very high"=0.5
risk += IL_RISK_PENALTIES[ilRisk]

# 7-day APY volatility
if |apyPct7D| > 50%:  risk += 0.30
if |apyPct7D| > 20%:  risk += 0.15

risk_penalty = clip(risk, 0, 1)
risk_factor  = 1 − 0.25 × risk_penalty   →   ∈ [0.75, 1.0]
```

**Risk level thresholds**:

| risk_score | Level | Meaning |
|-----------|-------|---------|
| < 0.2 | Low | Stable pool, suitable for conservative strategies |
| 0.2 – 0.4 | Medium | Moderate risk; monitor IL and APY swings |
| ≥ 0.4 | High | High risk; allocate with caution |

---

## API Reference

### `GET /top-pools`

**Query parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Number of results to return (1–200) |
| `chain` | string | (none) | Filter by blockchain name (case-insensitive) |
| `min_tvl` | float | 1,000,000 | Minimum TVL in USD |

**Response fields**:

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | Composite rank |
| `pool_id` | string | DefiLlama pool UUID |
| `pool_name` | string | Token pair symbol, e.g. USDC-USDT |
| `project` | string | Protocol name |
| `chain` | string | Blockchain |
| `tvl` | float | TVL (USD) |
| `apy` | float | Current APY (%) |
| `apy_mean_30d` | float \| null | 30-day mean APY (%) |
| `score` | float | Composite score (0–100) |
| `risk_level` | string | Low / Medium / High |
| `predicted_class` | string | Stable / Up / Down |

**Example response (top 3)**:

```json
{
  "generated_at": "2026-04-01T08:00:00+00:00",
  "total_pools_analyzed": 7834,
  "filters_applied": {
    "chain": null,
    "min_tvl": 1000000,
    "limit": 50
  },
  "top_pools": [
    {
      "rank": 1,
      "pool_id": "747c1d2a-c668-4682-b9f7-7e00d7d2ad34",
      "pool_name": "USDC-USDT",
      "project": "curve",
      "chain": "Ethereum",
      "tvl": 452000000.0,
      "apy": 5.23,
      "apy_mean_30d": 4.98,
      "score": 88.4,
      "risk_level": "Low",
      "predicted_class": "Stable"
    },
    {
      "rank": 2,
      "pool_id": "a82e3b1c-4512-48cd-b9f0-1234abcd5678",
      "pool_name": "WETH-USDC",
      "project": "uniswap-v3",
      "chain": "Ethereum",
      "tvl": 312000000.0,
      "apy": 12.87,
      "apy_mean_30d": 11.2,
      "score": 84.7,
      "risk_level": "Medium",
      "predicted_class": "Stable"
    },
    {
      "rank": 3,
      "pool_id": "f3a9cc81-7b22-4e6d-a001-9876fedcba01",
      "pool_name": "stETH",
      "project": "lido",
      "chain": "Ethereum",
      "tvl": 9800000000.0,
      "apy": 3.95,
      "apy_mean_30d": 4.02,
      "score": 82.1,
      "risk_level": "Low",
      "predicted_class": "Stable"
    }
  ]
}
```

---

## Tuning the Weights

The current weight allocation (TVL 37.5% / APY 37.5% / Confidence 25%) is optimized for
**sustainable yield** strategies. Adjust the constants in `scorer.py` to suit different goals:

| Strategy | Suggested adjustment |
|----------|----------------------|
| Aggressive yield | Increase `APY_HALF_LIFE` to reward higher APY |
| Ultra-conservative | Increase risk penalty weight; lower `APY_HALF_LIFE` |
| New chain exploration | Reduce confidence weight; add chain diversity bonus |
| Institutional grade | Filter out `ilRisk >= high`; raise `min_tvl` to $10M+ |
