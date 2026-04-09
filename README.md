# DeFi Yield Pool Scoring Service

Fetches the complete DefiLlama yield pool dataset and scores every pool with a
**6-component composite score** plus **8 classification dimension labels**.  The
result supports precise matching to three user profiles (conservative / balanced /
aggressive) and three subscription tiers (free / premium / elite).

---

## Quick Start

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/top-pools` | Top-N pools ranked by composite score |
| `GET` | `/pool/{pool_id}` | Full detail for one pool by DefiLlama UUID |

---

## GET /top-pools — Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int (1–200) | 50 | Maximum results to return |
| `chain` | string | — | Filter by blockchain, e.g. `Ethereum`, `Arbitrum` (case-insensitive) |
| `min_tvl` | float | 1 000 000 | Minimum TVL threshold in USD |
| `profile` | string | — | User profile filter: `conservative` \| `balanced` \| `aggressive` \| `whale` |
| `tier` | string | — | Pool exclusivity filter: `free` \| `premium` \| `elite` |

### Example Requests

```bash
# Top 20 conservative pools on Ethereum
GET /top-pools?limit=20&profile=conservative&chain=Ethereum

# Elite-tier pools for whale-sized capital
GET /top-pools?limit=50&profile=whale&tier=elite&min_tvl=50000000

# High-yield aggressive pools
GET /top-pools?limit=30&profile=aggressive&tier=premium
```

---

## GET /pool/{pool_id}

Returns all scoring components, 8-dimension labels, auxiliary metrics, and raw
API fields for a single pool.

```bash
GET /pool/747c1d2a-c668-4682-b9f9-296708a3dd90
```

---

## Scoring Model

### Component Weights

| # | Dimension | Weight | Key Inputs |
|---|-----------|--------|------------|
| 1 | TVL Score | 25% | `tvlUsd` — log10, p5-p95 normalized |
| 2 | APY Quality | 25% | `apy`, `apyMean30d`, `apyBase` — stability curve × sustainability ratio |
| 3 | Risk-Adjusted | 20% | `mu`, `sigma` — Sharpe-like: `mu/(sigma+0.01)`, p5-p95 normalized |
| 4 | Confidence | 15% | `binnedConfidence`, `predictedProbability`, `predictedClass` |
| 5 | Asset Safety | 10% | `stablecoin`, `exposure`, `ilRisk` |
| 6 | Data Maturity | 5% | `count` — `min(count, 365) / 365` |

### Composite Formula

```
positive    = 0.25×TVL + 0.25×APY_Quality + 0.20×RiskAdj + 0.15×Confidence
            + 0.10×AssetSafety + 0.05×DataMaturity       ∈ [0, 1]

risk_factor = clip(1 − 0.25 × risk_penalty, 0.75, 1.0)

score       = clip(positive × risk_factor × 100, 0, 100)
```

### Risk Penalty Components

| Condition | Penalty |
|-----------|---------|
| `outlier = True` | +0.40 |
| `ilRisk = very high` | +0.50 |
| `ilRisk = high` | +0.35 |
| `ilRisk = medium` | +0.20 |
| `ilRisk = low` | +0.10 |
| `\|apyPct7D\| > 50%` | +0.30 |
| `20% < \|apyPct7D\| ≤ 50%` | +0.15 |

All components are summed then clipped to `[0, 1]`.  The final `risk_factor`
reduces the composite score by **at most 25%**.

### APY Quality Details

**Sustainability factor** (from `apyBase / apy`):

| Organic-yield share | Factor |
|---------------------|--------|
| 100% organic (`apyBase = apy`) | 1.00 |
| 50% organic | 0.85 |
| 0% organic (pure incentive) | 0.70 |
| `apyBase` unavailable | 1.00 (neutral) |

**Unsustainability penalty** (from `apy / apyMean30d` ratio):

| Ratio | Multiplier |
|-------|-----------|
| ≤ 2× | 1.00 |
| 2× – 14× | linear decay to 0.50 |
| > 14× | 0.50 |

---

## 8-Dimension Classification Labels

Each pool receives 8 classification labels that drive profile matching and UI
presentation.  Six are computed from API data; two are product-layer tags.

### Data-Layer Labels (computed from API)

#### D1 — `volatility_tier`  (Risk Profile)

| Value | Condition | Meaning |
|-------|-----------|---------|
| `low` | `sigma < 2` | Near principal-safe, <5% APY swing |
| `medium` | `2 ≤ sigma < 10` | Moderate volatility |
| `high` | `sigma ≥ 10` | High-volatility / possibly leveraged |
| `unknown` | `sigma` null | Insufficient history |

#### D2 — `apy_tier`  (Target APY)

| Value | APY Range |
|-------|-----------|
| `low` | < 8% |
| `medium` | 8%–25% |
| `high` | ≥ 25% |

#### D3 — `strategy_complexity`  (Strategy Complexity)

| Value | Condition |
|-------|-----------|
| `simple` | `exposure=single` + no reward tokens |
| `medium` | `exposure=single` + reward token incentives |
| `complex` | `exposure=multi` or `ilRisk` in {high, very high} |

#### D4 — `capital_tier`  (Capital Threshold & Liquidity)

| Value | TVL Threshold |
|-------|---------------|
| `retail` | < $50M |
| `standard` | $50M – $500M |
| `whale` | ≥ $500M |

#### D6 — `asset_tier`  (Asset Type Coverage)

| Value | Condition |
|-------|-----------|
| `conservative` | `stablecoin = true` |
| `balanced` | `stablecoin = false` + `exposure=single` + `ilRisk` in {no, low} |
| `aggressive` | `exposure=multi` or `ilRisk` in {medium, high, very high} |

#### D7 — `security_tier`  (Security & Transparency)

| Value | Condition |
|-------|-----------|
| `high` | TVL ≥ $100M **and** count > 300 **and** not outlier |
| `experimental` | TVL < $10M **or** count < 100 **or** outlier |
| `medium` | Everything else |

> Audit status is not available in the DefiLlama API; TVL scale, historical data
> volume, and the ML outlier flag are used as proxies.

### Product-Layer Labels

#### D5 — `recommended_automation`  (AI Execution Depth)

Reserved placeholder (`"reserved"`).  Future implementation will derive a
recommended execution mode from D1 + D3:

| Future value | Criteria |
|--------------|----------|
| `auto` | `strategy_complexity=simple` and `volatility_tier=low` |
| `assisted` | `strategy_complexity=medium` and `volatility_tier` in {low, medium} |
| `manual` | `strategy_complexity=complex` or `volatility_tier=high` |

#### D8 — `pool_tier`  (Exclusivity & Fee)

| Value | Grade | Access |
|-------|-------|--------|
| `free` | B / C / D | Public — no subscription required |
| `premium` | A | Requires Pro subscription |
| `elite` | S | Elite subscription + performance fee |

---

## Grade Labels

| Grade | Score Range | Suitable Profiles | Pool Tier |
|-------|-------------|-------------------|-----------|
| **S** | ≥ 80 | All | elite |
| **A** | 67 – 80 | balanced, aggressive | premium |
| **B** | 53 – 67 | aggressive | free |
| **C** | 38 – 53 | aggressive (speculative) | free |
| **D** | < 38 | Not recommended | free |

> S threshold is set at 80 rather than 85 because key dimensions such as APY quality
> for low-APY stablecoin pools have practical ceilings around 0.65–0.70, making the
> empirical maximum achievable score ~82–88 for conservative institutional pools.

---

## User Profile Matching Matrix

A pool earns a `profile_tags` entry when it meets all criteria for that profile.

| Dimension | `conservative` | `balanced` | `aggressive` |
|-----------|---------------|------------|-------------|
| D1 volatility | `low` only | `low` or `medium` | any |
| D2 APY target | `low` (3–8%) | `low` or `medium` | any |
| D3 complexity | `simple` only | `simple` or `medium` | any |
| D4 capital | `retail` | `retail` or `standard` | any |
| D6 asset type | `conservative` | `conservative` or `balanced` | any |
| D7 security | `high` only | `high` or `medium` | any |
| outlier flag | must be `false` | must be `false` | any |

> All pools are tagged `aggressive`.  The `whale` tag is added independently
> when `capital_tier = whale` (TVL ≥ $500M).

---

## Subscription Tiers

| Tier | Accessible Grades | `profile` Filter | `tier` Filter | Max Limit |
|------|-------------------|------------------|---------------|-----------|
| Free | B, C, D | — | `free` | 10 |
| Premium | A, B, C, D | all | `free`, `premium` | 200 |
| Elite | S, A, B, C, D | all | all | 200 |

*Subscription enforcement is handled at the API-gateway / auth layer, not inside
this service.  The `tier` query parameter is a data filter, not an access gate.*

---

## Response Fields

### `/top-pools` — `PoolItem`

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | Score rank (1 = best) |
| `pool_id` | string | DefiLlama pool UUID |
| `pool_name` | string | Asset symbol |
| `project` | string | Protocol name |
| `chain` | string | Blockchain |
| `tvl` | float | TVL in USD |
| `apy` | float | Current APY (%) |
| `apy_mean_30d` | float\|null | 30-day mean APY (%) |
| `score` | float | Composite score 0–100 |
| `grade_label` | string | S / A / B / C / D |
| `risk_level` | string | Low / Medium / High |
| `pool_tier` | string | free / premium / elite |
| `predicted_class` | string | ML trend, e.g. Stable/Up |
| `volatility_tier` | string | D1 label |
| `apy_tier` | string | D2 label |
| `strategy_complexity` | string | D3 label |
| `capital_tier` | string | D4 label |
| `asset_tier` | string | D6 label |
| `security_tier` | string | D7 label |
| `recommended_automation` | string | D5 label (reserved) |
| `profile_tags` | string[] | Compatible user profiles |

### `/pool/{pool_id}` — `PoolDetail` (extends `PoolItem`)

Additional fields:

| Field | Type | Description |
|-------|------|-------------|
| `sustainability_ratio` | float\|null | apyBase / apy |
| `sharpe_score` | float\|null | Normalized Sharpe-like score |
| `data_maturity` | float\|null | Historical data completeness 0–1 |
| `predicted_probability` | float\|null | ML prediction probability |
| `tvl_score` | float\|null | TVL dimension score |
| `apy_score` | float\|null | APY quality dimension score |
| `risk_adjusted_score` | float\|null | Risk-adjusted dimension score |
| `confidence_score` | float\|null | Confidence dimension score |
| `asset_safety` | float\|null | Asset safety dimension score |
| `risk_penalty` | float\|null | Risk penalty [0, 1] |
| `sigma` | float\|null | Historical APY std dev |
| `mu` | float\|null | Historical APY mean |
| `apy_base` | float\|null | Base APY (protocol-native) |
| `apy_reward` | float\|null | Reward token APY |
| `volume_usd_7d` | float\|null | 7-day trading volume |
| `stablecoin` | bool | Stablecoin underlying flag |
| `exposure` | string | single / multi |
| `il_risk` | string | IL risk level |
| `count` | int\|null | Historical data-point count |

---

## Example Response

```json
{
  "generated_at": "2026-04-07T10:00:00+00:00",
  "total_pools_analyzed": 1842,
  "filters_applied": {
    "chain": "Ethereum",
    "min_tvl": 1000000,
    "limit": 3,
    "profile": "conservative",
    "tier": null
  },
  "top_pools": [
    {
      "rank": 1,
      "pool_id": "747c1d2a-c668-4682-b9f9-296708a3dd90",
      "pool_name": "STETH",
      "project": "lido",
      "chain": "Ethereum",
      "tvl": 19742876695.0,
      "apy": 2.383,
      "apy_mean_30d": 2.511,
      "score": 88.14,
      "grade_label": "S",
      "risk_level": "Low",
      "pool_tier": "elite",
      "predicted_class": "Stable/Up",
      "volatility_tier": "low",
      "apy_tier": "low",
      "strategy_complexity": "simple",
      "capital_tier": "whale",
      "asset_tier": "balanced",
      "security_tier": "high",
      "recommended_automation": "reserved",
      "profile_tags": ["aggressive", "balanced", "whale"]
    }
  ]
}
```

---

## Project Structure

```
defi_level/
├── main.py          # FastAPI entry point, routes, response models
├── scorer.py        # 8-dimension scoring engine, label functions, profile matching
├── fetcher.py       # Async HTTP client for DefiLlama yields API
├── requirements.txt
└── README.md
```

---

## Tuning Reference

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `APY_HALF_LIFE` | `scorer.py` | 4.0 | Shifts APY saturation curve; lower = more aggressive slope |
| `MIN_TVL_USD` | `scorer.py` | 1 000 000 | Global TVL floor for pool inclusion |
| Component weights | `score_pools()` | see above | Adjust per-use-case emphasis |
| `capital_tier` thresholds | `_label_capital_tier()` | 50M / 500M | Tune for user AUM distribution |
| `security_tier` thresholds | `_label_security_tier()` | 10M / 100M | Tune TVL safety bar |

cd C:\Users\hhhho\Desktop\defi_level
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000