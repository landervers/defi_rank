"""Temporary score distribution analysis — safe to delete after use."""
import httpx, statistics, json

r = httpx.get("http://127.0.0.1:8000/top-pools?limit=200", timeout=90)
pools = r.json()["top_pools"]

scores = sorted([p["score"] for p in pools], reverse=True)
grades = {}
for p in pools:
    grades[p["grade_label"]] = grades.get(p["grade_label"], 0) + 1

print("=== Score Distribution (top 200) ===")
for label, idx in [("max", 0), ("p99", int(len(scores)*0.01)),
                   ("p95", int(len(scores)*0.05)), ("p90", int(len(scores)*0.10)),
                   ("p80", int(len(scores)*0.20)), ("p70", int(len(scores)*0.30)),
                   ("p50", int(len(scores)*0.50)), ("min", -1)]:
    print(f"  {label:4s}: {scores[idx]:.2f}")
print(f"  mean : {statistics.mean(scores):.2f}  stdev: {statistics.stdev(scores):.2f}")
print()

print("=== Current Grade Breakdown ===")
for g, c in sorted(grades.items()):
    print(f"  {g}: {c:3d} pools ({c/len(pools)*100:.1f}%)")
print()

print("=== Pools in 78-90 range (S-boundary zone) ===")
boundary = [(p["score"], p["grade_label"], p["pool_name"], p["project"]) 
            for p in pools if 78 <= p["score"] <= 90]
for sc, gl, nm, pr in sorted(boundary, reverse=True):
    print(f"  {gl}  {sc:.2f}  {nm} / {pr}")
print()

print("=== Projected grade counts under new threshold S>=80 ===")
new_grades = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}
for sc in scores:
    if sc >= 80:   new_grades["S"] += 1
    elif sc >= 65: new_grades["A"] += 1
    elif sc >= 50: new_grades["B"] += 1
    elif sc >= 35: new_grades["C"] += 1
    else:          new_grades["D"] += 1
for g, c in sorted(new_grades.items()):
    print(f"  {g}: {c:3d} ({c/len(pools)*100:.1f}%)")
print()

# Fetch detail of top-3 to see sub-scores
top3_ids = [p["pool_id"] for p in pools[:3]]
print("=== Top-3 pool sub-score detail ===")
for pid in top3_ids:
    d = httpx.get(f"http://127.0.0.1:8000/pool/{pid}", timeout=90).json()
    nm = d["pool_name"]
    pr = d["project"]
    print(f"  {nm} / {pr}  score={d['score']} grade={d['grade_label']}")
    for key in ["tvl_score","apy_score","risk_adjusted_score",
                "confidence_score","asset_safety","data_maturity",
                "risk_penalty","sustainability_ratio","sharpe_score"]:
        val = d.get(key)
        print(f"    {key:26s}: {val}")
    print()
