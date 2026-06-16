import json

with open("eval_results_baseline.json") as f:
    baseline = json.load(f)
with open("eval_results_reranked.json") as f:
    reranked = json.load(f)

# pair them up by position (same question order in both runs)
for b, r in zip(baseline, reranked):
    if r["corr_score"] != b["corr_score"]:
        print(f"\nQ: {b['question']}")
        print(f"  corr: {b['corr_score']} -> {r['corr_score']}  |  hit: {b['hit']} -> {r['hit']}  |  rr: {b['rr']:.2f} -> {r['rr']:.2f}")
        print(f"  BASELINE reason: {b['corr_reason']}")
        print(f"  RERANKED reason: {r['corr_reason']}")
    
