#!/usr/bin/env python3
"""EON Phase C — strings robot (neural-matrix corpus QC robot).

Mandate 2026-10-16 (EON ONE). Purpose: the trainer crashed once on corpus
schema drift (KeyError 'prompt', run 38958079028). This robot runs BEFORE
training and guarantees a normalized, contract-checked corpus:
- census of record shapes (strings) with counts
- normalization: prompt<-prompt|input, completion<-completion|output
- length percentiles; unusable rows listed by shape
- writes distill_norm.jsonl + strings_qc.json (durable receipt)
Exit 0 always when receipt written; exit 1 only if corpus unreadable.
"""
import datetime, json, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "distill_data.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "distill_norm.jsonl"
REP = "strings_qc.json"

def main():
    shapes = {}
    unusable = {}
    norm = []
    lens = []
    bad = 0
    for l in open(SRC, encoding="utf-8"):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            bad += 1
            continue
        k = ",".join(sorted(r.keys()))
        shapes[k] = shapes.get(k, 0) + 1
        p = next((r[x] for x in ("prompt", "input") if r.get(x)), None)
        c = next((r[x] for x in ("completion", "output") if r.get(x)), None)
        if not p or not c:
            unusable[k] = unusable.get(k, 0) + 1
            continue
        p, c = str(p), str(c)
        lens.append(len(p) + len(c))
        norm.append({"prompt": p, "completion": c})
    lens.sort()
    def pct(q):
        return lens[min(len(lens) - 1, int(q * len(lens)))] if lens else 0
    rep = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "robot": "strings-robot/neural-matrix (Phase C corpus QC)",
        "src": SRC, "rows_in": sum(shapes.values()) + bad,
        "rows_normalized": len(norm), "bad_json": bad,
        "unusable_by_shape": unusable,
        "shapes": dict(sorted(shapes.items(), key=lambda kv: -kv[1])[:10]),
        "len_chars": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                      "max": lens[-1] if lens else 0},
        "contract": "every emitted row: {prompt:str>=1, completion:str>=1}",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        for r in norm:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(rep, open(REP, "w"), indent=1)
    print("[strings-robot]", json.dumps(rep)[:500], flush=True)

if __name__ == "__main__":
    main()
