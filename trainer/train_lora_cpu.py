#!/usr/bin/env python3
"""EON Phase C — unattended CPU LoRA trainer (all-locals-off learning lane).

Standing owner mandate 2026-10-16 (EON ONE): keep EON improving on cloud EON
even with all locals off. This trainer runs on GitHub Actions public runners
(4-core CPU), samples a seeded, deduplicated, diversity-balanced slice of the
AGI engine's own distilled trace corpus, fine-tunes a Qwen2.5-0.5B-Instruct
LoRA adapter, and writes a durable receipt + artifact (lora-adapter).

v1.5 (2026-10-17):
- gen_eval: generation-based exact-match heldout eval — greedy decode from
  heldout prompts before training (LoRA init == base) and after training;
  normalized exact-match + char-ratio; budget-guarded; skips honestly
  (runner-too-slow / budget) instead of lying or hanging.
- slice: dedupe exact duplicates + template-signature buckets + deterministic
  round-robin (was uniform random); receipt states pool/unique/dupes/buckets.

v1.4: heldout loss A/B (base vs adapter) in every receipt; len-sorted batching.
v1.3: threads pin + time-budget guard (receipts ALWAYS land) + strings robot.

Receipt discipline: prints real numbers only; fails loudly (no junk records).
"""
import argparse, datetime, difflib, hashlib, json, os, platform, time
from collections import OrderedDict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("EON_TRAIN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# v1.3 root-cause fix (run 38963798072: threads=2, ~209 s/step -> 75min cancel):
# explicit thread count + hard time-budget guard so a receipt ALWAYS lands.
_nt = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 2)
try:
    torch.set_num_threads(_nt)
except Exception:
    pass
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# v1.5 gen-eval knobs (all budget-guarded; auto-skip class = 'runner-too-slow')
GEN_K = int(os.environ.get("EON_GEN_EVAL_K", "8"))
GEN_MAX_NEW = int(os.environ.get("EON_GEN_MAX_NEW", "24"))
GEN_PROBE_LIMIT = float(os.environ.get("EON_GEN_PROBE_LIMIT", "20"))
GEN_PASS_MAX = float(os.environ.get("EON_GEN_PASS_MAX", "420"))


def norm(s):
    return " ".join(str(s).lower().split()).rstrip(" .")


def build_slice(pool, want, seed):
    """v1.5 diversity: dedupe, bucket by template signature, round-robin."""
    seen = set()
    uniq = []
    for r in pool:
        p = str(r.get("prompt") or r.get("input") or "")
        c = str(r.get("completion") or r.get("output") or "")
        key = hashlib.sha256((p + "\x00" + c).encode("utf-8", "ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    def bucket(r):
        p = " ".join(str(r.get("prompt") or r.get("input") or "").lower().split())[:24]
        L = len(str(r.get("completion") or r.get("output") or ""))
        lb = 0 if L < 24 else 1 if L < 64 else 2 if L < 160 else 3
        return (p, lb)

    buckets = OrderedDict()
    for r in uniq:
        buckets.setdefault(bucket(r), []).append(r)
    rng = np.random.RandomState(seed)
    keys = list(buckets.keys())
    # index-permutation style (version-safe across numpy variants)
    keys = [keys[int(i)] for i in rng.permutation(len(keys))]
    lists = [buckets[k] for k in keys]
    lists = [[l[int(i)] for i in rng.permutation(len(l))] for l in lists]
    picked = []
    i = 0
    while len(picked) < want:
        progressed = False
        for l in lists:
            if i < len(l):
                picked.append(l[i])
                progressed = True
                if len(picked) >= want:
                    break
        if not progressed:
            break
        i += 1
    meta = {"method": "dedupe+bucket-roundrobin-v1.5", "pool": len(pool),
            "unique": len(uniq), "dupes_dropped": len(pool) - len(uniq),
            "buckets": len(lists), "selected": len(picked), "seed": seed}
    return picked, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="distill_data.jsonl")
    ap.add_argument("--out", default="adapter_out")
    ap.add_argument("--report", default="train_report.json")
    ap.add_argument("--traces", type=int,
                    default=int(os.environ.get("EON_TRAIN_TRACES", "400")))
    ap.add_argument("--epochs", type=int,
                    default=int(os.environ.get("EON_TRAIN_EPOCHS", "1")))
    ap.add_argument("--seed", type=int,
                    default=int(os.environ.get("EON_TRAIN_SEED", "0")))
    ap.add_argument("--lr", type=float,
                    default=float(os.environ.get("EON_TRAIN_LR", "2e-4")))
    ap.add_argument("--batch", type=int,
                    default=int(os.environ.get("EON_TRAIN_BATCH", "2")))
    ap.add_argument("--maxlen", type=int,
                    default=int(os.environ.get("EON_TRAIN_MAXLEN", "160")))
    ap.add_argument("--max-seconds", type=int,
                    default=int(os.environ.get("EON_TRAIN_MAX_SECONDS", "2100")))
    a = ap.parse_args()

    t0 = time.time()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    corpus_all = len(rows)
    # v1.1: corpus is heterogeneous (6 shapes). Normalize keys + census.
    PKEYS = ("prompt", "input")
    CKEYS = ("completion", "output")
    def usable(r):
        return bool(next((r[k] for k in PKEYS if r.get(k)), None)) and \
               bool(next((r[k] for k in CKEYS if r.get(k)), None))
    stray = [r for r in rows if not usable(r)]
    pool = [r for r in rows if usable(r)]
    print(f"[census] corpus={corpus_all} usable={len(pool)} "
          f"unusable(no prompt/completion)={len(stray)}", flush=True)

    seed = a.seed or int(datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%y%m%d%H"))
    want = min(a.traces, len(pool))
    picked, slice_meta = build_slice(pool, want, seed)
    print(f"[slice] {json.dumps(slice_meta)}", flush=True)

    rng = np.random.RandomState(seed ^ 0x5EED)
    perm = rng.permutation(len(picked))
    picked = [picked[int(i)] for i in perm]
    if len(picked) <= 6:
        rows_tr, held = list(picked), []
    else:
        n_held = max(4, len(picked) // 10)
        held = picked[-n_held:]
        rows_tr = picked[:-n_held]
    rows_tr.sort(key=lambda r: len(str(r.get("prompt") or r.get("input") or "")) +
                 len(str(r.get("completion") or r.get("output") or "")))
    # v1.4: heldout split — base-vs-adapter signal in every receipt
    print(f"[split] train={len(rows_tr)} heldout={len(held)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=TARGET_MODULES)
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {MODEL} | LoRA trainable={trainable:,} | "
          f"py={platform.python_version()} torch={torch.__version__} "
          f"threads={torch.get_num_threads()}", flush=True)

    def enc(r):
        ptxt = next((r[k] for k in PKEYS if r.get(k)), None)
        ctxt = next((r[k] for k in CKEYS if r.get(k)), None)
        pid = tok(ptxt, add_special_tokens=False)["input_ids"]
        cid = tok(ctxt, truncation=True, max_length=a.maxlen,
                  add_special_tokens=False)["input_ids"]
        ids = (pid + cid)[:a.maxlen]
        labels = ([-100] * min(len(pid), a.maxlen) +
                  ids[min(len(pid), a.maxlen):])
        # clamp: labels list must match ids
        labels = labels[:len(ids)]
        if len(labels) < len(ids):
            labels += [-100] * (len(ids) - len(labels))
        return ids, labels

    def eval_loss(rows_eval):
        if not rows_eval:
            return None
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for s in range(0, len(rows_eval), a.batch):
                ii, ll, aa = batch(rows_eval[s:s + a.batch])
                tot += float(model(input_ids=ii, attention_mask=aa, labels=ll).loss)
        model.train()
        return round(tot / max(1, (len(rows_eval) + a.batch - 1) // a.batch), 4)

    def batch(bs_rows):
        items = [enc(r) for r in bs_rows]
        L = max(len(i[0]) for i in items)
        input_ids = torch.full((len(items), L), tok.pad_token_id,
                               dtype=torch.long)
        labels = torch.full((len(items), L), -100, dtype=torch.long)
        att = torch.zeros((len(items), L), dtype=torch.long)
        for n, (ids, lab) in enumerate(items):
            input_ids[n, :len(ids)] = torch.tensor(ids)
            labels[n, :len(lab)] = torch.tensor(lab)
            att[n, :len(ids)] = 1
        return input_ids, labels, att

    def gen_eval_pass(rows_gen, deadline, tag):
        """v1.5: greedy decode each heldout prompt; normalized exact-match."""
        model.eval()
        res = []
        try:
            for i, r in enumerate(rows_gen):
                if time.time() > deadline:
                    print(f"[gen-{tag}] deadline hit "
                          f"({len(res)}/{len(rows_gen)} done)", flush=True)
                    break
                ptxt = str(r.get("prompt") or r.get("input") or "")
                exp = str(r.get("completion") or r.get("output") or "")
                ids = tok(ptxt, return_tensors="pt", truncation=True,
                          max_length=a.maxlen)["input_ids"]
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=GEN_MAX_NEW,
                                         do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                res.append({"i": i, "em": norm(pred) == norm(exp),
                            "ratio": round(difflib.SequenceMatcher(
                                None, norm(pred), norm(exp)).ratio(), 3),
                            "pred": pred[:100], "exp": exp[:100]})
        finally:
            model.train()
        return res

    # v1.5 probe: classify runner pace before spending generation budget
    tp0 = time.time()
    if rows_tr:
        _ = eval_loss(rows_tr[:1])
    probe_sec = round(time.time() - tp0, 2)
    gen_ok = (probe_sec <= GEN_PROBE_LIMIT) and len(held) > 0
    print(f"[probe] single-batch fwd {probe_sec}s -> gen-eval "
          f"{'ENABLED' if gen_ok else 'SKIPPED (runner-too-slow or no heldout)'}",
          flush=True)

    gen_samples = held[:min(GEN_K, len(held))]
    gen_skipped = None
    before_gen = []
    if not gen_ok:
        gen_skipped = "runner-too-slow"
    elif (time.time() - t0) < a.max_seconds * 0.25:
        before_gen = gen_eval_pass(model, gen_samples,
                                   time.time() + GEN_PASS_MAX, "before")
        print(f"[gen-before] {len(before_gen)}/{len(gen_samples)} decoded", flush=True)

    base_held = eval_loss(held)
    print(f"[heldout] base loss={base_held}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
    n = len(rows_tr)
    order = np.arange(n)
    step = 0
    losses = []
    cut = False
    log_every = max(1, (n // a.batch) // 8) if n else 1
    for ep in range(a.epochs):
        if cut:
            break
        rng.shuffle(order)
        for s in range(0, n - a.batch + 1, a.batch):
            if time.time() - t0 > a.max_seconds:
                cut = True
                print(f"[cut] time budget {a.max_seconds}s reached at step {step} "
                      f"(pace {(time.time()-t0)/max(step,1):.1f} s/step) — saving receipts", flush=True)
                break
            sel = [rows_tr[int(i)] for i in order[s:s + a.batch]]
            input_ids, labels, att = batch(sel)
            out = model(input_ids=input_ids, attention_mask=att,
                        labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            v = float(loss.detach())
            losses.append(v)
            if step == 3:
                print(f"[pace] {((time.time()-t0)/3):.1f} s/step (threads={torch.get_num_threads()})", flush=True)
            if step % log_every == 0:
                m = sum(losses[-log_every:]) / min(log_every, len(losses[-log_every:]))
                print(f"[step {step}] loss={v:.4f} avg={m:.4f}", flush=True)

    after_held = eval_loss(held)
    if held:
        print(f"[heldout] after-adapter loss={after_held} (delta={round(after_held - base_held, 4)})", flush=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)

    after_gen = []
    if gen_ok:
        after_gen = gen_eval_pass(model, gen_samples,
                                  time.time() + GEN_PASS_MAX, "after")
        print(f"[gen-after] {len(after_gen)}/{len(gen_samples)} decoded", flush=True)

    def agg(res):
        done = [r for r in res if "em" in r]
        return {"done": len(done),
                "em": sum(1 for r in done if r["em"]),
                "ratio_mean": round(sum(r["ratio"] for r in done) / len(done), 3)
                if done else None}

    aligned = [(b, x) for b, x in zip(before_gen, after_gen)
               if "em" in b and "em" in x]
    gen_eval = {"k": len(gen_samples), "max_new_tokens": GEN_MAX_NEW,
                "probe_sec": probe_sec, "pass_budget_sec": GEN_PASS_MAX,
                "before": agg(before_gen), "after": agg(after_gen),
                "aligned": {"n": len(aligned),
                            "em_before": sum(1 for b, x in aligned if b["em"]),
                            "em_after": sum(1 for b, x in aligned if x["em"])},
                "skipped": gen_skipped,
                "examples": [{"p": str(gen_samples[b["i"]].get("prompt"))[:80],
                              "exp": b["exp"], "pred_before": b["pred"],
                              "pred_after": x["pred"]}
                             for b, x in aligned[:2]]}
    if gen_skipped is None and not before_gen:
        gen_eval["note"] = "before-pass skipped (pre-train budget); after-only stats"

    mean_loss = sum(losses) / len(losses) if losses else None
    report = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runner": "github-actions-cpu-4core",
        "mandate": "standing owner order 2026-10-16 (EON ONE): unattended cloud trainer",
        "trainer_version": "1.5",
        "model": MODEL,
        "adapter": {"out": a.out, "r": 16, "alpha": 32,
                    "target_modules": 7, "trainable_params": trainable},
        "data": {"file": a.data, "corpus": corpus_all, "usable_rows": len(pool),
                 "unusable_rows": len(stray), "sampled": len(picked),
                 "train": len(rows_tr), "heldout": len(held),
                 "seed": seed, "ts": started},
        "slice": slice_meta,
        "train": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                  "maxlen": a.maxlen, "steps": step},
        "loss": {"mean_last": round(mean_loss, 4) if mean_loss else None,
                 "first5": [round(x, 4) for x in losses[:5]],
                 "last5": [round(x, 4) for x in losses[-5:]],
                 "heldout_base": base_held, "heldout_after": after_held,
                 "heldout_delta": (round(after_held - base_held, 4)
                                   if (base_held is not None and after_held is not None) else None),
                 "heldout_n": len(held)},
        "gen_eval": gen_eval,
        "runtime_sec": round(time.time() - t0, 1),
        "cut_by_budget": cut,
        "threads": torch.get_num_threads(),
        "machine": platform.machine(),
    }
    json.dump(report, open(a.report, "w"), indent=1)
    print("[report]", json.dumps(report)[:900], flush=True)


if __name__ == "__main__":
    main()
