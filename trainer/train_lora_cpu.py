#!/usr/bin/env python3
"""EON Phase C — unattended CPU LoRA trainer (all-locals-off learning lane).

Standing owner mandate 2026-10-16 (EON ONE): keep EON improving on cloud EON
even with all locals off. This trainer runs on GitHub Actions public runners
(4-core CPU), samples a seeded slice of distill_data.jsonl (the AGI engine's
own distilled trace corpus), fine-tunes a Qwen2.5-0.5B-Instruct LoRA adapter,
and writes a durable receipt (train_report.json) + artifact (lora-adapter).

Receipt discipline: prints real numbers only; fails loudly (no junk records).
"""
import argparse, datetime, json, os, platform, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("EON_TRAIN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# v1.3 root-cause fix (run 38963798072: threads=2, ~209 s/step -> 75min cancel):
# explicit thread count + hard time-budget guard so a receipt ALWAYS lands.
import os as _os
_nt = int(_os.environ.get("OMP_NUM_THREADS", "0")) or (_os.cpu_count() or 2)
try:
    torch.set_num_threads(_nt)
except Exception:
    pass
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


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
    total = len(rows)
    # v1.1: corpus is heterogeneous (6 shapes). Normalize keys + census.
    PKEYS = ("prompt", "input")
    CKEYS = ("completion", "output")
    def usable(r):
        return bool(next((r[k] for k in PKEYS if r.get(k)), None)) and                bool(next((r[k] for k in CKEYS if r.get(k)), None))
    stray = [r for r in rows if not usable(r)]
    skipped_toolong = 0
    rows = [r for r in rows if usable(r)]
    print(f"[census] corpus={total} usable={len(rows)} unusable(no prompt/completion)={len(stray)}", flush=True)
    total = len(rows)
    seed = a.seed or int(datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%y%m%d%H"))
    rng = np.random.RandomState(seed)
    idx = rng.choice(total, size=min(a.traces, total), replace=False)
    rows = [rows[int(i)] for i in sorted(idx)]
    rows.sort(key=lambda r: len(str(r.get("prompt") or r.get("input") or "")) +
                            len(str(r.get("completion") or r.get("output") or "")))
    # v1.4: heldout split — base-vs-adapter signal in every receipt
    n_held = max(4, len(rows) // 10)
    held = rows[-n_held:]
    rows = rows[:-n_held]
    print(f"[split] train={len(rows)} heldout={len(held)}", flush=True)
    print(f"[data] corpus={total} sampled={len(rows)} seed={seed}", flush=True)

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

    base_held = eval_loss(held)
    print(f"[heldout] base loss={base_held}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
    n = len(rows)
    order = np.arange(n)
    step = 0
    losses = []
    cut = False
    log_every = max(1, (n // a.batch) // 8)
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
            sel = [rows[int(i)] for i in order[s:s + a.batch]]
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
    print(f"[heldout] after-adapter loss={after_held} (delta={round(after_held - base_held, 4)})", flush=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)

    mean_loss = sum(losses) / len(losses) if losses else None
    report = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "runner": "github-actions-cpu-4core",
        "mandate": "standing owner order 2026-10-16 (EON ONE): unattended cloud trainer",
        "model": MODEL,
        "adapter": {"out": a.out, "r": 16, "alpha": 32,
                    "target_modules": 7, "trainable_params": trainable},
        "data": {"file": a.data, "corpus": total, "usable_rows": len(rows), "unusable_rows": len(stray), "sampled": len(rows),
                 "seed": seed, "ts": started},
        "train": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                  "maxlen": a.maxlen, "steps": step},
        "loss": {"mean_last": round(mean_loss, 4) if mean_loss else None,
                 "first5": [round(x, 4) for x in losses[:5]],
                 "last5": [round(x, 4) for x in losses[-5:]],
                 "heldout_base": base_held, "heldout_after": after_held,
                 "heldout_delta": round(after_held - base_held, 4),
                 "heldout_n": len(held)},
        "runtime_sec": round(time.time() - t0, 1),
        "cut_by_budget": cut,
        "threads": torch.get_num_threads(),
        "machine": platform.machine(),
    }
    json.dump(report, open(a.report, "w"), indent=1)
    print("[report]", json.dumps(report)[:600], flush=True)


if __name__ == "__main__":
    main()
