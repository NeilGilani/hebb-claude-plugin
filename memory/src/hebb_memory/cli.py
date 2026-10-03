"""hebb-memory train | check | demo"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from .memory import attach, selfcheck
from .registry import RECIPE, Registry
from .training import toy_model, train

#: Key-head and meta-training steps for the toy: about two and a half minutes on a 4-core CPU,
#: enough for questions to route to their own page. The real-model recipe is RECIPE.
TOY_STEPS = 300


def _attach(a):
    reg = Registry(a.registry) if a.registry else Registry()
    if a.model == "toy":
        lm, tok = toy_model()
        return attach(lm, tok, model_id="toy", registry=reg, checkpoint=a.checkpoint)
    return attach(a.model, registry=reg, device=a.device, checkpoint=a.checkpoint)


def demo(mem) -> None:
    facts = [("acme", "refund window", "60 days"), ("acme", "support channel", "phone"),
             ("globex", "refund window", "14 days"), ("globex", "support channel", "email")]
    options = {"refund window": ["14 days", "30 days", "60 days", "90 days"],
               "support channel": ["email", "phone", "chat", "ticket"]}
    print("before anything is written:")
    print("  acme refund window ->", mem.choose("acme", "refund window", options["refund window"]).best)
    recs = [mem.remember(s, a, v, rivals=options[a]) for s, a, v in facts]
    for r in recs:
        print(f"remembered  {r.subject:7} {r.attribute:16} = {r.value:8} page {r.page}")
    for s, a, v in facts:
        got = mem.choose(s, a, options[a]).best
        hits = ", ".join(f"page {h.record.page} ({h.record.subject} {h.record.attribute}) {h.weight:.2f}"
                         for h in mem.trace(s, a))
        print(f"ask         {s:7} {a:16} -> {got:8} {'right' if got == v else 'WRONG'}   read: {hits}")
    gone = mem.forget("acme")
    print(f"forgot acme: pages {[r.page for r in gone]} zeroed and freed")
    print("  acme refund window ->", mem.choose("acme", "refund window", options["refund window"]).best,
          "(no acme pages left; this is the model alone)")
    print("  globex refund window ->", mem.choose("globex", "refund window", options["refund window"]).best)
    mem.forget("globex")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="hebb-memory", description="memory written into a frozen model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the memory for a base model, once")
    t.add_argument("--model", required=True, help="Hugging Face model id, or 'toy'")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--registry", type=Path, default=None)
    t.add_argument("--meta-steps", type=int, default=None,
                   help=f"default {RECIPE['meta_steps']}, or {TOY_STEPS} for the toy")
    t.add_argument("--key-steps", type=int, default=None,
                   help=f"default {RECIPE['key_steps']}, or {TOY_STEPS} for the toy")
    t.add_argument("--seed", type=int, default=RECIPE["seed"])
    t.add_argument("--ckpt-dir", type=Path, default=None, help="resume here if interrupted")
    for name, text in (("check", "attach and run the self-check"),
                       ("demo", "write, ask, trace and forget a few memories"),
                       ("bench", "the paper's multi-customer accuracy test")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--model", required=True)
        c.add_argument("--device", default=None)
        c.add_argument("--registry", type=Path, default=None)
        c.add_argument("--checkpoint", default=None, help="a file, or hf:owner/repo")
        if name == "bench":
            c.add_argument("--customers", type=int, nargs="+", default=[8, 32])
            c.add_argument("--seeds", type=int, nargs="+", default=[0])
            c.add_argument("--facts", type=int, default=3, help="facts per customer")
            c.add_argument("--out", type=Path, default=None, help="write results as JSON here")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        reg = Registry(a.registry) if a.registry else Registry()
        toy = a.model == "toy"
        path = train(a.model, registry=reg, device=a.device, seed=a.seed, ckpt_dir=a.ckpt_dir,
                     meta_steps=a.meta_steps or (TOY_STEPS if toy else RECIPE["meta_steps"]),
                     key_steps=a.key_steps or (TOY_STEPS if toy else RECIPE["key_steps"]))
        print(f"trained memory saved to {path}")
        return 0
    mem = _attach(a)
    if a.cmd == "demo":
        demo(mem)
        return 0
    if a.cmd == "bench":
        from .bench import bench, table
        rows = bench(mem, a.customers, a.seeds, a.facts, out=a.out,
                     log=lambda m: print(m, flush=True))
        print(table(rows))
        return 0
    rep = selfcheck(mem)
    print(json.dumps({"attach": mem.info, "selfcheck": rep}, indent=1, default=str))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
