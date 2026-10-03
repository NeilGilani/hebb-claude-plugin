"""The paper's multi-customer test, run through the public API.

Each customer holds a few facts that conflict with other customers' facts ("for acme the refund
window is 60 days", "for globex it is 14 days"). Every fact is learned with `remember`, in the
interleaved order real traffic would arrive in, and then every customer is asked about every
one of its facts with `choose`, among every value any customer holds for that attribute:

    correct  the model picked this customer's value
    leaked   it picked another customer's value for the same attribute
    other    it picked a value no customer holds for it

The same questions are first asked of the model with nothing learned, so the table shows what
the memory adds over the model alone. Chance is the average of 1 / (number of options).

    hebb-memory bench --model Qwen/Qwen2.5-0.5B --customers 8 32 --seeds 0 1
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ._train.rules import make_world


def run(mem, customers: int, facts: int = 3, seed: int = 0, log=print) -> Dict:
    """One world: `customers` customers with `facts` facts each. Leaves the memory empty."""
    if len(mem):
        raise RuntimeError("bench needs an empty memory")
    world = make_world(customers, facts, seed)
    queries = [(r, world.candidates(r.slot)) for r in world.rules]
    chance = sum(1 / len(c) for _, c in queries) / len(queries)
    bare = sum(mem.choose(r.tenant, r.slot, c).best == r.value for r, c in queries) / len(queries)
    t0 = time.time()
    for i, r in enumerate(world.stream()):
        mem.remember(r.tenant, r.slot, r.value, rivals=r.rivals)
        if (i + 1) % 8 == 0:
            log(f"    learned {i + 1}/{len(world.rules)} facts ({time.time() - t0:.0f}s)")
    learn_s = time.time() - t0
    counts = {"correct": 0, "leaked": 0, "other": 0}
    per: List[Dict] = []
    for r, cands in queries:
        got = mem.choose(r.tenant, r.slot, cands).best
        owners = world.owner_of(r.slot, got)
        verdict = "correct" if got == r.value else ("leaked" if owners and r.tenant not in owners else "other")
        counts[verdict] += 1
        per.append({"customer": r.tenant, "attribute": r.slot, "want": r.value, "got": got,
                    "verdict": verdict})
    for rec in mem.records():
        mem.forget(rec)
    n = len(queries)
    return {"customers": customers, "facts_each": facts, "seed": seed, "questions": n,
            "chance": round(chance, 3), "model_alone": round(bare, 3),
            "correct": round(counts["correct"] / n, 3), "leaked": round(counts["leaked"] / n, 3),
            "other": round(counts["other"] / n, 3),
            "seconds_per_fact": round(learn_s / n, 1), "per_question": per}


def table(rows: List[Dict]) -> str:
    out = [f"{'customers':>9} {'seed':>4} {'chance':>7} {'model alone':>11} {'with Hebb':>9} {'leaked':>7} {'s/fact':>7}"]
    for r in rows:
        out.append(f"{r['customers']:>9} {r['seed']:>4} {r['chance']:>7.3f} {r['model_alone']:>11.3f} "
                   f"{r['correct']:>9.3f} {r['leaked']:>7.3f} {r['seconds_per_fact']:>7.1f}")
    by: Dict[int, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["customers"], []).append(r)
    for n, rs in sorted(by.items()):
        if len(rs) > 1:
            mean = lambda k: sum(x[k] for x in rs) / len(rs)
            out.append(f"{n:>9} {'mean':>4} {mean('chance'):>7.3f} {mean('model_alone'):>11.3f} "
                       f"{mean('correct'):>9.3f} {mean('leaked'):>7.3f} {mean('seconds_per_fact'):>7.1f}")
    return "\n".join(out)


def bench(mem, customers: List[int], seeds: List[int], facts: int = 3,
          out: Optional[Path] = None, log=print) -> List[Dict]:
    rows: List[Dict] = []
    for seed in seeds:
        for n in customers:
            log(f"  {n} customers, seed {seed}")
            rows.append(run(mem, n, facts, seed, log=log))
            r = rows[-1]
            log(f"  -> with Hebb {r['correct']:.3f}, model alone {r['model_alone']:.3f}, "
                f"chance {r['chance']:.3f}, leaked {r['leaked']:.3f}")
            if out is not None:
                Path(out).write_text(json.dumps({"model": mem.info.get("model"), "rows": rows}, indent=1))
    return rows
