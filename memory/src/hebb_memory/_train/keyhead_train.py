"""Train the key head to route among HUNDREDS of tenant rules before meta-training starts.

WHY THIS EXISTS. On the real model the crossover sweep (results/crossover-qwen05b) put
`pages_keyed` at chance once a tenant held 16 rules. `hebb.tenants.routing_sweep` then measured
the address itself and found the reason: with 32 rules per tenant the query's key lands on the
right page 6% of the time, chance being 3%. Four knobs on the read side (a negative bank, read_k,
key_dim) moved that by a point or two. The head was not mis-tuned; it was never trained for
the job:

  * `meta_loss`'s contrastive term picks one page out of an episode of FOUR, so the head only
    ever learns four-way discrimination and faces 256-way at eval;
  * `training_rule` draws from the eight hand-written slots only, so a composed slot name such
    as "premium refund window" -- everything past eight rules per tenant -- is unseen until the
    eval asks the head to separate thirty of them.

This stage is a plain retriever objective: many tenants x many slots, big batches deliberately
packed with the two confusions that matter (same tenant / different slot, same slot / different
tenant), positives being every phrasing of one (tenant, slot) -- the teaching with either of two
values, the held-out question, the support forms. It is the same idea as `hebb.lm.keytrain`, on
this benchmark's own text.

HELD OUT. Tenants come from TRAIN_TENANTS, never the evaluated names. Slot names by default
come from a vocabulary DISJOINT from the eval's (`--key-slots disjoint`): different fields, the
same qualifiers, so the head must generalise the "qualifier + field" structure to fields it has
never seen. `--key-slots shared` trains on the eval slot vocabulary (values and tenants still
held out) -- the deployment case where the customer's schema is known -- and is reported
separately because it is an easier claim.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .._core.keytrain import _cache_states
from .slots import _QUALIFIERS, _VALUE_SETS, build_slots
from .rules import SLOTS, TRAIN_TENANTS, Rule

#: Fields for the held-out training vocabulary. None appears in `crossover._FIELDS` or SLOTS.
_TRAIN_FIELDS = [
    "shipping cutoff", "deposit amount", "cancellation fee", "response target", "storage quota",
    "signature rule", "discount ceiling", "invoice format", "holiday cover", "login policy",
    "refresh interval", "quote validity", "sample policy", "return address", "review cadence",
    "rate limit", "session length", "archive rule", "handoff step", "callback window",
    "credit hold", "pickup slot", "delivery zone", "lead source", "tax treatment",
    "safety stock", "reorder point", "contract term", "budget owner", "reporting line",
    "parking allowance", "uniform policy", "visitor rule", "key holder", "alarm code owner",
    "fuel card limit", "mileage rate", "overtime rule", "break schedule", "dress code",
    "tip policy", "comp voucher", "loyalty tier", "referral bonus", "price match rule",
    "gift wrap option", "layaway term", "restock fee", "damage claim window", "carrier choice",
    "signature threshold", "cold chain rule", "hazmat handling", "customs broker", "duty payer",
]
#: Qualifiers past the eval's own seven, so the head sees the qualifier slot filled many ways.
_EXTRA_QUALIFIERS = ["basic", "internal", "seasonal", "partner", "wholesale", "retail", "urgent",
                     "default", "override", "weekend"]


def train_slot_vocab(n: int) -> List[Tuple[str, List[str]]]:
    """`n` slots the eval never uses: held-out fields x the eval's qualifiers."""
    out: List[Tuple[str, List[str]]] = []
    quals = list(_QUALIFIERS) + _EXTRA_QUALIFIERS
    i = 0
    while len(out) < n:
        field = _TRAIN_FIELDS[i % len(_TRAIN_FIELDS)]
        qual = quals[(i // len(_TRAIN_FIELDS)) % len(quals)]
        name = field if i < len(_TRAIN_FIELDS) else f"{qual} {field}"
        if name not in {s for s, _ in out}:
            out.append((name, list(_VALUE_SETS[i % len(_VALUE_SETS)])))
        i += 1
        if i > n * 20:
            raise ValueError(f"cannot build {n} distinct training slots")
    return out[:n]


def slot_vocab_is_disjoint() -> bool:
    """Every held-out training slot name differs from every eval slot name."""
    train = {s for s, _ in train_slot_vocab(len(_TRAIN_FIELDS) * (len(_QUALIFIERS) + len(_EXTRA_QUALIFIERS)))}
    ev = {s for s, _ in build_slots(len(SLOTS) + 30 * len(_QUALIFIERS))}
    return not (train & ev)


def key_corpus(seed: int, n_tenants: int, n_slots: int, slots: str = "disjoint",
               paraphrases: bool = False):
    """Every text the head trains on, labelled by the (tenant, slot) it addresses.

    Per owner: the teaching with two different values (so the address ignores the value), the
    question form, and both support forms. Returns texts, owner ids, tenant ids, slot ids.
    """
    rng = np.random.default_rng(seed + 4242)
    tenants = list(TRAIN_TENANTS)
    rng.shuffle(tenants)
    tenants = tenants[:n_tenants]
    vocab = train_slot_vocab(n_slots) if slots == "disjoint" else build_slots(n_slots)
    texts: List[str] = []; owners: List[int] = []; tid: List[int] = []; sid: List[int] = []
    o = 0
    for ti, t in enumerate(tenants):
        for si, (slot, values) in enumerate(vocab):
            v1, v2 = rng.choice(values, size=2, replace=False)
            r1, r2 = Rule(t, slot, str(v1)), Rule(t, slot, str(v2))
            # Exactly the strings the eval will hand to the key head: the question keeps
            # its newline, the support prompts their form. Plus two paraphrases of the
            # question, so the address is tied to the (tenant, slot) and not to one template.
            forms = [r1.teach_text(), r2.teach_text(), r1.query()] + [p for p, _ in r1.support()]
            if paraphrases:
                forms += [f"What is {t}'s {slot}?", f"{t}: {slot}?"]
            for f in forms:
                texts.append(f); owners.append(o); tid.append(ti); sid.append(si)
            o += 1
    return texts, np.array(owners), np.array(tid), np.array(sid)


def _batch_owners(rng, n_tenants: int, n_slots: int, per_batch_tenants: int,
                  per_batch_slots: int) -> np.ndarray:
    """Owner ids for one batch: a block of tenants x a block of slots.

    A block, not a uniform draw, is what makes the batch hard. Every owner in it shares its
    tenant with `per_batch_slots - 1` others and its slot with `per_batch_tenants - 1` others,
    which is exactly the pair of confusions the address has to resolve.
    """
    ts = rng.choice(n_tenants, size=min(per_batch_tenants, n_tenants), replace=False)
    ss = rng.choice(n_slots, size=min(per_batch_slots, n_slots), replace=False)
    return (ts[:, None] * n_slots + ss[None, :]).reshape(-1)


def train_tenant_key_head(model, seed: int, steps: int = 400, lr: float = 1e-3,
                          temp: float = 0.07, n_tenants: int = 40, n_slots: int = 48,
                          batch_tenants: int = 12, batch_slots: int = 12,
                          slots: str = "disjoint", log_every: int = 100,
                          paraphrases: bool = False) -> Dict:
    """Fit `model.cue_proj` so a question's key lands on its own (tenant, slot) page.

    Supervised contrastive: two random phrasings per owner in the batch, every phrasing of the
    same owner a positive, everything else in the batch a negative. With the default blocks a
    batch holds 144 owners -- 288 keys -- against which each query must pick its own.
    """
    if slots not in ("disjoint", "shared"):
        raise ValueError("slots must be 'disjoint' or 'shared'")
    texts, owners, tid, sid = key_corpus(seed, n_tenants, n_slots, slots, paraphrases)
    n_owner = int(owners.max()) + 1
    by_owner: List[np.ndarray] = [np.nonzero(owners == o)[0] for o in range(n_owner)]
    t0 = time.time()
    H, M = _cache_states(model, texts)
    cache_s = time.time() - t0

    opt = torch.optim.AdamW(model.cue_proj.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 99)
    losses: List[float] = []
    hits: List[float] = []
    t1 = time.time()
    for step in range(steps):
        ow = _batch_owners(rng, n_tenants, n_slots, batch_tenants, batch_slots)
        sel = np.concatenate([rng.choice(by_owner[o], size=2, replace=False) for o in ow])
        idx = torch.as_tensor(sel, device=H.device)
        z = F.normalize(model.cue_proj(H[idx], M[idx]), dim=-1)
        own = torch.as_tensor(owners[sel], device=H.device)
        sim = z @ z.T / temp
        eye = torch.eye(len(sel), device=H.device)
        pos = (own[:, None] == own[None, :]).float() * (1 - eye)
        logp = (sim - 1e9 * eye).log_softmax(-1)
        loss = -((logp * pos).sum(-1) / pos.sum(-1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            top = (sim - 1e9 * eye).argmax(-1)
            hits.append(float((own[top] == own).float().mean()))
        if (step + 1) % log_every == 0 or step == 0:
            print(f"    key-train {step + 1}/{steps}  loss {np.mean(losses[-log_every:]):.3f}  "
                  f"in-batch@1 {np.mean(hits[-log_every:]):.3f}  "
                  f"({len(sel)} keys/batch)", flush=True)
    return {"steps": steps, "items": len(texts), "owners": n_owner, "slots": slots,
            "paraphrases": paraphrases,
            "n_tenants": n_tenants, "n_slots": n_slots, "keys_per_batch": 2 * batch_tenants * batch_slots,
            "final_loss": float(np.mean(losses[-25:])) if losses else float("nan"),
            "final_in_batch_top1": float(np.mean(hits[-25:])) if hits else float("nan"),
            "cache_seconds": round(cache_s, 1), "train_seconds": round(time.time() - t1, 1)}
