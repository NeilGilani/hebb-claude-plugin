"""The key head: the projection that decides which page a teaching addresses.

`build_stream` produces the addressing workload -- a stream where facts repeat verbatim,
get restated in other words, and get revised to new values -- with each event labelled by
the memory SLOT it belongs to, (kind, entity, attribute), rather than by its position.

`train_key_head` fits the projection to that workload. Meta-training's own contrastive term
only ever pairs a quiz question with its teaching inside a 4-teaching episode, so the key
head never sees the case that actually breaks it: two different facts wearing the same
template. Measured on the toy LM, training here takes retrieval@1 from 0.09 to 0.65 --
far more than any change to the routing rule buys.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import ATTRIBUTES, Teaching, TeachQuizGenerator, restate, revise


def build_stream(n: int, seed: int) -> Tuple[List[Teaching], List[Tuple[Teaching, int]], List[int]]:
    """Returns the base teachings and the ordered event stream of (teaching, owner_index).

    owner_index identifies the memory SLOT the event belongs to: the first teaching that
    occupied (kind, entity, attribute). Restatements, revisions and verbatim repeats all
    share their original's slot. `owner_of[i]` gives the slot of base teaching i.
    """
    gen = TeachQuizGenerator(seed)
    base = gen.stream(n, "eval")
    # Ownership is the SLOT a teaching occupies -- (kind, entity, attribute) -- not its
    # position in the stream. The generator draws with replacement, so a 60-teaching stream
    # repeats facts and rules verbatim; putting two identical teachings on one page is
    # correct behaviour, and an earlier version of this benchmark scored it as a collision.
    slot: Dict[Tuple[str, str, str], int] = {}
    owner_of: List[int] = []
    for i, t in enumerate(base):
        owner_of.append(slot.setdefault((t.kind, t.entity, t.attribute), i))
    events: List[Tuple[Teaching, int]] = []
    rng = np.random.default_rng(seed)
    for i, t in enumerate(base):
        events.append((t, owner_of[i]))
        if i % 3 == 2:
            r = Teaching(t.index, t.kind, t.entity, t.attribute, t.value, restate(t), t.support, [],
                         meta={"restates": t.index})
            events.append((r, owner_of[i]))
        if i % 5 == 4 and t.kind == "fact":
            pool = [v for v in ATTRIBUTES[t.attribute] if v != t.value]
            events.append((revise(t, pool[int(rng.integers(len(pool)))]), owner_of[i]))
    return base, events, owner_of


def _items(base, events, owner_of):
    """Every text the key head trains on, with the slot it belongs to and the slot's parts.

    Slot parts matter because they define the two hard cases: same subject, different attribute
    (two memories that look almost identical) and same attribute, different subject.
    """
    texts, owners, ent, attr = [], [], [], []
    def add(t, o, e, a):
        texts.append(t); owners.append(o); ent.append(e); attr.append(a)
    for t, o in events:
        add(t.text, o, t.entity, t.attribute)
    for i, t in enumerate(base):
        for q, _ in t.quizzes:
            add(q, owner_of[i], t.entity, t.attribute)
        for pr, _ in t.support:
            add(pr.replace("\n", " ").strip(), owner_of[i], t.entity, t.attribute)
    return texts, np.array(owners), ent, attr


@torch.no_grad()
def _cache_states(model, texts: List[str], chunk: int = 64):
    """Frozen token states for every training text, padded to one common length.

    The LM is frozen, so these never change and are computed once. Only the key head trains.
    """
    hs, ms = [], []
    for i in range(0, len(texts), chunk):
        h, m = model.token_states(texts[i:i + chunk])
        hs.append(h); ms.append(m)
    T = max(h.shape[1] for h in hs)
    d = hs[0].shape[-1]
    H = torch.zeros(len(texts), T, d, device=hs[0].device)
    M = torch.zeros(len(texts), T, dtype=ms[0].dtype, device=ms[0].device)
    i = 0
    for h, m in zip(hs, ms):
        n, t = h.shape[0], h.shape[1]
        H[i:i + n, :t] = h; M[i:i + n, :t] = m; i += n
    return H, M


def _hard_batch(rng, owners, ent, attr, by_slot, by_ent, by_attr, n_slots: int, per_slot: int):
    """A batch built out of the confusions that actually break addressing.

    A uniformly random batch of 64 from a few hundred items rarely contains two facts about the
    same module, so the head is almost never asked to tell them apart, and it does not learn to.
    Each batch here is anchored on a few slots and then packed with their near misses: same
    subject / different attribute, and same attribute / different subject.
    """
    slots = list(by_slot)
    rng.shuffle(slots)
    idx: List[int] = []
    for s in slots[:n_slots]:
        pool = by_slot[s]
        idx += list(rng.choice(pool, size=min(per_slot, len(pool)), replace=False))
        e, a = ent[pool[0]], attr[pool[0]]
        for near in (by_ent.get(e, []), by_attr.get(a, [])):
            cands = [j for j in near if owners[j] != owners[pool[0]]]
            if cands:
                idx += list(rng.choice(cands, size=min(2, len(cands)), replace=False))
    return list(dict.fromkeys(int(i) for i in idx))


def train_key_head(model, seed: int, n_facts: int, steps: int, lr: float = 1e-3,
                   batch: int = 64, temp: float = 0.07, hard_negatives: bool = False) -> Dict:
    """Train `model.cue_proj` (a MemoryLM key head) to address slots rather than sentences.

    Two things make an address correct, and the objective has to supply both:

      SEPARATION  two facts sharing a template are different memories and must be far apart
      INVARIANCE  a restatement or a revision of one fact is the same memory and must be close,
                  even though a revision changes the very value the sentence is about

    Positives are every item of one slot -- the teaching, its restatements, its revisions, its
    held-out questions, its support forms. Negatives are the rest of the batch, and with
    `hard_negatives` the batch is deliberately packed with same-subject and same-attribute near
    misses instead of being drawn uniformly.

    Meta-training's own contrastive term does neither: it pairs a quiz with its teaching inside a
    4-teaching episode, where a near miss almost never appears. Training happens on the TRAIN
    entity split; evaluation is on the disjoint EVAL split, so this measures generalisation.

    `hard_negatives` defaults to OFF because it did not earn its place. Measured over 3 seeds
    against uniform batches it moved retrieval@1 by -0.028 against a pooled standard deviation of
    0.050 -- inside the noise. The flag stays so the claim can be re-tested on a real model, where
    a few hundred training items is no longer the whole distribution. See RESULTS_STAGE1.md §4.3.
    """
    base, events, owner_of = build_stream(n_facts, seed + 1000)
    texts, owners, ent, attr = _items(base, events, owner_of)
    H, M = _cache_states(model, texts)

    by_slot: Dict[int, List[int]] = {}
    by_ent: Dict[str, List[int]] = {}
    by_attr: Dict[str, List[int]] = {}
    for i, o in enumerate(owners):
        by_slot.setdefault(int(o), []).append(i)
        by_ent.setdefault(ent[i], []).append(i)
        by_attr.setdefault(attr[i], []).append(i)

    own = torch.as_tensor(owners, device=H.device)
    opt = torch.optim.AdamW(model.cue_proj.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses: List[float] = []
    for _ in range(steps):
        if hard_negatives:
            sel = _hard_batch(rng, owners, ent, attr, by_slot, by_ent, by_attr,
                              n_slots=max(2, batch // 12), per_slot=2)[:batch]
        else:
            sel = list(rng.choice(len(texts), size=min(batch, len(texts)), replace=False))
        if len(sel) < 4:
            continue
        idx = torch.as_tensor(sel, device=H.device)
        z = F.normalize(model.cue_proj(H[idx], M[idx]), dim=-1)
        sim = z @ z.T / temp
        pos = (own[idx][:, None] == own[idx][None, :]).float()
        eye = torch.eye(len(idx), device=H.device)
        pos = pos * (1 - eye)                                   # a key is not its own positive
        keep = pos.sum(-1) > 0
        if not bool(keep.any()):
            continue
        logp = (sim - 1e9 * eye).log_softmax(-1)
        loss = -((logp * pos).sum(-1)[keep] / pos.sum(-1)[keep]).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return {"steps": steps, "items": len(texts), "hard_negatives": hard_negatives,
            "final_loss": float(np.mean(losses[-25:])) if losses else float("nan")}
