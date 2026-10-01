"""The multi-tenant collision benchmark.

THE QUESTION THIS EXISTS TO ANSWER. Every way of teaching a model something new survives the
small version of the problem. Prompting works for ten rules. Fine-tuning works for one
customer. The interesting question is which approach is still standing at a thousand
customers, each with its own evolving behaviour -- because that is where a memory layer stops
being an improvement and becomes the only option, or fails to.

THE DESIGN. Every tenant is asked the SAME questions and must give DIFFERENT answers.

    acme    "refund window" -> "60 days"
    globex  "refund window" -> "14 days"

That collision is the whole point. A shared model that learns both has no way to keep them
apart; the second write lands on top of the first. This turns the abstract worry about
forgetting into a concrete, enterprise-legible failure: **tenant A's data coming out of
tenant B's account.**

So the headline metric here is not accuracy. It is LEAKAGE -- how often a query for one
tenant is answered with another tenant's value. An approach can be accurate on average and
still be unshippable if it leaks, and no amount of averaging makes that acceptable.

Scoring is forced choice among the candidate values for a slot, so a weak model still yields
a meaningful signal: we are asking "whose answer did it give?", not "can it generate text".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Deliberately business-shaped: these are the kind of per-customer policies a support agent,
# a coding agent or an ops agent is configured with, and the kind a customer would be alarmed
# to see cross accounts.
SLOTS = [
    ("refund window", ["14 days", "30 days", "60 days", "90 days", "7 days"]),
    ("escalation tier", ["tier one", "tier two", "tier three", "the duty manager", "the owner"]),
    ("approval limit", ["500 dollars", "2000 dollars", "10000 dollars", "50 dollars", "no limit"]),
    ("support channel", ["email", "phone", "chat", "the portal", "slack"]),
    ("data region", ["us east", "eu west", "ap south", "ca central", "on premises"]),
    ("retention period", ["one year", "three years", "seven years", "thirty days", "forever"]),
    ("billing cycle", ["monthly", "quarterly", "annually", "weekly", "on completion"]),
    ("priority sla", ["one hour", "four hours", "one day", "three days", "next release"]),
]

TENANT_NAMES = ["acme", "globex", "initech", "umbrella", "soylent", "stark", "wayne", "cyberdyne",
                "tyrell", "aperture", "hooli", "piedpiper", "vandelay", "wonka", "gringotts",
                "duffbeer", "krusty", "planex", "oscorp", "abstergo"]


def tenant_name(i: int) -> str:
    """Stable, readable names; falls back to a numbered form past the hand-written list."""
    return TENANT_NAMES[i] if i < len(TENANT_NAMES) else f"tenant{i:04d}"


@dataclass
class Rule:
    tenant: str
    slot: str
    value: str
    #: the slot's other values in this world -- what forced choice will hold against `value`.
    #: Filled by the world builders; a write may use them (LMMemoryConfig.write_value_contrast).
    rivals: List[str] = field(default_factory=list)

    def teach_text(self) -> str:
        return f"For {self.tenant}, the {self.slot} is {self.value}."

    def support(self) -> List[Tuple[str, str]]:
        return [(f"Note: for {self.tenant}, {self.slot}?", f" {self.value}"),
                (f"{self.tenant} {self.slot} is", f" {self.value}")]

    def query(self) -> str:
        # Disjoint in surface form from `support`, so a model cannot pass by pattern-matching
        # the sentence it was taught (the leak that RESULTS_STAGE1.md records finding).
        return f"Question: For {self.tenant}, what is the {self.slot}?\nAnswer:"

    def answer(self) -> str:
        return f" {self.value}"


@dataclass
class TenantWorld:
    n_tenants: int
    slots_per_tenant: int
    rules: List[Rule]
    by_tenant: Dict[str, List[Rule]]
    by_slot: Dict[str, List[Rule]]

    def candidates(self, slot: str) -> List[str]:
        """Every value any tenant holds for this slot -- the forced-choice option set."""
        return sorted({r.value for r in self.by_slot[slot]})

    def owner_of(self, slot: str, value: str) -> List[str]:
        return sorted({r.tenant for r in self.by_slot[slot] if r.value == value})

    def fill_rivals(self) -> "TenantWorld":
        """Give every rule the slot's other values, exactly the scorer's candidate set."""
        for r in self.rules:
            r.rivals = [v for v in self.candidates(r.slot) if v != r.value]
        return self

    def stream(self) -> List[Rule]:
        """Teaching order: interleaved across tenants, as real traffic would arrive.

        Grouping by tenant would flatter sequential fine-tuning, because the last tenant
        taught is the one still in the weights.
        """
        out: List[Rule] = []
        i = 0
        while True:
            added = False
            for t in self.by_tenant:
                if i < len(self.by_tenant[t]):
                    out.append(self.by_tenant[t][i]); added = True
            if not added:
                break
            i += 1
        return out


def make_world(n_tenants: int, slots_per_tenant: int = 3, seed: int = 0) -> TenantWorld:
    """Build a world where tenants genuinely collide.

    Values are assigned so that tenants sharing a slot mostly DISAGREE -- if they happened to
    agree, leakage would be undetectable and the benchmark would flatter every approach.
    """
    rng = np.random.default_rng(seed)
    slots_per_tenant = min(slots_per_tenant, len(SLOTS))
    rules: List[Rule] = []
    by_tenant: Dict[str, List[Rule]] = {}
    by_slot: Dict[str, List[Rule]] = {}
    for i in range(n_tenants):
        name = tenant_name(i)
        chosen = rng.choice(len(SLOTS), size=slots_per_tenant, replace=False)
        by_tenant[name] = []
        for si in chosen:
            slot, values = SLOTS[int(si)]
            # rotate through the value list by tenant index so neighbours differ by construction
            value = values[(i + int(si)) % len(values)]
            r = Rule(name, slot, value)
            rules.append(r)
            by_tenant[name].append(r)
            by_slot.setdefault(slot, []).append(r)
    return TenantWorld(n_tenants, slots_per_tenant, rules, by_tenant, by_slot).fill_rivals()


def collision_report(w: TenantWorld) -> Dict:
    """How much genuine conflict the world contains. A world without collisions cannot
    measure leakage, so this is checked before any result is believed."""
    shared = {s: rs for s, rs in w.by_slot.items() if len({r.tenant for r in rs}) > 1}
    conflicting = 0
    total_pairs = 0
    for s, rs in shared.items():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                total_pairs += 1
                if rs[a].value != rs[b].value:
                    conflicting += 1
    return {"n_tenants": w.n_tenants, "n_rules": len(w.rules),
            "slots_in_use": len(w.by_slot),
            "shared_slots": len(shared),
            "tenant_pairs_sharing_a_slot": total_pairs,
            "pairs_that_disagree": conflicting,
            "disagreement_rate": conflicting / max(total_pairs, 1)}


def language_corpus(n: int = 900, seed: int = 0) -> List[str]:
    """Sentences that teach the toy model the LANGUAGE of this benchmark, never its answers.

    The model has to know the templates, the slot names and the vocabulary of values before the
    stream starts, or every arm scores at chance for a reason that has nothing to do with
    memory. It must NOT learn any tenant->value binding that will later be evaluated, so this
    corpus uses a disjoint pool of training tenant names ("northwind", "contoso", ...) and
    reshuffles values freely, which makes the binding uninformative even for those names.
    """
    rng = np.random.default_rng(seed + 777)
    train_tenants = [f"train{j:03d}" for j in range(60)] + [
        "northwind", "contoso", "fabrikam", "adventure", "litware", "proseware"]
    out: List[str] = []
    for _ in range(n):
        t = str(rng.choice(train_tenants))
        slot, values = SLOTS[int(rng.integers(len(SLOTS)))]
        v = str(rng.choice(values))
        r = Rule(t, slot, v)
        out.append(r.teach_text())
        for p_, a_ in r.support():
            out.append(p_ + a_)
        out.append(r.query() + r.answer())
    return out


def eval_tenant_names_are_disjoint_from_training() -> bool:
    """The training pool and the evaluated pool must not overlap, or pretraining leaks."""
    train = {s.split()[1] for s in language_corpus(40, 0) if s.startswith("For ")}
    train = {t.rstrip(",") for t in train}
    return not (train & set(TENANT_NAMES))


TRAIN_TENANTS = [f"train{j:03d}" for j in range(60)] + [
    "northwind", "contoso", "fabrikam", "adventure", "litware", "proseware"]


def training_rule(rng, slot_pool=None) -> Rule:
    """A rule about a TRAINING tenant. Never one that will be evaluated.

    `slot_pool` defaults to the eight hand-written SLOTS. Pass a wider pool (run.META_SLOTS =
    "composed") so meta-training also sees composed rule names; otherwise "premium refund
    window"-shaped rules are unseen until the crossover asks about thirty of them.
    """
    pool = SLOTS if slot_pool is None else slot_pool
    t = str(rng.choice(TRAIN_TENANTS))
    slot, values = pool[int(rng.integers(len(pool)))]
    return Rule(t, slot, str(rng.choice(values)))


def rule_to_teaching(r: Rule, index: int, values: Optional[Sequence[str]] = None):
    """Rule -> the Teaching shape MemoryLM's write path and meta-training expect.

    `values` is the slot's full candidate set; the other members become the teaching's
    distractors, which is what the forced-choice scorer will hold against the answer."""
    from .._core.data import Teaching
    return Teaching(index=index, kind="fact", entity=r.tenant, attribute=r.slot, value=r.value,
                    text=r.teach_text(), support=r.support(),
                    quizzes=[(r.query(), r.answer())],
                    distractors=[v for v in (values or []) if v != r.value])


def training_episode(n: int, rng, slot_pool=None, mode: str = "tenants") -> List:
    """One meta-training episode of n teachings.

    `mode` decides what the n pages differ BY, which is the skill the read path practises:

      tenants  n distinct training tenants (slots may repeat). The read head learns to open
               the page of the tenant named in the question -- the `pages` arm's problem.
      keyed    ONE training tenant, n distinct slots. This is what the `pages_keyed` arm --
               the product -- actually faces: the tenant is given, and the candidates are
               that tenant's own pages, which differ only by rule. An episode of distinct
               tenants never shows the read path this case.
      mixed    a coin flip per episode between the two.

    Until 2026-09-15 only `tenants` existed, and the keyed eval was read by a head that had
    never once chosen among one tenant's pages.
    """
    if mode == "mixed":
        mode = "keyed" if rng.random() < 0.5 else "tenants"
    if mode not in ("tenants", "keyed"):
        raise ValueError(f"unknown episode mode {mode!r}")
    used, rules = set(), []
    pool = SLOTS if slot_pool is None else slot_pool
    values_of = {slot: list(vals) for slot, vals in pool}
    tenant = str(rng.choice(TRAIN_TENANTS)) if mode == "keyed" else None
    while len(rules) < n:
        r = training_rule(rng, slot_pool)
        if tenant is not None:
            r = Rule(tenant, r.slot, r.value)
        if (r.tenant, r.slot) in used:
            continue
        used.add((r.tenant, r.slot))
        rules.append(r)
    return [rule_to_teaching(r, i, values_of.get(r.slot)) for i, r in enumerate(rules)]
