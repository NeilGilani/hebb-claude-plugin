"""Composed rule names ("premium refund window") past the eight hand-written ones."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .rules import SLOTS, Rule, TenantWorld, tenant_name

#: Generated slots reuse these value sets. Five options each keeps the chance rate comparable
#: to the hand-written SLOTS, so a change in accuracy across the sweep is the rule COUNT and
#: not a change in how hard each individual question is.
_VALUE_SETS = [vals for _, vals in SLOTS]

_FIELDS = [
    "refund window", "escalation tier", "approval limit", "support channel", "data region",
    "retention period", "billing cycle", "priority sla", "warranty term", "notice period",
    "dispute window", "onboarding path", "audit frequency", "backup schedule", "access review",
    "payment terms", "renewal notice", "trial length", "seat limit", "overage policy",
    "export format", "encryption standard", "incident sla", "maintenance window", "patch cycle",
    "support tier", "training allowance", "consulting rate", "travel policy", "expense cap",
]
_QUALIFIERS = ["standard", "premium", "legacy", "enterprise", "pilot", "regional", "extended"]


def build_slots(n: int) -> List[Tuple[str, List[str]]]:
    """`n` distinct slots, each with five conflicting values.

    The repository ships eight hand-written ones, which caps rules-per-tenant at eight and
    makes this whole experiment impossible. Past that, names are composed from a field and a
    qualifier -- "premium refund window" -- so they stay readable to a model that has to tell
    them apart, rather than becoming slot_0047.
    """
    if n <= len(SLOTS):
        return list(SLOTS[:n])
    out = list(SLOTS)
    i = 0
    while len(out) < n:
        field = _FIELDS[i % len(_FIELDS)]
        qual = _QUALIFIERS[(i // len(_FIELDS)) % len(_QUALIFIERS)]
        name = f"{qual} {field}"
        if name not in {s for s, _ in out}:
            out.append((name, list(_VALUE_SETS[i % len(_VALUE_SETS)])))
        i += 1
        if i > n * 20:                      # ran out of distinct names
            raise ValueError(f"cannot build {n} distinct slots from the vocabulary available")
    return out[:n]


def make_wide_world(n_tenants: int, slots_per_tenant: int, seed: int = 0) -> TenantWorld:
    """Like `make_world`, but able to go past eight rules per tenant.

    Every tenant gets EVERY slot, so rules-per-tenant is exactly `slots_per_tenant` and the
    sweep axis means what it says. Values rotate by tenant index so neighbours disagree by
    construction -- if they agreed, one shared model would answer everything and the benchmark
    would flatter every arm at once.
    """
    slots = build_slots(slots_per_tenant)
    rules: List[Rule] = []
    by_tenant: Dict[str, List[Rule]] = {}
    by_slot: Dict[str, List[Rule]] = {}
    for i in range(n_tenants):
        name = tenant_name(i)
        by_tenant[name] = []
        for si, (slot, values) in enumerate(slots):
            value = values[(i + si) % len(values)]
            r = Rule(name, slot, value)
            rules.append(r)
            by_tenant[name].append(r)
            by_slot.setdefault(slot, []).append(r)
    return TenantWorld(n_tenants, slots_per_tenant, rules, by_tenant, by_slot).fill_rivals()
