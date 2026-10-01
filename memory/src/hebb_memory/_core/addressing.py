"""Page addressing: deciding which page a teaching belongs to.

The failure this module exists to fix
-------------------------------------
Content keys come from the frozen LM's mean-pooled hidden state of the teaching sentence.
Over templated text those keys are nearly parallel, because the template is most of the
sentence. In the Qwen2.5-0.5B stream the observed cosines between *unrelated* facts were
0.97-0.99:

    "Remember: Kai's home city is Quito."        matched page 026 (cos 0.97)
        -- page 026 held "Gus's home city is Perth"
    "Remember: Gus's favourite food is ramen."   matched page 030 (cos 0.99)  -> ||dM||=0.000

Both are wrong writes. The first overwrites Gus's city with Kai's; the second lands on a
page whose content already satisfies the loss, so nothing is written and the fact is lost.
That is where the residual forgetting comes from, and no threshold fixes it: at 0.90 the
unrelated facts merge, at 0.97 some still merge while genuine restatements start splitting
into duplicate pages. The threshold is being asked to separate two distributions that
overlap.

Two changes were tried, measured over 3 seeds by `hebb.experiments.collisions`. Neither is
the fix; one of them is a real improvement.

CENTERING (on by default)
    Subtract the running mean of all keys seen. The shared template is close to the
    population mean, so removing it leaves the component that identifies *which* fact this
    is. It reliably widens the gap between same-fact and different-fact cosines -- on the toy
    LM, 0.166 -> 0.425 with an untrained key head, 0.839 -> 0.897 with a trained one, on
    every seed. On its own it does NOT improve the routing decision: at a matched page count
    it lands on the plain-threshold curve. A wider gap is not the same thing as a better
    decision.

MARGIN (0.10 by default)
    Require the best page to beat the runner-up before treating a write as a revision of it.
    A genuine restatement is close to one page and far from the rest; a collision is roughly
    equidistant from many, because what made it "close" was the template every page shares.
    With centering this is the only rule measured that beats plain thresholding on BOTH write
    errors at once: at ~55 pages for 45 facts, false merges 0.227 against 0.277 and missed
    revisits 0.402 against 0.485. It dominates by a similar margin on the untrained key head
    too, so it is not an artefact of one key space.

    Rules must be compared at a matched PAGE COUNT, never at a matched threshold -- the
    threshold is what slides along the trade-off curve, so two rules at the same threshold
    are two different operating points. `addressing_table.pareto` does this comparison.

WHAT ACTUALLY DOMINATES
    Whether the key projection was ever trained on the addressing task. Untrained, the gap is
    0.166 +/- 0.048 and retrieval@1 is 0.244 +/- 0.084; trained on the hard negatives
    (`keytrain.py`), 0.839 +/- 0.014 and 0.664 +/- 0.048. That is the only difference here
    outside the run-to-run spread by a wide margin. Retrieval@1 is nearly flat across every
    routing rule (0.594-0.689): the rule governs what gets destroyed on write, not recall.
    And the best operating point still merges one write in four. See RESULTS_STAGE1.md §4.

Both are decisions about geometry, not about the model, so they are testable on their own
(`tests/test_addressing.py`) and measurable on their own (`hebb.experiments.collisions`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Match:
    page: Optional[int]      # best allocated page, None if memory is empty
    cos: float               # similarity to it, in the addressing space
    margin: float            # how far it beat the runner-up (inf when it is the only page)
    revisit: bool            # True -> revise that page, False -> allocate a new one


class KeySpace(nn.Module):
    """Maps raw projected keys to addresses, and decides revisit-vs-allocate.

    Keys are stored raw (`z`). Centering happens at comparison time against the *current*
    running mean, so a stored key can never go stale relative to the mean that was current
    when it was written.
    """

    def __init__(self, key_dim: int, center: bool = True, threshold: float = 0.9,
                 margin: float = 0.10, warmup: int = 8, rule: str = "threshold", z: float = 2.5):
        super().__init__()
        self.center, self.threshold, self.margin, self.warmup = center, threshold, margin, warmup
        self.rule, self.z = rule, z
        self.register_buffer("key_sum", torch.zeros(key_dim))
        self.register_buffer("key_n", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def observe(self, z: torch.Tensor) -> None:
        """Accumulate the running mean over every key the model has ever computed."""
        z = z.detach().reshape(-1, z.shape[-1]).float()
        self.key_sum += z.sum(0).to(self.key_sum.dtype)
        self.key_n += z.shape[0]

    @property
    def mean(self) -> torch.Tensor:
        if not self.center or int(self.key_n) < self.warmup:
            return torch.zeros_like(self.key_sum)
        return self.key_sum / float(int(self.key_n))

    def address(self, z: torch.Tensor) -> torch.Tensor:
        """Raw key -> unit address vector. Cosines are taken in this space."""
        return F.normalize(z - self.mean, dim=-1)

    def match(self, z_new: torch.Tensor, z_pages: torch.Tensor) -> Match:
        """Decide where a single new teaching goes among the given allocated page keys.

        Two rules are available.

        `threshold` is an absolute cosine, and it cannot hold up as memory fills. With N keys on
        a sphere, the nearest neighbour of a *random* new key gets closer as N grows, purely by
        density. A fixed bar therefore becomes progressively more permissive in the only sense
        that matters, and the measured false-merge rate rises from .23 at eighteen memories to
        .52 at ninety-one (RESULTS_STAGE1.md §4.4).

        `zscore` asks a different question: not "is this match close?" but "is this match closer
        than this memory's own population would produce by chance?" The bar is the mean and
        spread of the similarities to every allocated page, both of which rise with occupancy, so
        the rule re-calibrates itself as the memory fills instead of being calibrated once.
        """
        if z_pages.numel() == 0:
            return Match(None, 0.0, float("inf"), False)
        cos = self.address(z_pages) @ self.address(z_new)
        order = cos.argsort(descending=True)
        best = int(order[0]); c1 = float(cos[best])
        c2 = float(cos[order[1]]) if len(order) > 1 else -1.0
        gap = float("inf") if len(order) == 1 else c1 - c2
        if self.rule == "zscore":
            if cos.numel() < 4:
                # too few pages for a population statistic; fall back to the absolute bar
                return Match(best, c1, gap, c1 >= self.threshold)
            mu = float(cos.mean())
            sd = float(cos.std(unbiased=False))
            zc = (c1 - mu) / (sd + 1e-6)
            return Match(best, c1, gap, zc >= self.z)
        return Match(best, c1, gap, c1 >= self.threshold and gap >= self.margin)

    def reset_stats(self) -> None:
        self.key_sum.zero_(); self.key_n.zero_()

    def extra_repr(self) -> str:
        return f"center={self.center}, threshold={self.threshold}, margin={self.margin}"
