"""How a teaching becomes an address.

The addressing failure measured in RESULTS_STAGE1.md §4 -- roughly one write in four landing
on another fact's page -- is downstream of one design decision: the key is a projection of the
*mean-pooled* hidden state of the whole sentence. Two things follow from that, and both are
wrong for addressing.

A key must SEPARATE facts that share a template.

    "In lease.lock, the serializer is msgpack."
    "In lease.lock, the timeout is 5 seconds."

These are two different memories. They differ in two tokens out of a dozen, and mean pooling
gives those two tokens a twelfth of the weight.

A key must be INVARIANT to the value it stores.

    "In lease.lock, the serializer is msgpack."
    "Actually, lease.lock's serializer is now protobuf."

These are the same memory, revised. A page write that does not land on the original leaves two
contradictory pages, which is the failure a user hits the first time they correct the system.
Mean pooling puts the value -- the one part that is supposed to change -- directly into the
address.

So the address should be a function of the SLOT the memory occupies, not of the sentence.
`AttnKeyHead` lets the head learn which tokens carry the slot: a learned query attends over the
token states and pools by that attention, so it can concentrate on the subject and attribute and
discount the value. `MeanKeyHead` is the original, kept so the comparison is measurable rather
than assumed -- `hebb.experiments.collisions --cue mean|attn` runs both.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanKeyHead(nn.Module):
    """The original: mean-pool the token states, then an MLP. Every token weighted alike."""

    kind = "mean"

    def __init__(self, d: int, key_dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, key_dim))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask[..., None].float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.proj(pooled)


class AttnKeyHead(nn.Module):
    """Learned attention pooling: the head decides which tokens are the address.

    A small set of learned queries attends over the frozen token states; the result is pooled by
    that attention and projected. Nothing here is told which tokens are the subject -- the
    contrastive objective in `keytrain.py` supplies that pressure, by making restatements and
    revisions of one fact positives for each other and same-template different facts negatives.
    Multiple queries because a slot is (subject, attribute) and one query would have to encode
    both in one attention pattern.
    """

    kind = "attn"

    def __init__(self, d: int, key_dim: int, n_queries: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, d) * (d ** -0.5))
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(n_queries * d, d), nn.GELU(), nn.Linear(d, key_dim))
        self.scale = d ** -0.5

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ln(h)                                     # [B,T,d]
        k, v = self.k(x), self.v(x)                        # [B,T,d]
        att = torch.einsum("qd,btd->bqt", self.q, k) * self.scale
        att = att.masked_fill(~mask[:, None, :].bool(), float("-inf")).softmax(-1)
        pooled = torch.einsum("bqt,btd->bqd", att, v)      # [B,Q,d]
        return self.proj(pooled.flatten(1))

    @torch.no_grad()
    def attention(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-token attention weights, averaged over queries. For inspecting what it keys on."""
        x = self.ln(h)
        att = torch.einsum("qd,btd->bqt", self.q, self.k(x)) * self.scale
        att = att.masked_fill(~mask[:, None, :].bool(), float("-inf")).softmax(-1)
        return att.mean(1)


def build_key_head(kind: str, d: int, key_dim: int) -> nn.Module:
    if kind == "attn":
        return AttnKeyHead(d, key_dim)
    if kind == "mean":
        return MeanKeyHead(d, key_dim)
    raise ValueError(f"unknown key head {kind!r}")
