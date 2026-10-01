"""Neural Memory attached to a frozen causal language model (Stage 1).

Differences from v0.1 (`hebb/ml/neural_memory.py`):
  * the slow core is a pretrained HF decoder; only read heads, mem_in, page_init,
    log_lr and the cue projection are trained (meta-training);
  * pages are CONTENT-ADDRESSED: the key is a learned projection of the frozen LM's
    mean-pooled hidden state of the text (a teaching when writing, a question when
    reading).  No task label is given at any point;
  * reads retrieve the top-k pages by cosine similarity and attend over all of their
    slots (k = 2 lets a question that needs two teachings read both pages);
  * a contrastive term in meta-training pulls question keys toward the key of the
    teaching that answers them, which is what makes content addressing work.

Everything else (RMS-normalised task-loss gradient writes with meta-learned rates,
allocation, usage/importance telemetry) is the same mechanism.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .addressing import KeySpace, Match
from .keyhead import build_key_head
from .data import Teaching, teaching_from_text


@dataclass
class LMMemoryConfig:
    n_pages: int = 256
    page_size: int = 8
    slot_dim: int = 64
    key_dim: int = 64
    read_every: int = 2          # read head after every k-th decoder layer
    read_k: int = 2              # pages retrieved per read
    read_temp: float = 0.1
    heads: int = 4
    inner_lr: float = 0.3
    passes: int = 8
    surprise_skip: float = 0.05
    match_threshold: float = 0.90      # min cosine, in the centered addressing space, to revise a page
    match_margin: float = 0.10         # ...and beat the runner-up page by this much. Compared at a
                                       # matched page count this is better on BOTH write errors than
                                       # moving the threshold alone (RESULTS_STAGE1.md §4)
    center_keys: bool = True           # subtract the running key mean before comparing (see addressing.py)
    cue_kind: str = "mean"             # "mean" | "attn" -- how a teaching becomes an address (keyhead.py)
    rigidity: float = 0.5
    contrastive_weight: float = 1.0
    max_len: int = 96
    value_contrast: float = 0.0        # weight of a meta-loss term that scores the quiz answer
                                       # AGAINST the teaching's distractor values (Teaching.distractors)
                                       # with a softmax over their total log-probabilities. The plain
                                       # quiz loss only raises the right value; forced choice is won
                                       # by beating the four rivals, and nothing trained for that.
    write_lr_scale: float = 1.0        # multiplies the learned inner-loop step in DEPLOYMENT writes
                                       # only. Meta-training learns log_lr for 4-20 steps on its
                                       # own episodes; at deployment on Qwen2.5-0.5B the same step
                                       # made the write loss RISE from 5.6 to 9.2 over 20 passes
                                       # (results/crossover-kt-shared-inner-s0/forced-base.json).
    write_fact: bool = True            # deployment writes from "Fact: <text>" + support (every result
                                       # so far). False writes from the support pairs alone, which is
                                       # what meta-training wrote from unless meta_write_fact was on:
                                       # a reader trained on support-only pages reads support-only pages.
    write_keep_best: bool = False      # a deployment write returns the page with the lowest write
                                       # loss along its trajectory instead of the last one
    write_value_contrast: float = 0.0  # weight of a term in the DEPLOYMENT write (`experience`)
                                       # that ranks the value against the teaching's distractors
                                       # on each support prompt, the same softmax over total
                                       # log-probabilities as value_contrast. A write that only
                                       # raises the answer leaves the frozen model's preference
                                       # among rivals intact; this one writes the ranking.
    meta_write_fact: bool = False       # meta-training writes from the "Fact: <text>" line plus the
                                       # support pairs, exactly what `experience` writes from. Off
                                       # reproduces every result before 2026-09-16, where the reader
                                       # was trained on support-only pages and read fact+support ones.
    neg_bank: int = 0                  # extra contrastive negatives kept from earlier episodes.
                                       # 0 = off. The in-episode contrastive term sees at most
                                       # episode_len pages, so the key head learns to pick one
                                       # of ~4; at eval it faces 128-256 and collapses to chance
                                       # (results/crossover-qwen05b). A bank of recent keys makes
                                       # every meta-step a many-way discrimination instead.


class ReadHead(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)   # starts as identity

    def forward(self, h, mem):
        B, T, D = h.shape; S = mem.shape[1]
        x = self.ln(h)
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(mem).view(B, S, self.h, self.dk).transpose(1, 2)
        v = self.v(mem).view(B, S, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2) / math.sqrt(self.dk)).softmax(-1)
        return self.o((att @ v).transpose(1, 2).reshape(B, T, D))


def _decoder_layers(lm) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = lm
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot find decoder layers on this model")


class MemoryLM(nn.Module):
    name = "memory_lm"

    def __init__(self, lm, tokenizer, cfg: LMMemoryConfig):
        super().__init__()
        self.lm, self.tok, self.cfg = lm, tokenizer, cfg
        for p in self.lm.parameters():
            p.requires_grad_(False)
        d = lm.config.hidden_size
        self.d = d
        layers = _decoder_layers(lm)
        self.read_layers = [i for i in range(len(layers)) if (i + 1) % cfg.read_every == 0]
        self.read_heads = nn.ModuleList([ReadHead(d, cfg.heads) for _ in self.read_layers])
        self.mem_in = nn.Linear(cfg.slot_dim, d)
        self.page_init = nn.Parameter(torch.randn(cfg.page_size, cfg.slot_dim) * 0.1)
        self.log_lr = nn.Parameter(torch.full((cfg.slot_dim,), math.log(cfg.inner_lr)))
        self.cue_proj = build_key_head(cfg.cue_kind, d, cfg.key_dim)
        self.keys = KeySpace(cfg.key_dim, cfg.center_keys, cfg.match_threshold, cfg.match_margin)
        # Detached keys from earlier meta-episodes, used only as contrastive negatives. Never
        # saved -- it is training scaffolding, not model state.
        self._neg_bank: List[torch.Tensor] = []
        N, P, dv = cfg.n_pages, cfg.page_size, cfg.slot_dim
        self.register_buffer("M", torch.zeros(N, P, dv)); self.register_buffer("K", torch.zeros(N, cfg.key_dim))
        self.register_buffer("allocated", torch.zeros(N, dtype=torch.bool))
        self.register_buffer("owner", torch.full((N,), -1, dtype=torch.long))
        self.register_buffer("usage", torch.zeros(N)); self.register_buffer("importance", torch.zeros(N))
        self.register_buffer("write_count", torch.zeros(N, dtype=torch.long)); self.register_buffer("read_count", torch.zeros(N, dtype=torch.long))
        self.register_buffer("last_delta", torch.zeros(N))
        self._mem: Optional[torch.Tensor] = None
        self.stream_step = 0; self.n_evictions = 0
        for j, i in enumerate(self.read_layers):
            layers[i].register_forward_hook(self._make_hook(j))

    # ---------------------------------------------------------------- hooks
    def _make_hook(self, j):
        def hook(mod, args, out):
            if self._mem is None:
                return out
            h = out[0] if isinstance(out, tuple) else out
            # The frozen LM may be bf16 while the read head is float32 -- it is trained, and
            # bf16 costs too much precision there. Run the head in ITS dtype and cast the
            # result back, rather than casting only the memory: feeding a bf16 hidden state
            # into an fp32 Linear raises "mat1 and mat2 must have the same dtype".
            hd = next(self.read_heads[j].parameters()).dtype
            r = self.read_heads[j](h.to(hd), self._mem.to(hd))
            h = h + r.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    @property
    def device(self):
        return self.page_init.device

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("lm.")]

    # ------------------------------------------------------------- encoding
    def encode_pairs(self, pairs: Sequence[Tuple[str, str]]):
        """Right-padded ids, attention mask, labels (-100 outside answer tokens)."""
        ids, labs = [], []
        for prompt, answer in pairs:
            p = self.tok(prompt, add_special_tokens=False).input_ids
            a = self.tok(answer, add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            seq = (p + a)[-self.cfg.max_len:]
            n_p = max(0, len(seq) - len(a))
            ids.append(seq); labs.append([-100] * n_p + seq[n_p:])
        L = max(len(s) for s in ids)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in ids], device=self.device)
        mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids], device=self.device)
        labels = torch.tensor([l + [-100] * (L - len(l)) for l in labs], device=self.device)
        return input_ids, mask, labels

    def _forward(self, input_ids, mask, mem: Optional[torch.Tensor]):
        self._mem = mem
        try:
            out = self.lm(input_ids=input_ids, attention_mask=mask, output_hidden_states=mem is None)
        finally:
            self._mem = None
        return out

    def cue_proj_dtype(self) -> torch.dtype:
        return next(self.cue_proj.parameters()).dtype

    def keys_for(self, texts: Sequence[str]) -> torch.Tensor:
        """Content keys: frozen-LM mean-pooled last hidden state → learned projection.

        Returned RAW (not normalised). Cosines are taken in the centered space produced by
        `self.keys.address`, so that the template component every teaching shares does not
        dominate every comparison. See `addressing.py`.
        """
        h, mask = self.token_states(texts)
        # The frozen LM may run in bf16 while the memory's own projections are float32 (they
        # are trained, and bf16 moments lose too much). Cast at the boundary rather than
        # forcing one dtype on both: a bf16 hidden state into an fp32 Linear is a hard error.
        h = h.to(self.cue_proj_dtype())
        z = self.cue_proj(h, mask)
        self.keys.observe(z)
        return z

    @torch.no_grad()
    def token_states(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frozen-LM per-token last hidden states and their mask.

        Per-token rather than pooled: pooling is the key head's decision now, because which
        tokens carry the address is exactly what the head has to learn (see keyhead.py).
        """
        enc = self.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False,
                       truncation=True, max_length=self.cfg.max_len).to(self.device)
        # Keys need the final hidden state only. Running the causal-LM wrapper computes the
        # vocabulary projection as well -- 152k x 896 on Qwen2.5-0.5B, about a quarter of a
        # forward pass and all of it discarded -- so use the bare transformer when the model
        # exposes one. Its last_hidden_state is the post-norm state hidden_states[-1] was.
        base = getattr(self.lm, "base_model", None)
        if base is not None and base is not self.lm:
            h = base(**enc).last_hidden_state
        else:
            h = self.lm(**enc, output_hidden_states=True).hidden_states[-1]
        return h.float(), enc["attention_mask"]

    @torch.no_grad()
    def pooled(self, texts: Sequence[str]) -> torch.Tensor:
        """Mean-pooled last hidden state. Kept for callers that want the raw material."""
        h, mask = self.token_states(texts)
        m = mask[..., None].float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    @staticmethod
    def _loss(logits, labels):
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100)

    @staticmethod
    def _seq_logprob(logits, labels) -> torch.Tensor:
        """Total log-probability of each row's answer tokens, [B]. The quantity forced-choice
        scoring ranks candidates by (tenants/scoring.py), so a loss on it trains the test."""
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = labels[:, 1:]
        keep = tgt != -100
        # encode_pairs ends every answer with EOS; the scorer does not count it, so drop the
        # last answer token of each row to rank by exactly what forced_choice ranks by.
        keep = keep & (keep.long().cumsum(-1) < keep.long().sum(-1, keepdim=True))
        tok = lp.gather(-1, tgt.clamp(min=0)[..., None]).squeeze(-1)
        return (tok * keep).sum(-1)

    def _write_set(self, t) -> List[Tuple[str, str]]:
        """What a page is written from. `experience` has always used the fact line plus the
        support pairs; meta-training used the support pairs alone until `meta_write_fact`."""
        return [("Fact:", " " + t.text)] + list(t.support)

    # ----------------------------------------------------------------- read
    def read(self, qkeys: torch.Tensor, page_keys: torch.Tensor, pages: torch.Tensor) -> torch.Tensor:
        """qkeys [B,dk], page_keys [n,dk], pages [n,P,dv] -> memory tokens [B, k*P, d]."""
        n = page_keys.shape[0]
        k = min(self.cfg.read_k, n)
        cos = self.keys.address(qkeys) @ self.keys.address(page_keys).T          # [B,n]
        top, idx = cos.topk(k, dim=-1)
        w = (top / self.cfg.read_temp).softmax(-1)                 # [B,k]
        sel = pages[idx]                                           # [B,k,P,dv]
        mem = self.mem_in(sel) * w[..., None, None]                # weight pages
        return mem.flatten(1, 2), idx

    def _mem_for(self, prompts: Sequence[str], restrict: Optional[torch.Tensor] = None):
        """Read memory for these prompts.

        `restrict` limits which pages are candidates, as page indices. Content addressing then
        chooses among those rather than among everything allocated. That is what an explicit
        key buys: when the caller already knows which partition a request belongs to -- a
        multi-tenant request carries its tenant in the API key, it is never inferred from the
        question text -- the pages of every other partition are not candidates at all, and a
        cross-partition read stops being unlikely and becomes impossible.
        """
        if not bool(self.allocated.any()):
            return None, None
        alloc = self.allocated.nonzero().flatten()
        if restrict is not None:
            restrict = restrict.to(alloc.device)
            keep = torch.isin(alloc, restrict)
            if not bool(keep.any()):
                return None, None          # nothing written for this key yet; read nothing
            alloc = alloc[keep]
        qk = self.keys_for(prompts)
        mem, idx = self.read(qk, self.K[alloc], self.M[alloc])
        return mem, alloc[idx]

    # ---------------------------------------------------------------- write
    def _inner_loop(self, pairs, page, steps, create_graph, importance=0.0, allow_skip=False,
                    contrast=None, lr_scale=1.0, keep_best=False):
        """`contrast`: optional [(prompt, [answer, *rivals]), ...] and a weight; each step adds
        weight * cross-entropy over the candidates' total log-probabilities (answer first).
        `lr_scale` multiplies the learned step; `keep_best` returns the lowest-loss page seen
        (the write loss is then measured once more on the final page, so `losses` has
        steps + 1 entries)."""
        ids, mask, labels = self.encode_pairs(pairs)
        c_ids = c_mask = c_labels = None
        if contrast is not None and contrast[0]:
            groups, w = contrast
            c_pairs = [(prompt, c) for prompt, cands in groups for c in cands]
            c_ids, c_mask, c_labels = self.encode_pairs(c_pairs)
            n_cand = len(groups[0][1])
        lr = lr_scale * self.log_lr.exp() / (1.0 + self.cfg.rigidity * importance)
        if page.is_leaf:
            page = page.view_as(page)
        losses = []
        best = (float("inf"), page)
        for s in range(steps):
            mem = self.mem_in(page)[None].expand(ids.shape[0], -1, -1)
            loss = self._loss(self._forward(ids, mask, mem).logits, labels)
            losses.append(float(loss.detach()))
            if keep_best and losses[-1] < best[0]:
                best = (losses[-1], page)
            if allow_skip and self.cfg.surprise_skip > 0 and losses[-1] < self.cfg.surprise_skip:
                break
            if c_ids is not None:
                c_mem = self.mem_in(page)[None].expand(c_ids.shape[0], -1, -1)
                scores = self._seq_logprob(self._forward(c_ids, c_mask, c_mem).logits, c_labels)
                scores = scores.view(-1, n_cand)
                target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
                loss = loss + w * F.cross_entropy(scores, target)
            (g,) = torch.autograd.grad(loss, page, create_graph=create_graph)
            page = page - lr * g / (g.pow(2).mean().sqrt() + 1e-8)
        if keep_best:
            with torch.no_grad():
                mem = self.mem_in(page)[None].expand(ids.shape[0], -1, -1)
                final = float(self._loss(self._forward(ids, mask, mem).logits, labels))
            losses.append(final)
            if final < best[0]:
                best = (final, page)
            page = best[1]
        return page, losses

    def begin_stream(self):
        for b in ("M", "K", "usage", "importance", "last_delta"):
            getattr(self, b).zero_()
        self.allocated.zero_(); self.owner.fill_(-1); self.write_count.zero_(); self.read_count.zero_()
        self.stream_step = 0; self.n_evictions = 0

    def experience(self, t) -> Dict:
        """Learn one thing. Accepts a Teaching, or a plain string as a user would type it.

        A bare string is turned into a minimal teaching: the sentence is both the address
        (its content key) and the thing to be learned. That is the whole public surface of
        the product -- `model.experience("Invoices over $10k go to Dana.")`.
        """
        if isinstance(t, str):
            t = teaching_from_text(t, index=self.stream_step)
        self.eval()
        with torch.no_grad():
            key = self.keys_for([t.text])[0]
        events = []
        alloc = self.allocated.nonzero().flatten()
        m: Match = self.keys.match(key, self.K[alloc])
        j = int(alloc[m.page]) if m.page is not None else None
        cos, margin = m.cos, m.margin
        if not m.revisit:
            free = (~self.allocated).nonzero().flatten()
            if len(free):
                j = int(free[0])
            else:
                j = int(self.usage.argmin()); self.n_evictions += 1
                events.append({"type": "memory_evict", "page": j, "msg": f"memory full: evicting page {j:03d}"})
            self.K[j] = key; self.M[j] = self.page_init.detach(); self.allocated[j] = True
            self.owner[j] = t.index; self.importance[j] = 0.0; self.write_count[j] = 0
            events.append({"type": "memory_alloc", "page": j, "msg": f"allocated page {j:03d} ← “{t.text}”"})
            revisit = False
        else:
            revisit = True
            events.append({"type": "memory_match", "page": j, "cos": round(cos, 4), "margin": round(margin, 4) if margin != float("inf") else None,
                           "msg": f"“{t.text}” revises page {j:03d} (cos {cos:.2f}, {margin:.2f} clear of the next page)"})
        page0 = self.M[j].detach().clone().requires_grad_(True)
        # Write from the teaching sentence AND its restatements -- exactly the material
        # `teaching_payload` hands the context and retrieval baselines, so no model in the
        # comparison learns from something the others never see.
        write_set = self._write_set(t) if self.cfg.write_fact else list(t.support)
        contrast = None
        rivals = list(getattr(t, "distractors", []) or [])
        if self.cfg.write_value_contrast > 0 and rivals:
            # rank the value against its rivals on the support prompts (never the quiz: the
            # question's surface form stays unseen by the write, as the benchmark requires)
            cands = [" " + t.value] + [" " + v for v in rivals]
            contrast = ([(prompt, cands) for prompt, _ in t.support], self.cfg.write_value_contrast)
        with torch.enable_grad():
            page, losses = self._inner_loop(write_set, page0, self.cfg.passes, False,
                                            float(self.importance[j]), allow_skip=True,
                                            contrast=contrast, lr_scale=self.cfg.write_lr_scale,
                                            keep_best=self.cfg.write_keep_best)
        page = page.detach(); delta = float((page - page0.detach()).norm())
        if delta == 0.0:
            # A write that changes nothing is silent by nature. It happens when the read
            # heads are still at their zero init (an un-meta-trained layer cannot route any
            # gradient into memory) or when the support loss was already under the surprise
            # threshold. Say which, loudly, rather than reporting a write that did not happen.
            reason = ("the memory layer is not meta-trained: its read heads are still zero, "
                      "so memory cannot affect the model and no gradient reaches the page"
                      if self.read_heads_are_inert() else
                      "the model already answered the support set, so nothing needed writing")
            events.append({"type": "memory_noop", "page": j,
                           "msg": f"write to page {j:03d} changed nothing: {reason}"})
        self.M[j] = page; self.write_count[j] += 1; self.last_delta[j] = delta; self.usage[j] = 1.0
        gain = max(0.0, losses[0] - losses[-1])
        self.importance[j] = 0.7 * float(self.importance[j]) + 0.3 * gain if revisit else gain
        events.append({"type": "memory_write", "page": j, "delta": delta, "steps": len(losses),
                       "msg": f"memory write page {j:03d}: ‖ΔM‖={delta:.3f}, loss {losses[0]:.3f}→{losses[-1]:.3f} in {len(losses)} steps"})
        self.stream_step += 1
        return {"steps": len(losses), "loss_start": losses[0], "loss_end": losses[-1], "losses": losses,
                "state_change_l2": delta,
                "no_op": delta == 0.0, "inert": self.read_heads_are_inert(),
                "state_elements_updated": page.numel(), "events": events, "page": j}

    # ----------------------------------------------------------------- quiz
    @torch.no_grad()
    def quiz(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """1.0 if every answer token is predicted (teacher-forced), else 0.0."""
        self.eval()
        ids, mask, labels = self.encode_pairs(pairs)
        mem, idx = self._mem_for([p for p, _ in pairs])
        if idx is not None:
            self.read_count.index_add_(0, idx.flatten(), torch.ones(idx.numel(), dtype=torch.long, device=self.device))
        logits = self._forward(ids, mask, mem).logits
        pred = logits[:, :-1].argmax(-1); tgt = labels[:, 1:]
        ignore = (tgt == -100) | (tgt == self.tok.eos_token_id)     # score answer tokens only, not the EOS
        ok = ((pred == tgt) | ignore).all(-1)
        return ok.float().tolist()

    @torch.no_grad()
    def read_trace(self, prompts: Sequence[str]) -> List[Dict]:
        """Which pages each prompt actually addresses, and how strongly.

        This is the retrieval path itself, not a reconstruction of it: the same
        `keys.address` cosines and the same top-k selection `read()` uses to build the
        memory tokens the frozen model attends to. Recorded so a visualisation can draw the
        real edge from a question to the page that answers it.
        """
        if not bool(self.allocated.any()):
            return [{"prompt": p, "pages": [], "cos": [], "weight": []} for p in prompts]
        self.eval()
        alloc = self.allocated.nonzero().flatten()
        qk = self.keys.address(self.keys_for(list(prompts)))
        pk = self.keys.address(self.K[alloc])
        cos = qk @ pk.T
        k = min(self.cfg.read_k, pk.shape[0])
        top, idx = cos.topk(k, dim=-1)
        w = (top / self.cfg.read_temp).softmax(-1)
        out = []
        for i, p in enumerate(prompts):
            pages = [int(alloc[j]) for j in idx[i]]
            out.append({"prompt": p, "pages": pages,
                        "cos": [round(float(c), 4) for c in top[i]],
                        "weight": [round(float(v), 4) for v in w[i]],
                        "owner": [int(self.owner[j]) for j in pages]})
        return out

    def read_heads_are_inert(self) -> bool:
        """True while every read head's output projection is still exactly zero, which is
        its initialisation. In that state memory is disconnected from the computation:
        reads contribute nothing and writes receive no gradient."""
        # .detach() before the scalar cast: these weights carry requires_grad, and torch warns
        # that converting such a tensor to a Python float can behave unexpectedly. This is a
        # read-only check, so drop out of the graph explicitly rather than let the warning
        # print on every real-model run.
        return all(float(h.o.weight.detach().abs().sum()) == 0.0 for h in self.read_heads)

    @torch.no_grad()
    def ask(self, question: str, max_new_tokens: int = 24, stop_on_newline: bool = True) -> str:
        """Answer a question, reading whichever pages the question addresses.

        Greedy decoding with the memory attached, so the answer is produced by the frozen
        model conditioned on what was written into memory, not retrieved as a stored string.
        """
        self.eval()
        prompt = question if question.rstrip().endswith(("?", ":")) else question
        if not prompt.rstrip().endswith(":"):
            prompt = f"Question: {prompt.strip()}\nAnswer:"
        mem, idx = self._mem_for([prompt])
        if idx is not None:
            self.read_count.index_add_(0, idx.flatten(), torch.ones(idx.numel(), dtype=torch.long, device=self.device))
        ids = self.tok(prompt, add_special_tokens=False).input_ids[-self.cfg.max_len:]
        ids = torch.tensor([ids], device=self.device)
        out = []
        for _ in range(max_new_tokens):
            mask = torch.ones_like(ids)
            logits = self._forward(ids, mask, mem).logits[0, -1]
            nxt = int(logits.argmax())
            if nxt == self.tok.eos_token_id:
                break
            piece = self.tok.decode([nxt])
            if stop_on_newline and "\n" in piece:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=self.device)], dim=1)
            if ids.shape[1] > self.cfg.max_len:
                ids = ids[:, -self.cfg.max_len:]
        return self.tok.decode(out).strip()

    # -------------------------------------------------------- meta-training
    def meta_loss(self, episode: List[Teaching], inner_steps: int, second_order: bool) -> torch.Tensor:
        pages, pkeys, total, n = [], [], 0.0, 0
        tkeys = self.keys_for([t.text for t in episode])                       # [n,dk]
        for i, t in enumerate(episode):
            page0 = self.page_init if second_order else self.page_init.detach().requires_grad_(True)
            write_set = self._write_set(t) if self.cfg.meta_write_fact else t.support
            page, _ = self._inner_loop(write_set, page0, inner_steps, create_graph=second_order)
            if not second_order:
                page = self.page_init + (page - page0).detach()
            pages.append(page); pkeys.append(tkeys[i])
            P, PK = torch.stack(pages), torch.stack(pkeys)
            quizzes, owners = [], []
            for s in episode[: i + 1]:
                for q in s.quizzes:
                    quizzes.append(q); owners.append(s.index)
                if s.combined_quiz and s.combined_with is not None and s.combined_with <= i:
                    quizzes.append(s.combined_quiz); owners.append(s.index)
            qk = self.keys_for([p for p, _ in quizzes])
            mem, _ = self.read(qk, PK, P)
            ids, mask, labels = self.encode_pairs(quizzes)
            total = total + self._loss(self._forward(ids, mask, mem).logits, labels); n += 1
            if self.cfg.value_contrast > 0:
                vc = self._value_contrast(episode[: i + 1], quizzes, owners, mem)
                if vc is not None:
                    total = total + self.cfg.value_contrast * vc
            # contrastive: each quiz key should match its own teaching's key
            if self.cfg.contrastive_weight > 0:
                cand = PK
                if self.cfg.neg_bank and self._neg_bank:
                    # Earlier episodes' keys as extra negatives. Targets index the first
                    # len(PK) rows, so appending changes what the query has to beat, not what
                    # it has to match. This is the difference between learning to pick one
                    # page out of four and one page out of a few hundred.
                    cand = torch.cat([PK, torch.cat(self._neg_bank, 0).to(PK.device)], 0)
                sim = self.keys.address(qk) @ self.keys.address(cand).T / 0.1
                target = torch.tensor([[s.index for s in episode[: i + 1]].index(o) for o in owners], device=self.device)
                total = total + self.cfg.contrastive_weight * F.cross_entropy(sim, target)
        if self.cfg.neg_bank:
            self._neg_bank.append(tkeys.detach())
            # keep roughly neg_bank keys; drop whole episodes from the front
            while sum(t.shape[0] for t in self._neg_bank) > self.cfg.neg_bank and len(self._neg_bank) > 1:
                self._neg_bank.pop(0)
        return total / n

    def _value_contrast(self, teachings, quizzes, owners, mem) -> Optional[torch.Tensor]:
        """Cross-entropy over [answer, *distractors] scored by total answer log-probability,
        each read with the SAME memory its quiz was read with. Groups quizzes by candidate
        count so one forward covers a group; returns None when no quiz has distractors."""
        by_index = {t.index: t for t in teachings}
        groups: Dict[int, List[int]] = {}
        for qi, o in enumerate(owners):
            d = by_index[o].distractors
            if d:
                groups.setdefault(len(d) + 1, []).append(qi)
        if not groups:
            return None
        loss, n = 0.0, 0
        for C, qis in groups.items():
            pairs, rows = [], []
            for qi in qis:
                prompt, answer = quizzes[qi]
                cands = [answer] + [" " + v for v in by_index[owners[qi]].distractors]
                pairs += [(prompt, c) for c in cands]
                rows += [qi] * C
            ids, mask, labels = self.encode_pairs(pairs)
            mem_rep = mem[torch.tensor(rows, device=mem.device)]
            scores = self._seq_logprob(self._forward(ids, mask, mem_rep).logits, labels).view(len(qis), C)
            target = torch.zeros(len(qis), dtype=torch.long, device=scores.device)
            loss = loss + F.cross_entropy(scores, target) * len(qis); n += len(qis)
        return loss / n

    # ------------------------------------------------------------ reporting
    def writable_state_bytes(self) -> int:
        return int((self.M.numel() + self.K.numel() + 3 * self.cfg.n_pages) * 4)

    def used_state_bytes(self) -> int:
        return int(int(self.allocated.sum()) * (self.cfg.page_size * self.cfg.slot_dim + self.cfg.key_dim + 3) * 4)

    def page_snapshot(self) -> Dict:
        norms = self.M.flatten(1).norm(dim=-1)
        return {"owner": self.owner.tolist(), "usage": [round(float(v), 4) for v in self.usage],
                "importance": [round(float(v), 4) for v in self.importance], "norm": [round(float(v), 4) for v in norms],
                "last_delta": [round(float(v), 4) for v in self.last_delta], "writes": self.write_count.tolist(), "reads": self.read_count.tolist()}

    def state_for_checkpoint(self) -> Dict:
        return {"params": {n: p.detach().cpu() for n, p in self.named_parameters() if not n.startswith("lm.")},
                # the key mean is fitted, not learned, but a checkpoint that drops it starts
                # every stream with centering disabled for its first few teachings
                "key_stats": {"key_sum": self.keys.key_sum.detach().cpu(), "key_n": self.keys.key_n.detach().cpu()},
                "cfg": asdict(self.cfg)}

    def load_checkpoint(self, state: Dict):
        own = dict(self.named_parameters())
        for n, v in state["params"].items():
            # cue_proj was a bare Sequential before the key head became a module
            n = n.replace("cue_proj.", "cue_proj.proj.", 1) if n.startswith("cue_proj.") and n.split(".")[1].isdigit() else n
            own[n].data.copy_(v.to(own[n].device))
        legacy = [n for n in state["params"] if n.startswith("cue_proj.") and n.split(".")[1].isdigit()]
        if legacy and getattr(self.cue_proj, "kind", "") != "mean":
            raise ValueError("this checkpoint holds a mean-pooled key head; load it with cue_kind='mean' "
                             "or re-fit the key head (hebb/lm/keytrain.py)")
        ks = state.get("key_stats")          # absent in checkpoints written before 2026-09-09
        if ks:
            self.keys.key_sum.copy_(ks["key_sum"].to(self.device)); self.keys.key_n.copy_(ks["key_n"].to(self.device))
