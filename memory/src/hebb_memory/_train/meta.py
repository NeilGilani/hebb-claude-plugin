"""Meta-training: teach the read path to use pages, once per base model.

Taken from the research code (hebb/tenants/run.py) with the experiment arms removed. The
module-level knobs below are the recipe; `hebb_memory.train` sets them and restores them.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .._core.memory_lm import LMMemoryConfig, MemoryLM
from .._core.toy import toy_model
from .rules import SLOTS, language_corpus, training_episode

_LM_CACHE: Dict = {}
MODEL: Optional[str] = None      # None = the toy model; otherwise a HF model id
DEVICE: str = "cpu"
META_STEPS: int = 800            # read-path meta-training episodes
KEY_STEPS: int = 0               # key-head pretraining steps before meta-training (keytrain.py); 0 = off
KEY_SLOTS: str = "disjoint"      # "disjoint" (held-out slot vocabulary) | "shared" (the eval's)
FREEZE_KEYS: bool = False        # keep the key head fixed during meta-training
CUE_KIND: str = "mean"           # key head: "mean" (pool every token alike) | "attn" (keyhead.py)
META_EPISODES: str = "tenants"   # what an episode's pages differ by: "tenants" | "keyed" | "mixed"
                                 # (data.training_episode). "keyed" is the pages_keyed eval's shape.
META_SLOTS: str = "basic"        # meta-training rule names: the 8 hand-written ("basic") or those
                                 # plus composed held-out names ("composed"), so the read path has
                                 # seen "premium refund window"-shaped rules before eval asks
META_BATCH: int = 1              # episodes per optimizer step. 1 is what every result before
                                 # 2026-09-16 used; the loss it prints swings 0.01 -> 4.4 between
                                 # consecutive steps and seed 0 beat seed 1 by 0.17-0.25 at every
                                 # width on the real model. Accumulating gradients over several
                                 # episodes is the cheapest thing that could change both.
META_SCHEDULE: str = "const"     # learning-rate schedule: "const" | "cosine" (5% warmup, decay to 0)
PASSES: int = 20                 # inner-loop steps per deployment write (experience)
INNER_LR: float = 1.0            # inner-loop step (RMS-normalised, per element) in meta-training AND
                                 # deployment; log_lr is initialised to it and, in first-order
                                 # meta-training, never moves (its gradient is detached)
SURPRISE_SKIP: float = 0.05      # a write stops early under this support loss; 0 = never
WRITE_VALUE_CONTRAST: float = 0.0  # rank the value against its rivals inside the write
WRITE_LR_SCALE: float = 1.0      # multiplies the learned inner-loop step in deployment writes
READ_K: int = 2                  # pages blended per read at deployment; top1 in the diagnostic = 1
WRITE_KEEP_BEST: bool = False    # a deployment write keeps its lowest-loss page
WRITE_FACT: bool = True          # deployment writes from Fact + support (False: support only)
META_INNER: Tuple[int, int] = (4, 4)   # inner-loop steps a meta-training write uses, as a range
                                 # sampled per episode. Eval writes run cfg.passes (20) steps, so
                                 # a reader trained only on 4-step pages has never seen a page of
                                 # the magnitude it reads at eval. (4, 20) covers both.
META_EPISODE_LEN: int = 4        # teachings per episode; the read head practises choosing among
                                 # this many pages, and pages_keyed is evaluated among 3-128
META_VALUE_CONTRAST: float = 0.0 # weight of the candidate-ranking meta-loss term (LMMemoryConfig.value_contrast)
META_WRITE_FACT: bool = False    # meta-training writes from fact+support like `experience` does
META_CKPT: Optional[Path] = None # directory for meta-training checkpoints. Set, and a run that is
                                 # killed mid-meta-training resumes from its last saved episode, and a
                                 # finished meta-training is loaded instead of redone. Exists so the
                                 # 0.5B crossover can run on CPU runners with a 6-hour job limit.
META_CKPT_EVERY: int = 25        # episodes between checkpoints


def _tenant_tokenizer(corpus: List[str], vocab_size: int = 700):
    """BPE trained on THIS benchmark's language.

    The stock toy tokenizer is trained on a different benchmark's corpus, so tenant names and
    slot values tokenised badly and every arm sat at chance for a reason unrelated to memory.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[EOS]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", eos_token="[EOS]",
                                   unk_token="[UNK]")


def _pretrain(lm, tok, corpus: List[str], steps: int, seed: int, lr: float = 3e-3,
              batch: int = 24, max_len: int = 64):
    """Teach the language, not the answers. The corpus binds only TRAINING tenant names."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(lm.parameters(), lr=lr)
    lm.train()
    for _ in range(steps):
        picks = [corpus[int(i)] for i in rng.integers(0, len(corpus), size=batch)]
        enc = tok(picks, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = lm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
        opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
    lm.eval()
    return float(out.loss.detach())


def _real_lm(name: str, device: str):
    """A pretrained model. The toy path exists only because this environment has no GPU and no
    model hub; a real model needs neither the hand-built tokenizer nor the language pretraining,
    because it already knows English."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    lm.to(device).eval()
    return lm, tok


def _fresh_lm(seed: int, pretrain_steps: int):
    """A model that knows the language and none of the answers.

    Cached per (seed, steps) and deep-copied per arm, so every arm starts from BYTE-IDENTICAL
    weights -- otherwise arms would differ by their initialisation as much as by their method.
    """
    if MODEL:
        # A real model is NEVER cached. Caching one per seed kept every previous model alive on
        # the GPU -- observed climbing 6.2 -> 9.3 -> 12.4 GB across three seeds on a 12.9 GB
        # card, heading for an OOM that would have looked like the benchmark's fault. Reloading
        # from the local HF cache costs about two seconds, which is nothing next to a run, and
        # it also removes the deepcopy, halving peak memory.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _real_lm(MODEL, DEVICE)

    key = (seed, pretrain_steps, MODEL, DEVICE)
    if key not in _LM_CACHE:
        torch.manual_seed(seed)
        corpus = language_corpus(900, seed)
        tok = _tenant_tokenizer(corpus)
        lm = toy_model(tok)
        loss = _pretrain(lm, tok, corpus, pretrain_steps, seed) if pretrain_steps else None
        _LM_CACHE[key] = (lm, tok, loss)
    lm, tok, _ = _LM_CACHE[key]
    return copy.deepcopy(lm), tok


_META_CACHE: Dict = {}


def _meta_train(m: MemoryLM, steps: int, seed: int, lr: float = 3e-4,
                episode_len: int = 4, inner_steps: Tuple[int, int] = (4, 4), freeze_keys: bool = False,
                slot_pool=None, episode_mode: str = "tenants", batch: int = 1,
                schedule: str = "const", ckpt: Optional[Path] = None, ckpt_every: int = 25) -> float:
    """Teach the READ path to use memory at all.

    Without this the read heads' output projections are zero and stay zero -- the model simply
    never consults the pages, and the arm silently degenerates into the frozen base model.
    That is exactly what happened on the first run of this benchmark
    (`read_heads_are_inert()` returned True and pages scored at chance).

    Episodes are drawn from TRAINING tenants only, so nothing evaluated later is seen here.
    This is a genuine one-time setup cost of the pages approach and is reported as such.
    """
    rng = np.random.default_rng(seed + 31337)
    params = [p for p in m.trainable_parameters()]
    if freeze_keys:
        # The key head was trained by keytrain.py against hundreds of candidates; the
        # four-way contrastive term here would only pull it back toward what it can get
        # away with in an episode.
        key_ids = {id(p) for p in m.cue_proj.parameters()}
        params = [p for p in params if id(p) not in key_ids]
    opt = torch.optim.Adam(params, lr=lr)
    batch = max(1, int(batch))
    n_updates = max(1, steps // batch)
    if schedule == "cosine":
        warm = max(1, n_updates // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda u: (u + 1) / warm if u < warm
            else 0.5 * (1.0 + math.cos(math.pi * (u - warm) / max(1, n_updates - warm))))
    elif schedule == "const":
        sched = None
    else:
        raise ValueError(f"unknown meta schedule {schedule!r}")
    lo, hi = inner_steps
    last = float("nan")
    i0 = 0
    if ckpt is not None and Path(ckpt).exists():
        # Resume: model (incl. the key head and its centering stats), optimizer, schedule,
        # the episode sampler's RNG and the episode index. Everything the next episode
        # depends on, so a resumed run is the uninterrupted run.
        st = torch.load(ckpt, map_location=m.device, weights_only=False)
        m.load_checkpoint(st["model"])
        opt.load_state_dict(st["opt"])
        if sched is not None and st.get("sched") is not None:
            sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        m._neg_bank = [t.to(m.device) for t in st.get("neg_bank", [])]
        i0, last = int(st["episode"]), float(st.get("last", float("nan")))
        print(f"    meta-train: resuming at episode {i0}/{steps} from {ckpt}", flush=True)

    def save(i_done: int):
        if ckpt is None:
            return
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(ckpt) + ".tmp")
        torch.save({"model": m.state_for_checkpoint(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "rng": rng.bit_generator.state, "episode": i_done, "last": last,
                    "neg_bank": [t.detach().cpu() for t in m._neg_bank]}, tmp)
        os.replace(tmp, ckpt)

    t0 = time.time()
    # Progress, because this phase prints nothing and is the longest thing in the benchmark.
    # A 2400-episode run on a 0.5B model silently occupied a GPU for twelve hours and was
    # indistinguishable from a hang; the only way anyone could tell was nvidia-smi.
    every = max(1, steps // 20)
    # `steps` counts EPISODES whatever the batch, so a run's cost is the same at any batch size
    # and the only thing that changes is how many episodes each update averages over.
    opt.zero_grad(set_to_none=True)
    for i in range(i0, steps):
        k = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        loss = m.meta_loss(training_episode(episode_len, rng, slot_pool, episode_mode),
                           k, second_order=False)
        (loss / batch).backward()
        last = float(loss.detach())
        stepped = False
        if (i + 1) % batch == 0 or i == steps - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad(set_to_none=True)
            stepped = True
        # Only ever checkpoint on an update boundary, so no half-accumulated gradient is lost.
        if stepped and ckpt_every > 0 and (i + 1) % ckpt_every == 0 and i + 1 < steps:
            save(i + 1)
        if (i + 1) % every == 0 or i == 0:
            done = i + 1
            rate = (done - i0) / max(1e-9, time.time() - t0)
            eta = (steps - done) / max(1e-9, rate)
            print(f"    meta-train {done}/{steps}  loss {last:.4f}  "
                  f"{rate:.1f} ep/s  eta {eta / 60:.1f} min", flush=True)
    if ckpt is not None and steps > i0:
        save(steps)
    return last


def _meta_trained(seed: int, pretrain_steps: int, cfg: LMMemoryConfig,
                  meta_steps: int = 250) -> MemoryLM:
    """A freshly-built MemoryLM carrying meta-trained weights.

    NOT a deepcopy of one. MemoryLM injects memory through forward hooks registered on the
    base model's layers, and the hook closure captures `self`; a deepcopy therefore ends up
    with hooks that still drive the ORIGINAL object, so memory never reaches the copy's
    forward pass. It fails loudly here ("element 0 of tensors does not require grad"), but on
    a read-only path it would fail silently and the arm would quietly become the frozen base
    model. Rebuild-and-load-checkpoint is the only safe way to clone one.
    """
    # Every field that changes a PARAMETER SHAPE must be in the key. Omitting page_size handed
    # back a checkpoint built for 8-slot pages when 16 was requested, and load_checkpoint died
    # on the shape mismatch -- loudly here, but a subtler omission would silently return a
    # model meta-trained for a different configuration.
    # n_pages is deliberately NOT in the key. No trained parameter depends on it (page_init is
    # [page_size, slot_dim]; the pages themselves are buffers a checkpoint never carries), and
    # meta-training touches at most episode_len pages. Keying on it made a 6-width crossover
    # meta-train six times per seed for six byte-identical results.
    key = (seed, pretrain_steps, cfg.page_size, cfg.slot_dim, cfg.key_dim, cfg.heads,
           cfg.read_every, cfg.cue_kind, meta_steps, KEY_STEPS, KEY_SLOTS, FREEZE_KEYS, META_SLOTS,
           META_EPISODES, META_BATCH, META_SCHEDULE, tuple(META_INNER), META_EPISODE_LEN,
           cfg.value_contrast, cfg.meta_write_fact)
    # inner_lr joins the key only when it differs from the historical 1.0, so every checkpoint
    # written before it was a knob keeps its tag (and the Actions cache keeps finding it).
    if cfg.inner_lr != 1.0:
        key = key + (("inner_lr", cfg.inner_lr),)
    tag = hashlib.sha1(repr(key).encode()).hexdigest()[:12]
    final = Path(META_CKPT) / f"meta-{tag}.final.pt" if META_CKPT is not None else None
    partial = Path(META_CKPT) / f"meta-{tag}.partial.pt" if META_CKPT is not None else None
    if key not in _META_CACHE and final is not None and final.exists():
        st = torch.load(final, map_location="cpu", weights_only=False)
        _META_CACHE[key] = (st["state"], float(st["loss"]))
        print(f"    meta-train: loaded finished checkpoint {final}", flush=True)
    if key not in _META_CACHE:
        lm, tok = _fresh_lm(seed, pretrain_steps)
        # .to(DEVICE) is NOT optional. MemoryLM creates its own parameters and buffers -- the
        # pages, the keys, the read heads -- at construction, on CPU. Wrapping a CUDA model
        # does not move them, so the memory would sit on a different device from the model it
        # is supposed to be reading. hebb/lm/train.py has always done this; this file did not.
        m = MemoryLM(lm, tok, cfg).to(DEVICE)
        resuming = partial is not None and partial.exists()
        if KEY_STEPS > 0 and not resuming:
            # A partial checkpoint already holds the trained key head; training it again
            # would be wasted, and its result is overwritten by the load anyway.
            from .keyhead_train import train_tenant_key_head
            train_tenant_key_head(m, seed, steps=KEY_STEPS, slots=KEY_SLOTS)
        pool = None
        if META_SLOTS == "composed":
            from .keyhead_train import train_slot_vocab
            pool = list(SLOTS) + train_slot_vocab(48)
        loss = _meta_train(m, meta_steps, seed, freeze_keys=FREEZE_KEYS and KEY_STEPS > 0,
                           slot_pool=pool, episode_mode=META_EPISODES, batch=META_BATCH,
                           schedule=META_SCHEDULE, inner_steps=tuple(META_INNER),
                           episode_len=META_EPISODE_LEN, ckpt=partial, ckpt_every=META_CKPT_EVERY)
        if m.read_heads_are_inert():
            raise RuntimeError("read heads still inert after meta-training: the pages arm "
                               "would silently be the frozen base model")
        _META_CACHE[key] = (m.state_for_checkpoint(), loss)
        if final is not None:
            final.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": _META_CACHE[key][0], "loss": loss, "key": repr(key)}, final)
            if partial is not None and partial.exists():
                partial.unlink()
    state, _ = _META_CACHE[key]
    lm, tok = _fresh_lm(seed, pretrain_steps)
    m = MemoryLM(lm, tok, cfg).to(DEVICE)
    m.load_checkpoint(state)
    if m.read_heads_are_inert():
        raise RuntimeError("read heads inert after loading the meta-trained checkpoint")
    return m
