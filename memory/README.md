# hebb-memory

**Memory written into a frozen model, one page per memory.** Teach an open model new facts
after it ships, without touching its weights. Delete any one of those facts exactly. See which
fact an answer came from. Keep each customer's facts where only that customer's questions can
read them.

```python
import hebb_memory as hebb

mem = hebb.attach("Qwen/Qwen2.5-0.5B")

r = mem.remember("acme", "refund window", "60 days")    # written into its own page
mem.ask("acme", "refund window")                         # the model answers, reading that page
mem.trace("acme", "refund window")                       # which pages the answer read, and how much
mem.forget(r)                                            # gone, exactly
mem.forget("acme")                                       # or everything about acme
```

## Why

A model is frozen the day it ships. Teaching it something new usually means fine-tuning, and a
fine-tune is the wrong shape for facts that belong to one customer and may have to be deleted:

- **It is big.** A LoRA adapter for one customer on Qwen2.5-0.5B is 8,798,208 bytes. Three of
  that customer's facts as pages are 6,948 bytes, **1,266x smaller**. A page does not grow with
  the model; an adapter does.
- **It can't forget one thing.** Facts trained into shared weights are smeared across all of
  them. Here each fact lives in one page, so deleting it means zeroing that page and nothing else.
- **It can't say where an answer came from.** Here every read goes through pages, so the pages
  it read are the answer's sources.

## Install

```bash
pip install "hebb-memory @ git+https://github.com/NeilGilani/hebb-claude-plugin#subdirectory=memory"
```

(`pip install hebb-memory` once it is on PyPI.)

Python 3.10+, PyTorch and Hugging Face `transformers`. It attaches to causal LMs whose decoder
layers sit at `model.layers` (Qwen, Llama, Mistral, Gemma), `transformer.h` (GPT-2),
`model.decoder.layers` (OPT) or `gpt_neox.layers` (Pythia). The measured results are on
Qwen2.5-0.5B only.

## Try it in a few minutes, on a CPU

```bash
hebb-memory train --model toy          # a tiny model, about 3 minutes on a laptop CPU
hebb-memory demo --model toy           # remember, ask, trace and forget a few facts
```

What it printed on our machine:

```
before anything is written:
  acme refund window -> 30 days
remembered  acme    refund window    = 60 days  page 0
remembered  acme    support channel  = phone    page 1
remembered  globex  refund window    = 14 days  page 2
remembered  globex  support channel  = email    page 3
ask         acme    refund window    -> 60 days  right   read: page 0 (acme refund window) 0.89, page 1 (acme support channel) 0.11
ask         acme    support channel  -> phone    right   read: page 1 (acme support channel) 0.86, page 0 (acme refund window) 0.14
ask         globex  refund window    -> 30 days  WRONG   read: page 2 (globex refund window) 0.78, page 3 (globex support channel) 0.22
ask         globex  support channel  -> email    right   read: page 3 (globex support channel) 0.74, page 2 (globex refund window) 0.26
forgot acme: pages [0, 1] zeroed and freed
  acme refund window -> 30 days (no acme pages left; this is the model alone)
  globex refund window -> 30 days
```

Every question read its own page first, and forgetting acme put its answer back to what the
bare model says. One answer is wrong: the toy is a 128-wide, 4-layer model trained for three
minutes, there to show the API working, not to judge accuracy.

## Use it on a real model

A memory is trained **once per base model**: a key head that turns a sentence into an address,
and read heads that let the frozen model attend to pages. The base model's weights never change.

```bash
hebb-memory train --model Qwen/Qwen2.5-0.5B      # about an hour on an RTX 4070
hebb-memory check --model Qwen/Qwen2.5-0.5B      # attach it and run the self-check
```

Add `--ckpt-dir ckpt` to `train` and an interrupted run picks up where it stopped. The trained
memory is a few megabytes in `~/.cache/hebb-memory`. Share it on the Hugging Face Hub and
anyone can skip training:

```bash
huggingface-cli upload <you>/hebb-qwen2.5-0.5b ~/.cache/hebb-memory/Qwen-Qwen2.5-0.5B-<id>.pt memory.pt
```

```python
mem = hebb.attach("Qwen/Qwen2.5-0.5B", checkpoint="hf:<you>/hebb-qwen2.5-0.5b")
```

## The API

| | |
|---|---|
| `mem.remember(subject, attribute, value)` | Write one fact into a fresh page. A new value for the same subject and attribute replaces the old page. Returns a `Record`. |
| `mem.ask(subject, attribute)` | The model's answer, reading only that subject's pages. |
| `mem.choose(subject, attribute, options)` | Which option the model finds most likely, with scores. This is how the accuracy below is measured. |
| `mem.trace(subject, attribute)` | The pages that question reads, with their weights: the answer's sources. |
| `mem.forget(record \| id \| subject)` | Zero and free a page, or all of a subject's pages. |
| `mem.records(subject=None)`, `mem.subjects()` | What is stored. |
| `mem.save(path, subject=None)`, `mem.load(path)` | Move memories between processes or machines. A file holds pages, not a model: about 2 KB per fact. |
| `hebb.selfcheck(mem)` | Are the heads trained, do writes change pages, do questions route to their page, does forgetting empty it? |

A subject is whoever the facts belong to: a customer, a user, a project. Reads are limited to
the subject's own pages before anything is compared, so one subject's question cannot read
another's page at all.

## What is exact, and what is not

**Exact, and tested bit for bit** (`tests/test_memory.py`):

- **Deletion.** Write A, B and C, forget B, and the model's scores on every question are
  identical, to the last bit, to a model that was only ever given A and C.
- **Attribution.** `trace` is the read path itself (the same addresses, top-k and weights that
  produce the answer), not an explanation made up afterwards.
- **Isolation.** A read for one subject only ever has that subject's pages as candidates.

**Not exact:**

- **Answers can still be wrong.** On the measured model, accuracy is well above chance and
  above the alternatives below, but not perfect.
- **Putting the facts in the prompt is still more accurate at small scale.** In the same
  experiment, pasting every fact into the prompt scored 1.000. Pages win on size, deletion,
  attribution and isolation, not yet on accuracy against prompting; where prompting stops
  working as facts pile up has not been measured on a real model.
- **Facts are subject, attribute, value.** That is the shape the memory was trained and measured
  on. Free-form notes are not supported in this version.

## Results

Qwen2.5-0.5B. Each customer has three facts that conflict with other customers' facts. Two
seeds, 800 meta-training episodes, forced choice among the possible values. Leakage (the share
of answers that were another customer's value) is in parentheses.

| customers | chance | facts in the prompt | LoRA per customer | **pages** |
|---|---|---|---|---|
| 8 | 0.382 | 1.000 | 0.646 (0.354) | **0.854** (0.146) |
| 32 | 0.214 | 1.000 | 0.490 (0.510) | **0.677** (0.323) |

| state per customer | LoRA | pages | ratio |
|---|---|---|---|
| Qwen2.5-0.5B | 8,798,208 B | 6,948 B | **1,266x** |

These were measured with the research code this library is taken from. One difference: the
research code kept updating the mean of the key addresses as it ran, and this library freezes
it at the trained value, because otherwise every write and every question nudges shared state
and deletion stops being exact. Run `hebb-memory check` to measure your own setup.

## How it works

The base model is frozen. Every other decoder layer gets a small read head that attends over
memory tokens. A **page** is a few slots of those tokens, and each fact gets its own.

- **Writing** a fact runs a few gradient steps on that page alone, so the frozen model, reading
  the page, says the fact. Nothing outside the page changes.
- **Reading** turns the question into an address with a trained key head, picks the closest
  pages among the subject's own, and lets the read heads attend to them.
- **Training** (once per base model) meta-learns the read heads, the page initialisation and the
  write step on facts about made-up companies, so that writing works on facts never seen in
  training.

The paper: *One Page per Memory: Allocation Makes Parametric Memory Persistent, Deletable and
Attributable, Given the Address* (preprint in preparation).

## License

Apache-2.0. Built by [Hebb](https://hebb-site.pages.dev).
