"""The library's promises, checked on the tiny CPU model: one page per memory, exact deletion,
exact attribution, scoped reads, saving, and refusing to run without a trained memory."""
import pytest
import torch

import hebb_memory as hebb
from hebb_memory.memory import question_for
from hebb_memory.training import toy_model

OPTIONS = ["14 days", "30 days", "60 days", "90 days"]


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    torch.set_num_threads(2)
    reg = hebb.Registry(tmp_path_factory.mktemp("registry"))
    hebb.train("toy", registry=reg, pretrain_steps=20, meta_steps=4, key_steps=3)
    return reg


def fresh(registry, **kw):
    lm, tok = toy_model(pretrain_steps=20)
    return hebb.attach(lm, tok, model_id="toy", registry=registry, **kw)


def scores(mem, subject, attribute="refund window"):
    return mem.choose(subject, attribute, OPTIONS).scores


def test_attach_refuses_without_a_trained_memory_and_says_how_to_get_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HEBB_MEMORY_OFFLINE", "1")
    lm, tok = toy_model(pretrain_steps=20)
    with pytest.raises(hebb.NoCheckpoint) as e:
        hebb.attach(lm, tok, model_id="toy", registry=hebb.Registry(tmp_path))
    assert "hebb-memory train --model toy" in str(e.value)


def test_each_memory_gets_its_own_page_and_the_write_changes_it(registry):
    mem = fresh(registry)
    a = mem.remember("acme", "refund window", "60 days")
    b = mem.remember("acme", "support channel", "phone")
    c = mem.remember("globex", "refund window", "14 days")
    assert len({a.page, b.page, c.page}) == 3
    assert all(r.write["change"] > 0 for r in (a, b, c))
    assert [r.id for r in mem.records()] == [a.id, b.id, c.id]


def test_forgetting_leaves_exactly_the_model_that_never_saw_it(registry):
    # write A, B, C and forget B; then write only A and C into a fresh memory
    m1 = fresh(registry)
    m1.remember("acme", "refund window", "60 days")
    b = m1.remember("initech", "refund window", "30 days")
    m1.remember("globex", "refund window", "14 days")
    m1.forget(b)
    m2 = fresh(registry)
    m2.remember("acme", "refund window", "60 days")
    m2.remember("globex", "refund window", "14 days")
    for s in ("acme", "globex", "initech"):
        assert scores(m1, s) == scores(m2, s)          # bit for bit, not approximately
    for subject in ("acme", "globex"):
        p1, = (r.page for r in m1.records(subject))
        p2, = (r.page for r in m2.records(subject))
        assert torch.equal(m1.model.M[p1], m2.model.M[p2])
        assert torch.equal(m1.model.K[p1], m2.model.K[p2])
    assert float(m1.model.M[b.page].abs().sum()) == 0.0 and not bool(m1.model.allocated[b.page])


def test_forgetting_a_subject_removes_all_of_its_pages_and_nothing_else(registry):
    mem = fresh(registry)
    mem.remember("acme", "refund window", "60 days")
    mem.remember("acme", "support channel", "phone")
    keep = mem.remember("globex", "refund window", "14 days")
    before = mem.model.M[keep.page].clone()
    gone = mem.forget("acme")
    assert {r.subject for r in gone} == {"acme"} and len(gone) == 2
    assert mem.subjects() == ["globex"]
    assert torch.equal(mem.model.M[keep.page], before)
    assert mem.trace("acme", "refund window") == []


def test_reads_are_scoped_to_the_subject(registry):
    mem = fresh(registry)
    mem.remember("acme", "refund window", "60 days")
    mem.remember("globex", "refund window", "14 days")
    mem.remember("globex", "support channel", "email")
    for h in mem.trace("acme", "refund window"):
        assert h.record.subject == "acme"
    _, idx = mem.model._mem_for([question_for("acme", "refund window")],
                                restrict=mem._restrict("acme"))
    assert {int(i) for i in idx.flatten()} == {r.page for r in mem.records("acme")}


def test_a_subject_with_nothing_learned_reads_nothing(registry):
    bare = scores(fresh(registry), "acme")                # no memories at all: the model alone
    mem = fresh(registry)
    mem.remember("globex", "refund window", "14 days")
    assert scores(mem, "acme") == bare                    # never globex's memory
    mem.remember("acme", "refund window", "60 days")
    mem.forget("acme")
    assert scores(mem, "acme") == bare                    # nor after acme's is forgotten
    assert mem.generate(question_for("acme", "refund window"), subject="acme") == \
        fresh(registry).generate(question_for("acme", "refund window"), subject="acme")


def test_trace_is_the_read_path(registry):
    mem = fresh(registry)
    for slot, value in (("refund window", "60 days"), ("support channel", "phone"),
                        ("data region", "eu")):
        mem.remember("acme", slot, value)
    hits = mem.trace("acme", "refund window")
    _, idx = mem.model._mem_for([question_for("acme", "refund window")],
                                restrict=mem._restrict("acme"))
    assert [h.record.page for h in hits] == [int(i) for i in idx.flatten()]
    assert abs(sum(h.weight for h in hits) - 1.0) < 1e-3
    assert hits == sorted(hits, key=lambda h: -h.cos)


def test_a_new_value_replaces_the_old_page(registry):
    mem = fresh(registry)
    old = mem.remember("acme", "refund window", "60 days")
    new = mem.remember("acme", "refund window", "90 days")
    assert [r.value for r in mem.records("acme")] == ["90 days"]
    assert new.id != old.id


def test_save_and_load_bring_back_the_same_answers(registry, tmp_path):
    m1 = fresh(registry)
    m1.remember("acme", "refund window", "60 days")
    m1.remember("globex", "refund window", "14 days")
    path = m1.save(tmp_path / "pages.pt", subject="acme")
    m2 = fresh(registry)
    loaded = m2.load(path)
    assert [(r.subject, r.value) for r in loaded] == [("acme", "60 days")]
    m3 = fresh(registry)
    m3.remember("acme", "refund window", "60 days")
    assert scores(m2, "acme") == scores(m3, "acme")


def test_a_full_memory_refuses_instead_of_overwriting(registry):
    mem = fresh(registry, n_pages=2)
    mem.remember("acme", "refund window", "60 days")
    mem.remember("acme", "support channel", "phone")
    with pytest.raises(hebb.MemoryFull):
        mem.remember("acme", "data region", "eu")
    mem.remember("acme", "refund window", "30 days")      # a replacement still fits


def test_a_memory_trained_for_another_model_is_refused(registry):
    lm, tok = toy_model(pretrain_steps=20)
    path = registry.entries()[0]["file"]
    with pytest.raises(hebb.CheckpointMismatch):
        hebb.attach(lm, tok, model_id="some/other-model", checkpoint=registry.root / path)


def test_selfcheck_runs_the_mechanics_and_leaves_the_memory_empty(registry):
    mem = fresh(registry)
    rep = hebb.selfcheck(mem)
    assert rep["read_heads_active"] and rep["writes_changed_pages"] and rep["forget_ok"]
    assert len(mem) == 0 and not bool(mem.model.allocated.any())


def publish(registry, folder, sha=None):
    """Lay out a release folder the way the pretrain workflow does: the file and manifest.json."""
    import hashlib, json, shutil
    from hebb_memory.registry import checkpoint_name, fingerprint, product_config
    fp = fingerprint(product_config())
    name = checkpoint_name("toy", fp)
    folder.mkdir()
    shutil.copy(registry.root / registry.entries()[0]["file"], folder / name)
    data = (folder / name).read_bytes()
    (folder / "manifest.json").write_text(json.dumps({"files": {name: {
        "model": "toy", "fingerprint": fp, "bytes": len(data), "loss": 1.0, "recipe": {},
        "created": "2026-10-02T00:00:00Z", "sha256": sha or hashlib.sha256(data).hexdigest()}}}))
    return folder.as_uri()


def test_attach_downloads_the_published_memory_once(registry, tmp_path, monkeypatch):
    import hebb_memory.registry as reg
    monkeypatch.setattr(reg, "PRETRAINED", publish(registry, tmp_path / "release"))
    local = hebb.Registry(tmp_path / "cache")
    lm, tok = toy_model(pretrain_steps=20)
    mem = hebb.attach(lm, tok, model_id="toy", registry=local)
    assert len(local.entries()) == 1 and not mem.model.read_heads_are_inert()
    monkeypatch.setattr(reg, "PRETRAINED", (tmp_path / "gone").as_uri())   # no network needed now
    hebb.attach(lm, tok, model_id="toy", registry=local)


def test_a_download_that_does_not_match_its_checksum_is_discarded(registry, tmp_path, monkeypatch):
    import hebb_memory.registry as reg
    monkeypatch.setattr(reg, "PRETRAINED", publish(registry, tmp_path / "release", sha="0" * 64))
    local = hebb.Registry(tmp_path / "cache")
    lm, tok = toy_model(pretrain_steps=20)
    with pytest.raises(hebb.CheckpointMismatch):
        hebb.attach(lm, tok, model_id="toy", registry=local)
    assert local.entries() == [] and not any(local.root.glob("*.pt"))


def test_offline_never_downloads(registry, tmp_path, monkeypatch):
    import hebb_memory.registry as reg
    monkeypatch.setattr(reg, "PRETRAINED", publish(registry, tmp_path / "release"))
    monkeypatch.setenv("HEBB_MEMORY_OFFLINE", "1")
    lm, tok = toy_model(pretrain_steps=20)
    with pytest.raises(hebb.NoCheckpoint):
        hebb.attach(lm, tok, model_id="toy", registry=hebb.Registry(tmp_path / "cache"))
