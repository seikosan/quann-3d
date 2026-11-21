from types import SimpleNamespace

import torch

from cloudlm import BPETokenizer
from training import train_wiki
from training.eval_model import build_model_from_checkpoint


def test_resume_restores_optimizer_scheduler_rng_and_epoch(tmp_path, monkeypatch):
    corpus = ["merhaba dünya güzel", "merhaba kitap güzel"] * 4
    monkeypatch.setattr(train_wiki, "load_txt", lambda *args, **kwargs: corpus)
    tokenizer = BPETokenizer(vocab_size=32).train(corpus)
    common = dict(
        corpus="ignored",
        max_texts=None,
        seq_len=2,
        d_model=16,
        d_cloud=12,
        d_dim=4,
        layers=1,
        pool="max",
        cloud_attn=False,
        cloud_attn_heads=8,
        batch=4,
        lr=1e-3,
        grad_clip=0.5,
        point_loss_weight=0.1,
        save_dir=str(tmp_path),
        seed=42,
    )
    torch.manual_seed(151)
    initial_model = train_wiki.CloudLM(
        vocab_size=tokenizer.num_tokens, d_model=16, d_cloud=12,
        d_cloud_dim=4, n_layers=1, pool="max",
    )
    initial_point_head = {
        name: value.detach().clone()
        for name, value in initial_model.point_head.state_dict().items()
    }
    del initial_model
    torch.manual_seed(151)
    args = SimpleNamespace(**common, epochs=1, resume=None)
    train_wiki.step_train(args, tokenizer)

    checkpoint_path = tmp_path / "wiki_cloud_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 1
    assert checkpoint["optimizer_state"]
    assert checkpoint["scheduler_state"]["last_epoch"] == 1
    assert {"python", "numpy", "torch"}.issubset(checkpoint["rng_state"])
    trained_model = build_model_from_checkpoint(checkpoint)
    assert any(
        not torch.equal(initial_point_head[name], value)
        for name, value in trained_model.point_head.state_dict().items()
    )

    args.epochs = 2
    args.resume = str(checkpoint_path)
    train_wiki.step_train(args, tokenizer)
    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert resumed["scheduler_state"]["last_epoch"] == 2


def test_training_batches_keep_singleton_remainder():
    tokens = [[1, 2, 3, 4], [5, 6, 7]]
    batches = list(train_wiki.iter_shuffled_batches(tokens, seq_len=2, batch_size=2))
    examples = [(tuple(inputs), target) for batch in batches for inputs, target in batch]
    assert len(examples) == 3
    assert set(examples) == {(tuple(tokens[0][i:i + 2]), tokens[0][i + 2]) for i in range(2)} | {
        (tuple(tokens[1][0:2]), tokens[1][2])
    }
    assert sorted(len(batch) for batch in batches) == [1, 2]


def test_split_corpus_follows_agents_4_1():
    texts = [str(i) for i in range(100)]
    tr, va, te = train_wiki.split_corpus(texts, seed=42)
    assert (len(tr), len(va), len(te)) == (80, 10, 10)
    assert set(tr) | set(va) | set(te) == set(texts)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    # deterministik: aynı girdi+tohum aynı bölünme
    assert train_wiki.split_corpus(texts, seed=42) == (tr, va, te)
    assert train_wiki.split_corpus(texts, seed=7) != (tr, va, te)
    # küçük kenar durum: çökmemeli, küme korunmalı
    tr2, va2, te2 = train_wiki.split_corpus(texts[:3], seed=42)
    assert sorted(tr2 + va2 + te2) == sorted(texts[:3])


def test_step_test_eval_branch_loads_corpus_and_splits(tmp_path, monkeypatch):
    # Regression: step_test bir dönemi `texts`i yüklemeden referans alıyordu
    # (NameError, 16a4225); C5'ten sonra §4.1 test bölümünü kullanır.
    corpus = [f"cümle {i} güzel dünya" for i in range(10)]
    monkeypatch.setattr(train_wiki, "load_txt", lambda *a, **k: list(corpus))
    tokenizer = BPETokenizer(vocab_size=32).train(corpus)
    model = train_wiki.CloudLM(
        vocab_size=tokenizer.num_tokens, d_model=16, d_cloud=12,
        d_cloud_dim=4, n_layers=1, pool="max",
    )
    calls = []

    def fake_eval(model, tok, texts, all_tokens, args, device, eos_id, split_name="val"):
        calls.append((split_name, list(texts)))
        return 42.0

    monkeypatch.setattr(train_wiki, "evaluate_model", fake_eval)
    args = SimpleNamespace(
        eval_split=20, seq_len=2, save_dir=str(tmp_path), seed=42,
        corpus="ignored", max_texts=None,
    )
    train_wiki.step_test(args, model, tokenizer)
    expected_train, _, expected_test = train_wiki.split_corpus(corpus, 42)
    assert [name for name, _ in calls] == ["test", "train"]
    assert calls[0][1] == expected_test  # %10 test bölümü (eğitimde görülmemiş)
    assert calls[1][1] == expected_train


def test_step_train_uses_val_split_for_best_selection(tmp_path, monkeypatch):
    # C5: eğitim yalnız %80 train üzerinde yapılır; best seçimi val-PPL ile.
    corpus = [f"cümle {i} güzel dünya" for i in range(20)]
    monkeypatch.setattr(train_wiki, "load_txt", lambda *a, **k: list(corpus))
    tokenizer = BPETokenizer(vocab_size=32).train(corpus)
    seen_splits = []

    def fake_eval(model, tok, texts, all_tokens, args, device, eos_id, split_name="val"):
        seen_splits.append((split_name, len(texts)))
        return 12.34

    monkeypatch.setattr(train_wiki, "evaluate_model", fake_eval)
    args = SimpleNamespace(
        corpus="ignored", max_texts=None, seq_len=2, d_model=16, d_cloud=12,
        d_dim=4, layers=1, pool="max", cloud_attn=False, cloud_attn_heads=8,
        batch=4, lr=1e-3, grad_clip=0.5, point_loss_weight=0.1,
        save_dir=str(tmp_path), seed=42, epochs=1, resume=None,
    )
    train_wiki.step_train(args, tokenizer)
    assert seen_splits == [("val", 2)]  # int(20*0.1)=2 cümle val
    checkpoint = torch.load(tmp_path / "wiki_cloud_best.pt", map_location="cpu",
                            weights_only=False)
    assert checkpoint["val_ppl"] == 12.34
    assert checkpoint["ppl"] == 12.34  # seçim metriği = val
    assert checkpoint["train_ppl"] != 12.34
    assert checkpoint["split_seed"] == 42


def test_checkpoint_model_loader_honors_saved_pool_mode():
    source = train_wiki.CloudLM(
        vocab_size=20, d_model=16, d_cloud=12, d_cloud_dim=4,
        n_layers=1, hidden=8, proj_hidden=10, point_hidden=6, pool="attn",
    )
    checkpoint = {
        "vocab_size": 20,
        "config": {
            "d_model": 16, "d_cloud": 12, "d_cloud_dim": 4,
            "n_layers": 1, "hidden": 8, "proj_hidden": 10,
            "point_hidden": 6, "pool": "attn",
        },
        "model_state": source.state_dict(),
    }
    loaded = build_model_from_checkpoint(checkpoint)
    assert loaded.cloud.pool == "attn"
    assert loaded.cloud.point_mlp[0].out_features == 8
    assert loaded.to_nd.proj[0].out_features == 10
    assert loaded.point_head[0].out_features == 6
