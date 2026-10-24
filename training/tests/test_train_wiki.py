from types import SimpleNamespace

import torch

from cloudlm import BPETokenizer
from training import train_wiki
from training.test_model import build_model_from_checkpoint


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
        batch=4,
        lr=1e-3,
        save_dir=str(tmp_path),
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
