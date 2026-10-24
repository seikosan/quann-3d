import random

import numpy as np
import torch

from quann_cloud.src.qcloud_network import QCloudNetwork
from quann_cloud.src.qcloud_trainer import QCloudTrainer


class _Tokenizer:
    def encode(self, text):
        return [2, 3, 4, 5]


def _trainer(corpus, eval_corpus=None, hybrid=False):
    model = QCloudNetwork(
        vocab_size=16, d_model=16, d_cloud=12, d_cloud_dim=3, n_layers=1
    )
    return QCloudTrainer(
        model=model,
        corpus=corpus,
        eval_corpus=eval_corpus,
        device=torch.device("cpu"),
        tokenizer=_Tokenizer(),
        seq_len=2,
        hybrid=hybrid,
        max_epochs=4,
    )


def test_automatic_eval_split_is_disjoint_from_training():
    corpus = [f"sentence-{i}" for i in range(10)]
    trainer = _trainer(corpus)
    assert trainer.corpus == corpus[:-1]
    assert trainer.eval_corpus == corpus[-1:]
    assert not set(trainer.corpus).intersection(trainer.eval_corpus)


def test_quann_checkpoint_round_trip_restores_search_and_rng_state(tmp_path):
    trainer = _trainer(["train"], ["eval"], hybrid=True)
    weight = trainer.model.get_weight(8)
    weight.low[:] = [-0.5, -0.4, -0.3]
    weight.high[:] = [0.5, 0.4, 0.3]
    weight.value[:] = [0.1, -0.1, 0.2]
    weight.direction[:] = [-1, 1, -1]
    weight.sigma = 0.12
    random.seed(101)
    np.random.seed(202)
    trainer._save(str(tmp_path / "qcloud.pt"), epoch=3, ppl=2.5)
    expected_random = random.random()
    expected_numpy = np.random.random()

    restored = _trainer(["train"], ["eval"], hybrid=True)
    restored.load_checkpoint(str(tmp_path / "qcloud.pt"))
    loaded = restored.model.get_weight(8)
    np.testing.assert_allclose(loaded.low, weight.low)
    np.testing.assert_allclose(loaded.high, weight.high)
    np.testing.assert_allclose(loaded.value, weight.value)
    np.testing.assert_array_equal(loaded.direction, weight.direction)
    assert loaded.sigma == weight.sigma
    assert restored.start_epoch == 3
    assert restored.best_ppl == 2.5
    assert restored.optimizer is not None and restored.scheduler is not None
    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
