import numpy as np


def test_saved_policy_reload_preserves_bounded_actions(tmp_path):
    from ring_learning.dataset import Normalizer
    from ring_learning.policy import Policy, save_checkpoint, load_checkpoint

    model = Policy(obs_size=30, hidden_size=64)
    normalizer = Normalizer.fit(np.zeros((2, 30), dtype=np.float32))
    path = tmp_path / "policy.pt"
    obs = np.linspace(-2, 2, 30).astype(np.float32)
    before = model.act(obs, normalizer)
    assert before.shape == (4,)
    assert np.max(np.abs(before)) <= 1
    save_checkpoint(path, model, normalizer, {"seed": 17})
    restored, norm, meta = load_checkpoint(path)
    np.testing.assert_allclose(restored.act(obs, norm), before, atol=1e-6)
    assert meta["seed"] == 17
