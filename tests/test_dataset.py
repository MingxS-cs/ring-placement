import numpy as np


def test_source_split_keeps_all_descendants_together_and_holds_out_scene():
    from ring_learning.dataset import split_episodes, augment_episode

    episodes = [{"episode_id": f"E{i:02}", "scene_group": (i - 1) // 8 + 1,
                 "ring_xy_m": [.49, .08], "bearing_xy_m": [.45, 0.0],
                 "final_offset_xy_m": [.002, -.001]} for i in range(1, 25)]
    splits = split_episodes(episodes, seed=17)
    assigned = [e["episode_id"] for group in splits.values() for e in group]
    assert len(assigned) == len(set(assigned)) == 24
    assert all(e["scene_group"] == 3 for e in splits["ood_test"])
    rng = np.random.default_rng(7)
    for split, group in splits.items():
        for ep in group:
            child = augment_episode(ep, rng, split=split)
            assert child["source_episode_id"] == ep["episode_id"]
            assert child["split"] == split
            assert child["geometry_scale"] == 1.0


def test_rotation_preserves_object_goal_geometry_and_rotates_actions():
    from ring_learning.dataset import rotate_xy

    rotated = rotate_xy(np.array([.02, .03]), np.pi / 2)
    np.testing.assert_allclose(rotated, [-.03, .02], atol=1e-10)
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm([.02, .03]))
    action = np.array([.004, -.002])
    np.testing.assert_allclose(rotate_xy(action, np.pi / 2), [.002, .004], atol=1e-10)


def test_normalization_uses_only_training_samples_and_handles_constant_features():
    from ring_learning.dataset import Normalizer

    train = np.array([[1., 8.], [3., 8.]], dtype=np.float32)
    norm = Normalizer.fit(train)
    np.testing.assert_allclose(norm.mean, [2., 8.])
    assert np.isfinite(norm.transform([[100., 8.]])).all()
    np.testing.assert_allclose(norm.transform(train).mean(axis=0), [0., 0.])


def test_measurement_noise_preserves_relative_feature_consistency_and_phase():
    from ring_learning.dataset import noisy_observation

    obs = np.zeros(30, dtype=np.float32)
    obs[:9] = [.45, 0., .14, .49, .05, .1035, .43, -.01, .1025]
    obs[9:12] = obs[3:6] - obs[:3]
    obs[12:15] = obs[6:9] - obs[3:6]
    obs[21] = .01
    obs[22] = 1
    noisy = noisy_observation(obs, np.random.default_rng(9), std=.0002)
    np.testing.assert_allclose(noisy[9:12], noisy[3:6] - noisy[:3], atol=1e-8)
    np.testing.assert_allclose(noisy[12:15], noisy[6:9] - noisy[3:6], atol=1e-8)
    np.testing.assert_array_equal(noisy[21:], obs[21:])
