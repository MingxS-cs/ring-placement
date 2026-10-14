import pytest
import torch


def test_text_tokens_are_words_and_reject_untrained_instructions():
    from ring_learning.vla_policy import encode_text
    tokens = encode_text([
        "place the ring inside the bearing",
        "place the ring to the left of the bearing",
        "place the ring to the right of the bearing",
    ])
    assert tokens.dtype == torch.long and tokens.shape[0] == 3
    assert not torch.equal(tokens[1], tokens[2])
    with pytest.raises(ValueError, match="instruction"):
        encode_text(["tighten a bottle cap"])
    with pytest.raises(ValueError, match="instruction"):
        encode_text(["ring bearing left"])


def test_learned_image_masks_map_centres_to_fixed_camera_world_coordinates():
    from ring_learning.vla_policy import TinyVLA
    model = TinyVLA()
    logits = torch.full((1, 2, 65, 65), -40.)
    logits[0, 0, 16, 16] = 40.
    logits[0, 1, 48, 48] = 40.
    features = model.scene_features_from_logits(logits)
    expected = torch.tensor([.47+(16.5/65-.5)*.56, (.5-16.5/65)*.56,
                             .47+(48.5/65-.5)*.56, (.5-48.5/65)*.56, 1.])
    torch.testing.assert_close(features[0, :5], expected)
    moved = logits.flip(-1)
    moved_features = model.scene_features_from_logits(moved)
    assert moved_features[0, 0] > features[0, 0]
    assert moved_features[0, 2] < features[0, 2]


def test_forward_returns_full_action_chunk_and_has_gradients_for_both_modalities():
    from ring_learning.vla_policy import TinyVLA, encode_text
    torch.manual_seed(4)
    model = TinyVLA(n_waypoints=23)
    images = torch.rand(2, 3, 32, 32)
    tokens = encode_text(["place the ring inside the bearing", "place the ring to the left of the bearing"])
    waypoints, logits = model(images, tokens)
    assert waypoints.shape == (2, 23, 4)
    assert logits.shape == (2, 2, 32, 32)
    waypoints.square().mean().backward()
    assert model.token_embedding.weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model.vision.parameters() if p.grad is not None) > 0
    assert torch.isfinite(waypoints).all()


def test_rgb_translation_changes_predicted_coordinates_without_supplied_object_state():
    from ring_learning.vla_policy import TinyVLA
    model = TinyVLA()
    with torch.no_grad():
        for parameter in model.vision.parameters():
            parameter.zero_()
        model.vision[0].weight[0, 0, 1, 1] = 1.
        model.vision[0].weight[1, 1, 1, 1] = 1.
        model.vision[2].weight[0, 0, 1, 1] = 1.
        model.vision[2].weight[1, 1, 1, 1] = 1.
        model.vision[4].weight[0, 0, 0, 0] = 30.
        model.vision[4].weight[1, 1, 0, 0] = 30.
        model.vision[4].bias[:] = -15.
    images = torch.zeros(1, 3, 32, 32)
    images[0, 0, 8, 8] = 1.
    images[0, 1, 24, 24] = 1.
    left_features, _ = model.scene_features(images)
    right_features, _ = model.scene_features(images.flip(-1))
    assert right_features[0, 0]-left_features[0, 0] > .25
    assert right_features[0, 2]-left_features[0, 2] < -.25


def test_direction_basis_remains_finite_when_predicted_objects_coincide():
    from ring_learning.vla_policy import scene_feature_basis
    features = scene_feature_basis(torch.tensor([[.47, 0., .47, 0.]]))
    assert features.shape == (1, 10) and torch.isfinite(features).all()
    torch.testing.assert_close(features[0, 5:], torch.zeros(5))


def test_foreground_centroid_weights_are_nearly_uniform_despite_brightness_variation():
    from ring_learning.vla_policy import TinyVLA
    model = TinyVLA()
    logits = torch.full((1, 2, 64, 64), -30.)
    logits[:, :, 24, 16] = 10.
    logits[:, :, 24, 48] = 20.
    features = model.scene_features_from_logits(logits)
    # Both pixels are confidently foreground; their mean, not the brighter one,
    # should determine the object center.
    assert abs(float(features[0, 0])-(.47+(.5/64)*.56)) < .0001


def test_same_scene_different_words_can_learn_different_robot_goals():
    from ring_learning.vla_policy import TinyVLA, encode_text
    torch.manual_seed(11)
    torch.set_num_threads(2)
    model = TinyVLA(n_waypoints=1)
    tokens = encode_text(["place the ring to the left of the bearing", "place the ring to the right of the bearing"])
    features = torch.zeros((2, model.feature_dim))
    features[:, :5] = torch.tensor([.35, -.05, .49, .02, 1.])
    expected = torch.tensor([[[.43, .02, .105, 1.]], [[.55, .02, .105, 1.]]])
    optimizer = torch.optim.Adam(model.decoder_parameters(), lr=.02)
    for _ in range(180):
        optimizer.zero_grad()
        loss = (model.predict_from_scene_features(features, tokens)-expected).square().mean()
        loss.backward()
        optimizer.step()
    actual = model.predict_from_scene_features(features, tokens)
    torch.testing.assert_close(actual, expected, atol=.003, rtol=0.)
    assert actual[0, 0, 0] < actual[1, 0, 0]


def test_different_length_paraphrases_learn_the_same_left_and_right_goal_offsets():
    from ring_learning.vla_policy import TinyVLA, encode_text
    torch.manual_seed(9)
    torch.set_num_threads(2)
    model = TinyVLA(n_waypoints=1)
    phrases = ["put the ring left of the bearing", "place the ring to the left of the bearing",
               "put the ring right of the bearing", "place the ring to the right of the bearing"]
    tokens = encode_text(phrases)
    features = torch.zeros(4, 10)
    features[:, :5] = torch.tensor([.35, -.05, .49, .02, 1.])
    targets = torch.tensor([[[.43, .02, .105, 1.]], [[.43, .02, .105, 1.]],
                            [[.55, .02, .105, 1.]], [[.55, .02, .105, 1.]]])
    optimizer = torch.optim.Adam(model.decoder_parameters(), lr=.02)
    for _ in range(450):
        optimizer.zero_grad()
        loss = (model.predict_from_scene_features(features, tokens)-targets).square().mean()
        loss.backward()
        optimizer.step()
    actual = model.predict_from_scene_features(features, tokens)
    torch.testing.assert_close(actual, targets, atol=.0005, rtol=0.)


@pytest.mark.parametrize("tokens", [torch.zeros(1, 4, dtype=torch.long), torch.tensor([[-1]]), torch.tensor([[999]]), torch.ones(1, 2)])
def test_invalid_text_token_batches_are_rejected(tokens):
    from ring_learning.vla_policy import TinyVLA
    with pytest.raises(ValueError, match="text tokens"):
        TinyVLA().text_features(tokens)


@pytest.mark.parametrize("image", [torch.zeros(1, 3, 10, 10).fill_(float("nan")), torch.ones(1, 3, 10, 10)*255, torch.zeros(1, 10, 10, 3)])
def test_invalid_rgb_input_is_rejected(image):
    from ring_learning.vla_policy import TinyVLA, encode_text
    with pytest.raises(ValueError, match="image"):
        TinyVLA()(image, encode_text(["place the ring inside the bearing"]))


def test_checkpoint_restores_exact_predictions_and_rejects_changed_source(tmp_path):
    from ring_learning.vla_policy import TinyVLA, encode_text, save_vla_policy, load_vla_policy
    torch.manual_seed(3)
    model = TinyVLA(n_waypoints=3)
    images = torch.rand(1, 3, 16, 16)
    tokens = encode_text(["place the ring inside the bearing"])
    expected = model(images, tokens)[0]
    path = tmp_path/"policy.pt"
    save_vla_policy(path, model, {"training_only": True})
    restored, metadata = load_vla_policy(path)
    torch.testing.assert_close(restored(images, tokens)[0], expected, rtol=0., atol=0.)
    assert metadata["training_only"]
    payload = torch.load(path, weights_only=True)
    payload["source_sha256"] = "0"*64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="source"):
        load_vla_policy(path)
