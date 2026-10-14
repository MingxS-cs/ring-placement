import numpy as np
import pytest
import torch


class Plan(torch.nn.Module):
    def forward(self, images, tokens):
        points = torch.tensor([.45, 0., .16, -1.]).repeat(1,23,1)
        return points, torch.zeros(1,2,256,256)


class OwnStateOnly:
    command_xyz = np.array([.40,0.,.16])
    tool_xyz = command_xyz.copy()
    finger_opening = .006
    def render_policy(self):
        return np.full((256,256,3),100,dtype=np.uint8)
    def __getattr__(self, name):
        raise AssertionError('Privileged environment access: '+name)


def test_rgb_language_plan_uses_only_own_state_and_bounded_actions():
    from ring_learning.vla_control import VLAController
    controller = VLAController(Plan(), 'put the ring inside the bearing')
    env = OwnStateOnly()
    for _ in range(400):
        action = controller(env)
        assert np.linalg.norm(action[:3]*.003) <= .003000001
        env.command_xyz = env.command_xyz+action[:3]*.003
        env.tool_xyz = env.command_xyz.copy()
        if controller.finished:
            break
    assert controller.finished and controller.neural_forward_calls == 1
    assert env.command_xyz == pytest.approx([.45,0.,.16],abs=1e-6)


def test_black_image_enters_same_model_without_oracle_fallback():
    from ring_learning.vla_control import VLAController
    class Black(Plan):
        def forward(self, images, tokens):
            assert images.max() == 0.
            return super().forward(images,tokens)
    controller = VLAController(Black(), 'put the ring left of the bearing', ablation='black_image')
    controller(OwnStateOnly())
    assert controller.neural_forward_calls == 1


def test_invalid_or_unknown_plan_is_rejected():
    from ring_learning.vla_control import VLAController
    with pytest.raises(ValueError,match='Unsupported'):
        VLAController(Plan(),'make tea')
    class Invalid(Plan):
        def forward(self,images,tokens):
            points, logits = super().forward(images,tokens)
            points[0,1,0] = float('nan')
            return points,logits
    with pytest.raises(ValueError,match='finite'):
        VLAController(Invalid(),'put the ring inside the bearing')(OwnStateOnly())
