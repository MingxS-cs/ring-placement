import numpy as np
import pytest


def test_recorded_controller_action_matches_executed_clipping():
    from ring_learning.generate import run_controller
    class Env:
        def reset(self, *args, **kwargs):
            return np.zeros(30, dtype=np.float32)
        def step(self, action):
            self.executed = action.copy()
            return np.zeros(30, dtype=np.float32), True, {}
        def success(self):
            return True
    env = Env()
    cfg = {"ring_xy_m": [0, 0], "bearing_xy_m": [.1, 0], "reset_seed": 1}
    result = run_controller(env, cfg, lambda e, o: np.array([1.25, 0, 0, 1.]))
    np.testing.assert_array_equal(result["actions"][0], env.executed)


@pytest.mark.parametrize("splits,limit", [([], None), (["test", "test"], None),
    (["absent"], None), (["test"], 0), (["test"], -1), (["test"], True)])
def test_invalid_evaluation_request_is_rejected(splits, limit):
    from ring_learning.evaluate import validate_evaluation_request
    with pytest.raises(ValueError):
        validate_evaluation_request({"test": [], "ood_test": []}, splits, limit)
