import numpy as np


def test_retargeted_action_chunk_has_three_distinct_final_goals_and_contact_stages():
    from ring_learning.vla_data import action_chunk
    class Shape:
        def waypoint(self,start,goal,progress):
            return (1-progress)*np.asarray(start)+progress*np.asarray(goal)
    points = [action_chunk([.4,.03],[.5,0],name,Shape()) for name in ('inside','left','right')]
    assert all(p.shape == (23,4) for p in points)
    assert np.allclose([p[21,0] for p in points],[.5,.44,.56])
    assert all(p[2,3] == -1 and p[21,3] == 1 for p in points)
    assert all(np.allclose(p[2,:2],[.4,.04125]) for p in points)
