import numpy as np
from dsp2.env.egcs_env import EGCSEnv

def test_state_size_and_reset():
    env = EGCSEnv(n_floors=5, m_elevators=2, capacity=4, seed=123)
    s = env.reset()
    assert s.shape[0] == env.state_size
    # positions and directions initially zero
    assert np.allclose(s[:2], 0)


def test_mask_boundaries():
    env = EGCSEnv(n_floors=3, m_elevators=1, capacity=2, seed=123)
    env.reset()
    env.positions[0] = 2  # top
    mask = env.action_mask()
    assert mask.shape == (1,4)
    assert mask[0,1] == False  # up illegal
    env.positions[0] = 0  # bottom
    mask = env.action_mask()
    assert mask[0,2] == False  # down illegal


def test_step_shapes():
    env = EGCSEnv(n_floors=5, m_elevators=2, capacity=4, seed=0)
    s = env.reset()
    a = np.array([0, 1], dtype=np.int64)
    s2, r, done, info = env.step(a)
    assert s2.shape == s.shape
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert 'n_waiting' in info

