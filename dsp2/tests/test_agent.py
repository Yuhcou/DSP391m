import numpy as np
from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.ddqn_agent import DDQNAgent, AgentConfig
from dsp2.agents.replay import ReplayBuffer

def test_network_shapes_and_action():
    env = EGCSEnv(n_floors=4, m_elevators=2, capacity=3, seed=1)
    agent = DDQNAgent(env.state_size, env.N, env.M, config=AgentConfig(batch_size=4))
    s = env.reset()
    mask = env.action_mask()
    a = agent.select_action(s, mask, epsilon=0.0)
    assert a.shape == (env.M,)
    assert ((a >= 0) & (a <= 3)).all()


def test_replay_and_train_step():
    env = EGCSEnv(n_floors=4, m_elevators=2, capacity=3, seed=2)
    agent = DDQNAgent(env.state_size, env.N, env.M, config=AgentConfig(batch_size=4))
    buf = ReplayBuffer(100, env.state_size, env.M, seed=0)
    s = env.reset()
    for _ in range(20):
        a = agent.select_action(s, env.action_mask(), epsilon=1.0)
        s2, r, done, info = env.step(a)
        buf.add(s, a, r, s2, done)
        s = env.reset() if done else s2
    batch = buf.sample(4)
    def mask_fn(st):
        # reconstruct mask from state vector
        from dsp2.agents.masks import legal_action_mask_from_state
        return legal_action_mask_from_state(st, env.N, env.M)
    loss, q_mean = agent.train_step(batch, mask_fn)
    assert isinstance(loss, float)
    assert isinstance(q_mean, float)

