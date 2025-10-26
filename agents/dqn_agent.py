# agents/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from agents.replay_buffer import ReplayBuffer

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.adv = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.adv(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 buffer_size=100000, batch_size=128, tau=0.005, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=buffer_size)

        # exploration params
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state, epsilon=None):
        eps = self.epsilon if epsilon is None else epsilon
        if np.random.rand() < eps:
            return np.random.randint(0, self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.policy_net(state_t)
        return int(qvals.argmax().item())

    # alias for older code
    def select_action(self, state, epsilon=None):
        return self.act(state, epsilon)

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # current Q
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # double DQN target: action selection by policy_net, evaluation by target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target
        self.soft_update(self.tau)

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_( (1.0 - tau) * target_param.data + tau * param.data )

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
