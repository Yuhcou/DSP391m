from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import DQNNet, VDNMixer

@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    target_update_steps: int = 10000
    tau: float = 1.0  # if <1.0 use soft update
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_steps: int = 200000
    grad_clip: float = 5.0
    # Improvements
    dueling: bool = True
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.0
    # VDN/CTDE
    use_vdn: bool = True
    use_central_bias: bool = False

class DDQNAgent:
    def __init__(self, state_size: int, n_floors: int, m_elevators: int, device: str = None, config: AgentConfig = AgentConfig()):
        self.state_size = state_size
        self.N = n_floors
        self.M = m_elevators
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DQNNet(state_size, m_elevators, hidden=256, dueling=self.cfg.dueling).to(self.device)
        self.target = DQNNet(state_size, m_elevators, hidden=256, dueling=self.cfg.dueling).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.mixer = VDNMixer(state_size, use_central_bias=self.cfg.use_central_bias) if self.cfg.use_vdn else None
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.step_counter = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.step_counter / max(1, self.cfg.decay_steps))
        return float(self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start))

    def select_action(self, state: np.ndarray, mask: np.ndarray, epsilon: float = None) -> np.ndarray:
        eps = self.epsilon() if epsilon is None else epsilon
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy(s)  # (1, M, 4)
        q = q.cpu().squeeze(0)  # (M,4)
        q_np = q.numpy()
        # mask illegal
        q_np[~mask] = -1e9
        a = np.zeros(self.M, dtype=np.int64)
        for i in range(self.M):
            if np.random.random() < eps:
                legal = np.where(mask[i])[0]
                a[i] = int(np.random.choice(legal))
            else:
                a[i] = int(np.argmax(q_np[i]))
        return a

    def train_step(self, batch, mask_next_fn: Callable[[np.ndarray], np.ndarray],
                   per_indices: Optional[np.ndarray] = None, per_weights: Optional[np.ndarray] = None) -> Tuple[float, float, Optional[np.ndarray]]:
        s, a, r, s2, d = batch
        B = s.shape[0]
        s_t = torch.from_numpy(s).float().to(self.device)
        a_t = torch.from_numpy(a).long().to(self.device)  # (B,M)
        r_t = torch.from_numpy(r).float().to(self.device)  # (B,)
        s2_t = torch.from_numpy(s2).float().to(self.device)
        d_t = torch.from_numpy(d).float().to(self.device)
        # Q(s', a*) with masking
        with torch.no_grad():
            q_next_policy = self.policy(s2_t)  # (B,M,4)
            # build mask for s'
            masks = []
            s2_np = s2  # (B, S)
            for b in range(B):
                m = mask_next_fn(s2_np[b])  # (M,4)
                masks.append(m)
            mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(self.device)  # (B,M,4)
            q_next_policy_masked = q_next_policy.masked_fill(~mask_tensor, -1e9)
            a_star = torch.argmax(q_next_policy_masked, dim=-1)  # (B,M)
            q_target_all = self.target(s2_t)  # (B,M,4)
            q_target_eval = q_target_all.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)  # (B,M)
            # Per-elevator targets
            y_per_elev = r_t.unsqueeze(1) + self.cfg.gamma * q_target_eval * (1.0 - d_t.unsqueeze(1))  # (B,M)

        # Q(s, a)
        q_curr = self.policy(s_t)  # (B,M,4)
        q_taken_per_elev = q_curr.gather(-1, a_t.unsqueeze(-1)).squeeze(-1)  # (B,M)
        # TD error per elevator
        td_errors_per_elev = q_taken_per_elev - y_per_elev  # (B,M)
        # Aggregate with VDN mixer (sum over elevators) possibly with central bias
        if self.mixer is not None:
            q_agg = self.mixer(q_taken_per_elev, s_t).squeeze(-1)  # (B,)
            y_agg = self.mixer(y_per_elev, s_t).squeeze(-1)        # (B,)
            td_errors = q_agg - y_agg                               # (B,)
        else:
            # Mean over elevators
            td_errors = td_errors_per_elev.mean(dim=-1)            # (B,)

        # PER weighting
        if per_weights is not None:
            w = torch.from_numpy(per_weights).float().to(self.device)  # (B,)
            loss = (w * (td_errors ** 2)).mean()
        else:
            loss = (td_errors ** 2).mean()

        self.optim.zero_grad()
        loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.optim.step()
        self.step_counter += 1

        # target update
        if self.cfg.tau < 1.0:
            with torch.no_grad():
                for p, t in zip(self.policy.parameters(), self.target.parameters()):
                    t.data.mul_(1 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)
        elif self.step_counter % max(1, self.cfg.target_update_steps) == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # For logging: mean Q over elevators
        q_mean = float(q_taken_per_elev.detach().mean().item())

        # Compute absolute TD errors for PER priorities update (per sample)
        per_td = None
        if per_indices is not None:
            per_td = np.abs(td_errors.detach().cpu().numpy())  # (B,)

        return float(loss.item()), q_mean, per_td
