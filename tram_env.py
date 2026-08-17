"""
tram_env.py
Extracted verbatim from Tram_RL.ipynb (Sections: Environment, Helper Utilities,
Neural Network Architectures). No logic changes — only moved into a module so
it can be imported by agents.py / app.py.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# LINEAR TRAM NETWORK ENVIRONMENT  (PDF Section 2.0 — Environment Design)
# ---------------------------------------------------------------------------
class TramNetworkEnv:
    """
    5-station linear tram network simulator.

    Station 0  : Dispatch depot / Terminus A
    Stations 1-3: Intermediate boarding stops
    Station 4  : Terminus B

    Decision step = 5 minutes.
    Episode = 1080 minutes (18 hours) = 216 steps.
    """

    # Network constants
    N_STATIONS    = 5
    TRAM_CAPACITY = 80
    STEP_DURATION = 5      # minutes per decision step
    TOTAL_MINUTES = 1080   # 06:00 to 24:00
    TOTAL_STEPS   = 216    # 1080 / 5
    MAX_TRAMS     = 4

    # Reward weights (from PDF)
    R_DELIVERED = 2.0
    C_OP        = 40.0
    W_WAIT      = 0.2
    W_TRAVEL    = 0.05

    # Minutes over which demand ramps in/out of a peak window (realism only —
    # see _get_demand_rates docstring for why this is NOT applied to
    # _get_demand_context).
    DEMAND_RAMP_MINUTES = 30

    def __init__(self):
        self.reset()

    @staticmethod
    def _smoothstep(x):
        x = max(0.0, min(1.0, x))
        return x * x * (3 - 2 * x)

    def _demand_ramp_weight(self, minute, start, end, ramp):
        """Smoothstep weight in [0,1]: 0 before start-ramp, eases up to 1 by
        `start`, holds 1 through [start, end], eases back down to 0 by
        end+ramp. Used only to shape the Poisson arrival RATES below."""
        if minute < start - ramp:
            return 0.0
        if minute < start:
            return self._smoothstep((minute - (start - ramp)) / ramp)
        if minute <= end:
            return 1.0
        if minute <= end + ramp:
            return 1.0 - self._smoothstep((minute - end) / ramp)
        return 0.0

    def _get_demand_rates(self, minute):
        """Poisson arrival rates per intermediate station. Blended smoothly
        across the peak-window boundaries instead of the report's
        instantaneous step-jump — riders don't materialise the instant the
        clock hits 07:30, ridership ramps up/down over ~DEMAND_RAMP_MINUTES.
        Peak windows and target rate magnitudes are unchanged from the
        report; only the transition shape is smoothed."""
        off     = np.array([0.4, 0.2, 0.2, 0.1])
        morning = np.array([1.8, 0.8, 0.6, 0.4])
        evening = np.array([1.4, 1.0, 0.8, 0.6])
        wm = self._demand_ramp_weight(minute, 90, 210, self.DEMAND_RAMP_MINUTES)
        we = self._demand_ramp_weight(minute, 630, 750, self.DEMAND_RAMP_MINUTES)
        rates = off + wm * (morning - off) + we * (evening - off)
        return rates.tolist()

    def _get_demand_context(self, minute):
        """Deliberately left as the report's exact step function (1.0 /
        0.5 / 0.0) — NOT smoothed. _get_discrete_state() below buckets this
        with an exact `ctx == 0.0` / `ctx == 0.5` equality check to build the
        tabular Q-learning state index; smoothing this value would make
        those comparisons fail almost everywhere and silently corrupt the
        27-state table for QLearningAgent/MCQLearningAgent. Only the
        Poisson arrival rates above are smoothed for visual/behavioural
        realism — the RL state representation matches the report exactly."""
        if 90 <= minute < 210:
            return 1.0
        elif 630 <= minute < 750:
            return 0.5
        return 0.0

    def reset(self):
        self.minute           = 0
        self.step_count       = 0
        self.elapsed_dispatch = 0
        self.queues           = np.zeros(4, dtype=np.float32)
        self.trams            = []
        return self._get_continuous_state()

    def _get_continuous_state(self):
        q_norm    = np.clip(self.queues / 100.0, 0, 1)
        t_day     = self.minute / self.TOTAL_MINUTES
        t_elapsed = min(self.elapsed_dispatch / 60.0, 1.0)
        occ       = np.zeros(4, dtype=np.float32)
        for tram in self.trams:
            pos = int(tram['position'])
            if 0 <= pos <= 3:
                occ[pos] += sum(tram['passengers'].values())
        occ_norm   = np.clip(occ / self.TRAM_CAPACITY, 0, 1)
        demand_ctx = self._get_demand_context(self.minute)
        n_active   = min(len(self.trams), self.MAX_TRAMS) / self.MAX_TRAMS
        return np.array([
            q_norm[0], q_norm[1], q_norm[2], q_norm[3],
            t_day, t_elapsed,
            occ_norm[0], occ_norm[1], occ_norm[2], occ_norm[3],
            demand_ctx, n_active
        ], dtype=np.float32)

    def _get_discrete_state(self):
        total_q = float(np.sum(self.queues))
        q_bin = 0 if total_q < 10 else (1 if total_q < 30 else 2)
        ctx   = self._get_demand_context(self.minute)
        d_bin = 0 if ctx == 0.0 else (1 if ctx == 0.5 else 2)
        e     = self.elapsed_dispatch
        e_bin = 0 if e < 10 else (1 if e < 20 else 2)
        return q_bin * 9 + d_bin * 3 + e_bin

    def step(self, action):
        """action: 0=Wait, 1=Dispatch. Returns (next_state, reward, done, info)."""
        reward      = 0.0
        n_delivered = 0
        dispatched  = False

        # 1. Passenger arrivals
        rates = self._get_demand_rates(self.minute)
        for _ in range(self.STEP_DURATION):
            for i in range(4):
                self.queues[i] += np.random.poisson(rates[i])

        # 2. Dispatch action
        if action == 1:
            dispatched = True
            reward    -= self.C_OP
            self.elapsed_dispatch = 0
            self.trams.append({'position': 0, 'passengers': {}})
        else:
            self.elapsed_dispatch += self.STEP_DURATION

        # 3. Board passengers (FIFO, capacity-limited)
        for tram in self.trams:
            pos = int(tram['position'])
            if pos < 4:
                cap_left = self.TRAM_CAPACITY - sum(tram['passengers'].values())
                if cap_left > 0 and self.queues[pos] > 0:
                    boarding = min(int(self.queues[pos]), cap_left)
                    self.queues[pos] -= boarding
                    for _ in range(boarding):
                        dest = random.randint(pos + 1, 4)
                        tram['passengers'][dest] = tram['passengers'].get(dest, 0) + 1

        # 4. Move trams and alight passengers
        remaining = []
        for tram in self.trams:
            tram['position'] += 1
            pos = int(tram['position'])
            n_delivered += tram['passengers'].pop(pos, 0)
            if pos < self.N_STATIONS:
                remaining.append(tram)
        self.trams = remaining

        # 5. Reward components
        reward += self.R_DELIVERED * n_delivered
        reward -= self.W_WAIT   * float(np.sum(self.queues))
        reward -= self.W_TRAVEL * sum(sum(t['passengers'].values()) for t in self.trams)

        # 6. Advance time
        self.minute     += self.STEP_DURATION
        self.step_count += 1
        done = self.step_count >= self.TOTAL_STEPS

        info = {
            'n_delivered': n_delivered,
            'dispatched' : dispatched,
            'total_wait' : float(np.sum(self.queues))
        }
        return self._get_continuous_state(), reward, done, info


# ---------------------------------------------------------------------------
# HELPER UTILITIES
# ---------------------------------------------------------------------------
def moving_average(values, window=10):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')


class ExperimentLogger:
    """Per-episode logger used by train_and_save.py. Collects reward/
    delivered/dispatch history for one controller across a training run and
    summarises the final-window performance (report convention: averaged
    over the final 20 episodes) for the cross-controller comparison table."""

    def __init__(self, name, n_episodes, final_window=20):
        self.name = name
        self.n_episodes = n_episodes
        self.final_window = final_window
        self.reward_history = []
        self.delivered_history = []
        self.dispatch_history = []

    def log(self, episode, total_reward, n_delivered, n_dispatched):
        self.reward_history.append(total_reward)
        self.delivered_history.append(n_delivered)
        self.dispatch_history.append(n_dispatched)

    def summary(self):
        w = min(self.final_window, len(self.reward_history))
        avg = lambda hist: (sum(hist[-w:]) / w) if w else 0.0
        return {
            'name': self.name,
            'reward_history': self.reward_history,
            'delivered_history': self.delivered_history,
            'dispatch_history': self.dispatch_history,
            'final_reward': avg(self.reward_history),
            'final_delivered': avg(self.delivered_history),
            'final_dispatches': avg(self.dispatch_history),
        }


def continuous_action_to_discrete(a):
    """DDPG/TD3 continuous action a in [-1,1] -> binary decision (Dispatch if a>0)."""
    return 1 if a > 0.0 else 0


# ---------------------------------------------------------------------------
# NEURAL NETWORK ARCHITECTURES  (PDF: Neural Network Architecture section)
# ---------------------------------------------------------------------------
STATE_DIM  = 12
ACTION_DIM = 1
HIDDEN     = 32


class Actor(nn.Module):
    """Actor: 12 -> 32 -> 32 -> 1 (Tanh). Outputs a in [-1,1].

    The final Linear layer is given a small uniform init (+/-3e-3) instead
    of PyTorch's default Kaiming-uniform — the standard trick from the
    original DDPG paper (Lillicrap et al., 2015), used here to fix a real
    symptom, not just cosmetics: with the default init, the pre-Tanh sum
    over 32 hidden units routinely lands far enough from 0 that Tanh
    saturates to near -1 or +1 the moment the network is created. Because
    continuous_action_to_discrete() dispatches only when a>0, a saturated
    actor starts EPISODE 1 deterministically always-Wait (or always-
    Dispatch) rather than a roughly even mix. Always-Wait means station
    queues are never drained for a full 216-step episode, so the wait-time
    penalty (W_WAIT * sum(queues), charged every step) compounds into the
    deeply negative first-few-episodes reward seen in training. A near-0
    starting output lets exploration noise push the action across the 0
    threshold in both directions from episode 1, so queues get serviced
    at least some of the time before the critic has learned anything.
    This only changes how the (still random) initial weights are drawn —
    same architecture, same algorithm, same reward function."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),    nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM), nn.Tanh()
        )
        final_linear = self.net[-2]  # the Linear(HIDDEN, ACTION_DIM) just before Tanh
        nn.init.uniform_(final_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(final_linear.bias, -3e-3, 3e-3)

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    """Critic: 13 -> 32 -> 32 -> 1. Estimates Q(s,a)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),                 nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class ReplayBuffer:
    """Experience replay buffer (PDF: capacity=10,000, batch=32)."""
    def __init__(self, max_size=10_000):
        self.buf = deque(maxlen=max_size)

    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s),  dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(a),  dtype=torch.float32).view(-1, 1).to(DEVICE),
            torch.tensor(np.array(r),  dtype=torch.float32).view(-1, 1).to(DEVICE),
            torch.tensor(np.array(ns), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(d),  dtype=torch.float32).view(-1, 1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)