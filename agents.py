"""
agents.py
Thin per-step wrappers around the exact algorithms from Tram_RL.ipynb.
Same hyperparameters and update math as the notebook's train_* functions,
just broken into single-step calls so the Streamlit app can drive one
decision at a time instead of running a full batch training loop.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tram_env import (
    Actor, Critic, ReplayBuffer, soft_update,
    continuous_action_to_discrete, DEVICE, TramNetworkEnv,
)

ACTION_NAMES = {0: 'Wait', 1: 'Dispatch'}


# ---------------------------------------------------------------------------
# 0. BASELINE — Fixed-Time Controller (dispatch every 15 min / 3 steps)
# ---------------------------------------------------------------------------
class BaselineAgent:
    name = 'Baseline'
    is_trainable = False

    def __init__(self):
        self.timer = 0

    def reset_episode(self):
        self.timer = 0

    def select_action(self, env, training=False):
        action = 1 if (self.timer % 3 == 0) else 0
        self.timer += 1
        return action, {}

    def observe(self, state, action, reward, next_state, done):
        pass  # no learning

    def end_episode(self):
        pass


# ---------------------------------------------------------------------------
# 1. TABULAR Q-LEARNING (27 discrete states)
# ---------------------------------------------------------------------------
class QLearningAgent:
    name = 'Q-Learning'
    is_trainable = True

    def __init__(self, q_table=None):
        self.alpha         = 0.10
        self.gamma         = 0.99
        self.epsilon       = 0.50
        self.epsilon_min   = 0.02
        self.epsilon_decay = 0.96
        self.q_table = q_table if q_table is not None else np.zeros((27, 2))

    def reset_episode(self):
        pass

    def select_action(self, env, training=True):
        s_idx = env._get_discrete_state()
        if training and random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = int(np.argmax(self.q_table[s_idx]))
        return action, {'state_idx': s_idx, 'q_values': self.q_table[s_idx].copy()}

    def observe(self, state, action, reward, next_state, done):
        s_idx, ns_idx = state, next_state  # discrete indices passed in by caller
        self.q_table[s_idx, action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[ns_idx]) - self.q_table[s_idx, action]
        )

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# 2. MONTE CARLO-INITIALIZED Q-LEARNING
# ---------------------------------------------------------------------------
def mc_init_q_table(mc_episodes=100, gamma=0.99, n_states=27, n_actions=2):
    """Phase 1: pure random exploration -> first-visit MC returns -> Q-table init."""
    ret_sum   = np.zeros((n_states, n_actions))
    ret_count = np.zeros((n_states, n_actions))
    q_table   = np.zeros((n_states, n_actions))
    env       = TramNetworkEnv()

    for _ in range(mc_episodes):
        env.reset()
        memory = []
        for _ in range(TramNetworkEnv.TOTAL_STEPS):
            s_idx  = env._get_discrete_state()
            action = random.choice([0, 1])
            _, reward, done, _ = env.step(action)
            memory.append((s_idx, action, reward))
            if done:
                break

        G, visited = 0.0, set()
        for s, a, r in reversed(memory):
            G = r + gamma * G
            if (s, a) not in visited:
                ret_sum[s, a]   += G
                ret_count[s, a] += 1
                visited.add((s, a))

    mask = ret_count > 0
    q_table[mask] = ret_sum[mask] / ret_count[mask]
    return q_table


class MCQLearningAgent(QLearningAgent):
    """Same update rule as QLearningAgent; only the Q-table initialisation differs."""
    name = 'MC-Q-Learning'

    def __init__(self):
        super().__init__(q_table=mc_init_q_table())


# ---------------------------------------------------------------------------
# 3. DDPG
# ---------------------------------------------------------------------------
class DDPGAgent:
    name = 'DDPG'
    is_trainable = True

    def __init__(self):
        self.gamma      = 0.99
        self.tau        = 0.005
        self.batch_size = 32
        self.sigma      = 0.30
        self.sigma_min  = 0.05
        self.sigma_decay = 0.96

        self.actor         = Actor().to(DEVICE)
        self.target_actor  = Actor().to(DEVICE)
        self.critic        = Critic().to(DEVICE)
        self.target_critic = Critic().to(DEVICE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.buffer     = ReplayBuffer()

    def reset_episode(self):
        pass

    def select_action(self, env, training=True):
        state = env._get_continuous_state()
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a_cont = self.actor(st).cpu().numpy()[0, 0]
        if training:
            a_cont = float(np.clip(a_cont + np.random.normal(0, self.sigma), -1, 1))
        action = continuous_action_to_discrete(a_cont)
        return action, {'a_cont': a_cont, 'state': state}

    def observe(self, state, action, reward, next_state, done):
        # here `action` is the continuous a_cont value (caller passes aux['a_cont'])
        self.buffer.add(state, action, reward, next_state, float(done))
        if len(self.buffer) >= self.batch_size:
            sb, ab, rb, nsb, db = self.buffer.sample(self.batch_size)
            with torch.no_grad():
                na = self.target_actor(nsb)
                y  = rb + self.gamma * (1 - db) * self.target_critic(nsb, na)
            c_loss = nn.MSELoss()(self.critic(sb, ab), y)
            self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()

            a_loss = -self.critic(sb, self.actor(sb)).mean()
            self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

            soft_update(self.target_actor,  self.actor,  self.tau)
            soft_update(self.target_critic, self.critic, self.tau)

    def end_episode(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)


# ---------------------------------------------------------------------------
# 4. TD3
# ---------------------------------------------------------------------------
class TD3Agent:
    name = 'TD3'
    is_trainable = True

    def __init__(self):
        self.gamma        = 0.99
        self.tau          = 0.005
        self.batch_size   = 32
        self.sigma        = 0.30
        self.sigma_min    = 0.05
        self.sigma_decay  = 0.96
        self.policy_noise = 0.20
        self.noise_clip   = 0.50
        self.policy_delay = 2
        self.total_steps  = 0

        self.actor           = Actor().to(DEVICE)
        self.target_actor    = Actor().to(DEVICE)
        self.critic_1        = Critic().to(DEVICE)
        self.critic_2        = Critic().to(DEVICE)
        self.target_critic_1 = Critic().to(DEVICE)
        self.target_critic_2 = Critic().to(DEVICE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_opt    = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_1_opt = optim.Adam(self.critic_1.parameters(), lr=1e-3)
        self.critic_2_opt = optim.Adam(self.critic_2.parameters(), lr=1e-3)
        self.buffer       = ReplayBuffer()

    def reset_episode(self):
        pass

    def select_action(self, env, training=True):
        state = env._get_continuous_state()
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a_cont = self.actor(st).cpu().numpy()[0, 0]
        if training:
            a_cont = float(np.clip(a_cont + np.random.normal(0, self.sigma), -1, 1))
        action = continuous_action_to_discrete(a_cont)
        return action, {'a_cont': a_cont, 'state': state}

    def observe(self, state, action, reward, next_state, done):
        self.total_steps += 1
        self.buffer.add(state, action, reward, next_state, float(done))
        if len(self.buffer) >= self.batch_size:
            sb, ab, rb, nsb, db = self.buffer.sample(self.batch_size)

            with torch.no_grad():
                noise = torch.clamp(
                    torch.normal(0, self.policy_noise, ab.shape).to(DEVICE),
                    -self.noise_clip, self.noise_clip)
                na    = torch.clamp(self.target_actor(nsb) + noise, -1, 1)
                q_min = torch.min(self.target_critic_1(nsb, na), self.target_critic_2(nsb, na))
                y     = rb + self.gamma * (1 - db) * q_min

            loss_c1 = nn.MSELoss()(self.critic_1(sb, ab), y)
            self.critic_1_opt.zero_grad(); loss_c1.backward(); self.critic_1_opt.step()

            loss_c2 = nn.MSELoss()(self.critic_2(sb, ab), y)
            self.critic_2_opt.zero_grad(); loss_c2.backward(); self.critic_2_opt.step()

            if self.total_steps % self.policy_delay == 0:
                a_loss = -self.critic_1(sb, self.actor(sb)).mean()
                self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

                soft_update(self.target_actor,    self.actor,    self.tau)
                soft_update(self.target_critic_1, self.critic_1, self.tau)
                soft_update(self.target_critic_2, self.critic_2, self.tau)

    def end_episode(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
AGENT_REGISTRY = {
    'Baseline':      BaselineAgent,
    'Q-Learning':    QLearningAgent,
    'MC-Q-Learning': MCQLearningAgent,
    'DDPG':          DDPGAgent,
    'TD3':           TD3Agent,
}


def make_agent(name):
    return AGENT_REGISTRY[name]()
