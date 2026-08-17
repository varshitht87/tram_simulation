"""
train_and_save.py
==================
Standalone, offline multi-algorithm training orchestrator for the tram
dispatch environment.

Trains each of the project's 4 *trainable* controllers — Q-Learning,
MC-Q-Learning, DDPG, TD3 — for exactly 150 episodes apiece, in complete
isolation from one another, then runs each through a fixed test phase
with learning disabled. (Baseline is excluded: it has no learnable
parameters, so "train/test" doesn't apply to it.)

Isolation guarantees:
  - Each algorithm gets its own fresh TramNetworkEnv() and its own fresh
    agent instance (agents.make_agent) — nothing is shared across
    algorithms during training or testing.
  - Only running aggregates (sums/counts) are kept during the 150
    training episodes, never a full per-episode history — the saved
    stats files contain nothing but the final numbers.
  - Each algorithm's stats/model files are named independently and never
    merged, averaged, or overwritten by another algorithm's run.

Reset semantics: this is a script, not a long-running process with a
Reset button — re-running it *is* the reset (fresh Python process, fresh
random weights/tables, fresh envs). To honor "don't mix runs together",
any stats_*/model_* files already in OUTPUT_DIR from a previous run are
moved into a timestamped archive subfolder before this run writes
anything, rather than silently overwritten.

Run:
    python train_and_save.py
"""

import json
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch

from tram_env import TramNetworkEnv
from agents import make_agent, QLearningAgent, MCQLearningAgent, DDPGAgent, TD3Agent

OUTPUT_DIR = "artifacts"
TRAIN_EPISODES = 150
TEST_EPISODES = 20        # batch-script default for the post-training test phase
TEST_BASE_SEED = 1000     # same fixed-seed convention as the app's Comparison Mode
SEED = 42

# (agent name recognised by agents.make_agent, filename-safe id for
#  stats_<id>.json / model_<id>.pth). Order = execution order.
ALGORITHMS = [
    ("Q-Learning", "QLearning"),
    ("MC-Q-Learning", "MCQLearning"),
    ("DDPG", "DDPG"),
    ("TD3", "TD3"),
]


class RunningAggregate:
    """Tracks only the running sums needed for the final per-algorithm
    stats. Deliberately does NOT retain a per-episode history — matches
    the 'no episode-by-episode clutter in the saved stats' requirement."""

    def __init__(self):
        self.n = 0
        self.reward_sum = 0.0
        self.dispatch_sum = 0
        self.delivered_sum = 0

    def add(self, reward, dispatches, delivered):
        self.n += 1
        self.reward_sum += reward
        self.dispatch_sum += dispatches
        self.delivered_sum += delivered

    @property
    def avg_reward(self):
        return self.reward_sum / self.n if self.n else 0.0

    @property
    def avg_dispatches(self):
        return self.dispatch_sum / self.n if self.n else 0.0

    @property
    def total_delivered(self):
        return self.delivered_sum


def archive_previous_run(output_dir):
    """Script-level 'reset': never silently overwrite a previous run's
    stats/model files. If any exist, move them into a timestamped archive
    subfolder before this run starts writing."""
    os.makedirs(output_dir, exist_ok=True)
    existing = [
        f for f in os.listdir(output_dir)
        if f.startswith(("stats_", "model_")) and os.path.isfile(os.path.join(output_dir, f))
    ]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(output_dir, f"_previous_run_{stamp}")
    os.makedirs(archive_dir, exist_ok=True)
    for f in existing:
        shutil.move(os.path.join(output_dir, f), os.path.join(archive_dir, f))
    print(f"[reset] Archived {len(existing)} file(s) from a previous run -> {archive_dir}/\n")


def save_model(agent, agent_name, path):
    """Every algorithm's weights are saved under a uniform model_<id>.pth
    name. Q-Learning/MC-Q-Learning have no neural net — their 'model' is
    the learned Q-table — so it's wrapped as a tensor and torch.save'd
    purely for naming/format consistency with the actor-critic agents."""
    if isinstance(agent, (QLearningAgent, MCQLearningAgent)):
        torch.save({"q_table": torch.from_numpy(agent.q_table)}, path)
    elif isinstance(agent, (DDPGAgent, TD3Agent)):
        torch.save({"actor_state_dict": agent.actor.state_dict()}, path)
    else:
        raise ValueError(f"Don't know how to save a model for {agent_name}")


def run_training_phase(agent_name):
    """Exactly 150 training episodes, fully isolated from every other
    algorithm: fresh env, fresh agent, only running aggregates kept."""
    env = TramNetworkEnv()
    agent = make_agent(agent_name)
    agg = RunningAggregate()

    for episode in range(1, TRAIN_EPISODES + 1):
        env.reset()
        agent.reset_episode()
        ep_reward, ep_dispatches, ep_delivered = 0.0, 0, 0

        for _ in range(TramNetworkEnv.TOTAL_STEPS):
            action, aux = agent.select_action(env, training=True)
            ns, reward, done, info = env.step(action)

            if isinstance(agent, (QLearningAgent, MCQLearningAgent)):
                next_s_idx = env._get_discrete_state()
                agent.observe(aux["state_idx"], action, reward, next_s_idx, done)
            elif isinstance(agent, (DDPGAgent, TD3Agent)):
                agent.observe(aux["state"], aux["a_cont"], reward, ns, done)

            ep_reward += reward
            ep_dispatches += int(info["dispatched"])
            ep_delivered += info["n_delivered"]
            if done:
                break

        agent.end_episode()
        agg.add(ep_reward, ep_dispatches, ep_delivered)

        # Minimal progress only — no per-episode stats are written to disk.
        if episode % 25 == 0 or episode == TRAIN_EPISODES:
            print(f"  Algorithm {agent_name}: episode {episode}/{TRAIN_EPISODES}")

    return agent, agg


def run_test_phase(agent):
    """Fixed-length test phase: policy only, learning disabled — no
    agent.observe() / optimizer step is ever called here. Uses the same
    fixed test-seed convention (1000+) as the app's Comparison Mode."""
    agg = RunningAggregate()
    for seed in range(TEST_BASE_SEED, TEST_BASE_SEED + TEST_EPISODES):
        random.seed(seed)
        np.random.seed(seed)
        env = TramNetworkEnv()
        agent.reset_episode()
        ep_reward, ep_dispatches, ep_delivered = 0.0, 0, 0

        for _ in range(TramNetworkEnv.TOTAL_STEPS):
            action, _ = agent.select_action(env, training=False)  # no updates
            _, reward, done, info = env.step(action)
            ep_reward += reward
            ep_dispatches += int(info["dispatched"])
            ep_delivered += info["n_delivered"]
            if done:
                break

        agg.add(ep_reward, ep_dispatches, ep_delivered)
    return agg


def run_one_algorithm(agent_name, algo_id, output_dir):
    print(f"\n[{agent_name}] Training — {TRAIN_EPISODES} episodes...")
    agent, train_agg = run_training_phase(agent_name)

    stats_path = os.path.join(output_dir, f"stats_{algo_id}.json")
    model_path = os.path.join(output_dir, f"model_{algo_id}.pth")

    train_stats = {
        "algorithm": agent_name,
        "phase": "train",
        "episodes": TRAIN_EPISODES,
        "avg_dispatches": round(train_agg.avg_dispatches, 4),
        "avg_rewards": round(train_agg.avg_reward, 4),
        "total_passengers_delivered": train_agg.total_delivered,
    }
    with open(stats_path, "w") as f:
        json.dump(train_stats, f, indent=2)
    save_model(agent, agent_name, model_path)

    print(f"=== {agent_name} Training Summary ({TRAIN_EPISODES} episodes) ===")
    print(f"Avg Dispatches: {train_agg.avg_dispatches:.2f}")
    print(f"Avg Rewards: {train_agg.avg_reward:.2f}")
    print(f"Total Passengers Delivered: {train_agg.total_delivered}")
    print(f"Model saved to: {model_path}")
    print(f"Stats saved to: {stats_path}")

    # --- Testing phase: switch to test mode only, no parameter updates ---
    print(f"[{agent_name}] Testing — {TEST_EPISODES} episodes (policy only, no learning)...")
    test_agg = run_test_phase(agent)

    test_stats_path = os.path.join(output_dir, f"stats_{algo_id}_test.json")
    test_stats = {
        "algorithm": agent_name,
        "phase": "test",
        "episodes": TEST_EPISODES,
        "avg_dispatches": round(test_agg.avg_dispatches, 4),
        "avg_rewards": round(test_agg.avg_reward, 4),
        "total_passengers_delivered": test_agg.total_delivered,
    }
    with open(test_stats_path, "w") as f:
        json.dump(test_stats, f, indent=2)

    print(f"=== {agent_name} Test Summary ({TEST_EPISODES} episodes) ===")
    print(f"Avg Dispatches: {test_agg.avg_dispatches:.2f}")
    print(f"Avg Rewards: {test_agg.avg_reward:.2f}")
    print(f"Total Passengers Delivered: {test_agg.total_delivered}")
    print(f"Test stats saved to: {test_stats_path}\n")

    return train_stats, test_stats


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    archive_previous_run(OUTPUT_DIR)

    print(
        f"Training {len(ALGORITHMS)} algorithms independently — "
        f"{TRAIN_EPISODES} train episodes + {TEST_EPISODES} test episodes each.\n"
        f"(Baseline is excluded — it has no learnable parameters.)"
    )

    t0 = time.time()
    all_results = {}
    for agent_name, algo_id in ALGORITHMS:
        train_stats, test_stats = run_one_algorithm(agent_name, algo_id, OUTPUT_DIR)
        all_results[agent_name] = {"train": train_stats, "test": test_stats}

    print(f"All {len(ALGORITHMS)} algorithms complete in {time.time() - t0:.1f}s. "
          f"Artifacts written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
