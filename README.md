# 🚊 Tram Dispatch RL — 3D Simulator

A reinforcement-learning testbed for **tram dispatch scheduling**, paired with a live, browser-based **3D visualization** built on Streamlit. Five controllers — a fixed-time baseline, tabular Q-Learning, Monte Carlo Q-Learning, DDPG, and TD3 — compete to learn *when to dispatch a tram* along a 5-station linear network under realistic morning/evening peak demand.

<p align="center">
  <em>Watch trained agents dispatch trams in real time, compare algorithms head-to-head, and inspect the reward function driving every decision.</em>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [The Environment](#the-environment)
- [Agents / Algorithms](#agents--algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the live 3D simulator](#run-the-live-3d-simulator)
  - [Train all agents from scratch](#train-all-agents-from-scratch)
- [Configuration](#configuration)
- [Reward Function](#reward-function)
- [Report](#report)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Transit agencies constantly trade off two costs: dispatching a tram too often wastes operating budget, dispatching too rarely leaves passengers waiting. This project models that trade-off as a **sequential decision-making problem** and solves it with reinforcement learning.

A 5-station linear tram network is simulated at 5-minute resolution across an 18-hour operating day (06:00–24:00). At every step, a controller chooses to **Wait** or **Dispatch** a new tram from the depot, and the tram network evolves — passengers arrive via a time-varying Poisson process, board waiting trams, and alight at their destination.

The project ships with two ways to interact with the environment:

1. **A live 3D dashboard** (`app.py`) — pick an agent, watch it operate the network in an animated 3D scene, and inspect live metrics (queues, occupancy, reward decomposition, efficiency).
2. **An offline training pipeline** (`train_and_save.py`) — trains all four learnable agents from scratch, in isolation, and saves their models/Q-tables plus summary statistics.

## Features

- **5 controllers, one environment** — Baseline (fixed-time), tabular Q-Learning, Monte Carlo Q-Learning, DDPG, and TD3, all driving the exact same `TramNetworkEnv`.
- **Live 3D scene** (Three.js, embedded via `st.components.v1`) — realistic stations with shelters, benches, lamps, and displays; a scripted tram dwell sequence (decelerate → dock → doors open → alight → board → doors close → depart); day/night sky lighting keyed to time-of-day; 5 camera modes (Top View, Tracking, Driver View, Station View, Free Orbit); and Clear / Rain / Fog weather.
- **Transport control-centre dashboard** — simulation clock, scenario, weather, live reward, queue lengths, tram occupancy, congestion index, and success rate.
- **Named scenario presets** — jump straight into Morning Peak, Evening Peak, Uneven Demand, Late Night, and more, without waiting for the simulation clock to get there.
- **Comparison mode** — run multiple agents against the same fixed test seeds for an apples-to-apples performance comparison.
- **Reward decomposition panel** — see exactly how much of the reward each step came from deliveries, wait-time penalty, travel-time penalty, and operating cost.
- **Q-table heatmap** — visualize the learned policy of the tabular agents.
- **Reproducible offline training** — `train_and_save.py` trains every learnable agent for 150 episodes in complete isolation (fresh env, fresh weights, fresh RNG state per algorithm) and archives previous runs instead of overwriting them.

## The Environment

`TramNetworkEnv` (in `tram_env.py`) is a custom Gym-style environment:

| | |
|---|---|
| **Network** | 5 stations — Station 0 (depot / Terminus A), Stations 1–3 (intermediate stops), Station 4 (Terminus B) |
| **Episode length** | 216 steps (18 simulated hours, 06:00–24:00) |
| **Decision interval** | 5 minutes per step |
| **Action space** | `0` = Wait, `1` = Dispatch a new tram from the depot |
| **Tram capacity** | 80 passengers |
| **Max concurrent trams** | 4 |
| **Demand model** | Time-varying Poisson arrivals per station, with smoothed ramps in/out of two peak windows: Morning Peak (07:30–09:30) and Evening Peak (16:30–18:30) |
| **State (continuous, DDPG/TD3)** | 12-dim vector — normalized queue lengths (4), time-of-day, time-since-last-dispatch, normalized tram occupancy per station (4), demand context, active-tram fraction |
| **State (discrete, Q-Learning)** | 27-state tabular encoding — queue-size bucket × demand-context bucket × elapsed-time-since-dispatch bucket |

## Agents / Algorithms

| Agent | Type | Description |
|---|---|---|
| **Baseline** | Fixed-time | Dispatches every 3 steps (15 min), no learning |
| **Q-Learning** | Tabular, discrete | Classic ε-greedy tabular Q-learning over the 27-state encoding |
| **MC-Q-Learning** | Tabular, discrete | Every-visit Monte Carlo control, same state/action space as Q-Learning |
| **DDPG** | Deep RL, continuous→discrete | Deep Deterministic Policy Gradient with actor/critic networks and a replay buffer; continuous output mapped to a discrete Wait/Dispatch decision |
| **TD3** | Deep RL, continuous→discrete | Twin Delayed DDPG — adds twin critics, delayed policy updates, and target policy smoothing on top of DDPG |

All agents are implemented in `agents.py` as thin, single-step wrappers around the same algorithm logic originally developed and validated in the project's research notebook — so the app can drive one decision at a time instead of a full batch training loop.

## Project Structure

```
tram_sim/
├── app.py                     # Streamlit entry point — live 3D simulator & dashboard
├── train_and_save.py          # Offline training pipeline for all 4 learnable agents
├── agents.py                  # Baseline, Q-Learning, MC-Q-Learning, DDPG, TD3
├── tram_env.py                # TramNetworkEnv + Actor/Critic networks + ReplayBuffer
├── config.py                  # Visual-layer constants (colors, timing, camera presets)
├── requirements.txt
│
├── simulation/
│   ├── scenario_manager.py    # Named demand-scenario presets
│   └── reward_decomposer.py   # Splits reward into interpretable components
│
├── ui/
│   ├── scene3d.py             # Three.js 3D scene builder
│   ├── dashboard.py           # Control-centre metrics panel
│   ├── camera.py              # Camera mode controls
│   ├── reward_panel.py        # Live reward decomposition chart
│   └── qtable_heatmap.py      # Q-table policy heatmap
│
├── actor_ddpg.pt               # Trained DDPG actor weights
├── actor_td3.pt                # Trained TD3 actor weights
├── q_table_qlearning.pkl       # Trained Q-Learning table
├── q_table_mc.pkl              # Trained MC-Q-Learning table
├── results.pkl                 # Saved training/comparison results
└── Reinforcement Learning .pdf # Project report (environment & reward design spec)
```

## Installation

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/<your-username>/tram_sim.git
cd tram_sim
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies:** `streamlit`, `numpy`, `torch`, `matplotlib`, `plotly`

## Usage

### Run the live 3D simulator

```bash
streamlit run app.py
```

This opens the dashboard in your browser. From there you can:

- Pick an agent (Baseline, Q-Learning, MC-Q-Learning, DDPG, TD3)
- Choose a scenario preset or run the live full-day simulation
- Adjust simulation speed, camera mode, and weather
- Watch live metrics update as the agent dispatches trams

### Train all agents from scratch

```bash
python train_and_save.py
```

This trains Q-Learning, MC-Q-Learning, DDPG, and TD3 for 150 episodes each, fully isolated from one another (fresh environment and fresh random weights per algorithm), then evaluates each for 20 test episodes with learning disabled using fixed seeds (1000–1019). Any existing output files are archived to a timestamped subfolder before the new run writes anything — previous results are never silently overwritten.

Outputs are written to `artifacts/`.

## Configuration

`config.py` controls only the **visual layer** — colors, lighting keyframes, camera presets, weather options, and scene layout — and is deliberately kept separate from the environment's RL constants (which live in `tram_env.py`), so tweaking the look of the simulation can never accidentally change how the environment behaves.

## Reward Function

At each step, the reward is:

```
reward = R_DELIVERED × passengers_delivered
       − C_OP        × dispatch_cost        (only if a tram is dispatched)
       − W_WAIT       × total_passengers_waiting
       − W_TRAVEL     × total_passengers_in_transit
```

| Constant | Value | Meaning |
|---|---|---|
| `R_DELIVERED` | 2.0 | Reward per passenger successfully delivered |
| `C_OP` | 40.0 | Operating cost incurred per dispatch |
| `W_WAIT` | 0.2 | Penalty weight per passenger-minute spent waiting |
| `W_TRAVEL` | 0.05 | Penalty weight per passenger-minute spent in transit |

This is exposed live in the app via the reward decomposition panel, so you can see exactly which term is driving an agent's behavior at any point in the simulation.


## Tech Stack

- **Python** — core language
- **PyTorch** — DDPG / TD3 neural networks
- **Streamlit** — web app framework
- **Three.js** (via CDN, embedded HTML) — 3D scene rendering
- **Plotly / Matplotlib** — charts and heatmaps
- **NumPy** — numerical simulation

## License

Add a license of your choice (e.g. MIT) before making this repository public.
