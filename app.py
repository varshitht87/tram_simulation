"""
app.py
Entry point. Wires the UNCHANGED environment (tram_env.py) and UNCHANGED
agents (agents.py) into the 3D scene (ui/scene3d.py), control-centre
dashboard (ui/dashboard.py), camera/weather controls (ui/camera.py), and
named scenario presets (simulation/scenario_manager.py).

No RL/environment logic lives in this file — only orchestration and
Streamlit session-state bookkeeping for the live 3D visualisation.
"""

import random
import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import config as cfg
from tram_env import TramNetworkEnv
from agents import (
    make_agent, QLearningAgent, MCQLearningAgent,
    DDPGAgent, TD3Agent, ACTION_NAMES,
)
from ui import scene3d, dashboard, camera, reward_panel, qtable_heatmap
from simulation.scenario_manager import SCENARIO_NAMES, apply_scenario
from simulation.reward_decomposer import default_weights, raw_components, weighted_terms

st.set_page_config(page_title="Tram Dispatch RL — 3D Simulator", layout="wide")

AGENT_NAMES = ['Baseline', 'Q-Learning', 'MC-Q-Learning', 'DDPG', 'TD3']
COMPARISON_BASE_SEED = 1000  # matches report's fixed test-seed convention (1000-1009)
LIVE_OPTION = '— Live Simulation (no preset) —'
SCENARIO_OPTIONS = [LIVE_OPTION] + SCENARIO_NAMES
TRAIN_EPISODES_TARGET = 150  # training-plan default: train 150 eps, then auto-switch to Test


def speed_to_params(speed):
    """Map a single 1x-50x 'Simulation Speed' dial to the (steps_per_frame,
    frame_delay) pair the animation loop actually needs. Two-phase curve,
    chosen so the dial's own endpoints reproduce the two behaviours the
    sliders it replaces used to document by hand:
      1x  -> steps_per_frame=1,   frame_delay=2.5s  (full dwell animation)
      50x -> steps_per_frame=216, frame_delay=0.0s  (1 episode/rerun, headless)
    Phase 1 (1x-10x): keep steps_per_frame=1, shorten the pause 2.5s -> 0.1s
    (playback gets snappier but each decision step still animates).
    Phase 2 (10x-50x): pause is already negligible, so from here on speed
    comes from batching more decision-steps into each redraw, 1 -> 216.
    """
    speed = max(1, min(50, speed))
    if speed <= 10:
        frac = (speed - 1) / 9.0
        steps_per_frame = 1
        frame_delay = 2.5 * (1 - frac) + 0.1 * frac
    else:
        frac = (speed - 10) / 40.0
        steps_per_frame = round(1 + frac * (216 - 1))
        frame_delay = 0.1 * (1 - frac)
    return steps_per_frame, frame_delay


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def blank_history():
    return {
        'reward_history': [], 'delivered_history': [], 'dispatch_history': [],
        'completed_episodes': 0, 'success_count': 0, 'episode': 1,
        'episode_log': [],
    }


def ensure_global_state():
    if 'agents_cache' in st.session_state:
        return
    st.session_state.agents_cache = {}
    st.session_state.history_cache = {name: blank_history() for name in AGENT_NAMES}
    st.session_state.active_agent_name = 'Q-Learning'
    st.session_state.env = TramNetworkEnv()
    st.session_state.running = False
    st.session_state.step_in_episode = 0
    st.session_state.total_reward_ep = 0.0
    st.session_state.delivered_ep = 0
    st.session_state.dispatch_ep = 0
    st.session_state.wait_time_accum = 0.0
    st.session_state.last_action = None
    st.session_state.last_reward = 0.0
    st.session_state.last_aux = {}
    st.session_state.prev_queues = [0, 0, 0, 0]
    st.session_state.prev_trams = {}
    st.session_state.raw_history = []
    st.session_state.weighted_history = []
    st.session_state.next_tram_id = 1
    st.session_state.max_episodes = 20
    st.session_state.comparison_results = None
    st.session_state.selected_scenario = LIVE_OPTION
    st.session_state.speed_choice = 1


ensure_global_state()


def get_or_create_agent(name):
    if name not in st.session_state.agents_cache:
        if name == 'MC-Q-Learning':
            with st.spinner("Running Monte Carlo warm-start (100 episodes)..."):
                st.session_state.agents_cache[name] = make_agent(name)
        else:
            st.session_state.agents_cache[name] = make_agent(name)
    return st.session_state.agents_cache[name]


def reset_episode_bookkeeping():
    st.session_state.step_in_episode = 0
    st.session_state.total_reward_ep = 0.0
    st.session_state.delivered_ep = 0
    st.session_state.dispatch_ep = 0
    st.session_state.wait_time_accum = 0.0
    st.session_state.prev_queues = [int(q) for q in st.session_state.env.queues]
    st.session_state.prev_trams = {}
    st.session_state.raw_history = []
    st.session_state.weighted_history = []


def ensure_tram_ids(env):
    """Attach a stable 'id' to each tram dict so the 3D scene can track
    the same mesh across frames. Rendering aid only — env behaviour
    (reward, state, transitions) is untouched."""
    for t in env.trams:
        if 'id' not in t:
            t['id'] = st.session_state.next_tram_id
            st.session_state.next_tram_id += 1


def snapshot_trams(env):
    return {t['id']: {'position': int(t['position'])} for t in env.trams if 'id' in t}


def describe_state(agent_name, env, aux):
    if agent_name in ('Q-Learning', 'MC-Q-Learning'):
        s_idx = aux.get('state_idx')
        q_total = sum(env.queues)
        q_bin = 'Low(<10)' if q_total < 10 else ('Med(10-30)' if q_total < 30 else 'High(>=30)')
        scenario = cfg.get_scenario(env.minute)
        e = env.elapsed_dispatch
        e_bin = 'Short(<10)' if e < 10 else ('Med(10-20)' if e < 20 else 'Long(>=20)')
        return f"idx={s_idx} | Queue:{q_bin} Demand:{cfg.SCENARIO_LABELS[scenario]} Elapsed:{e_bin}"
    state = aux.get('state')
    if state is None:
        return "-"
    return "[" + ", ".join(f"{v:.2f}" for v in state) + "]"


def describe_policy(agent, agent_name, training):
    if agent_name == 'Baseline':
        return "Fixed-time: dispatch every 3 steps (15 min)"
    if agent_name in ('Q-Learning', 'MC-Q-Learning'):
        return f"ε-greedy (ε={agent.epsilon:.2f})" if training else "Greedy (test, ε=0)"
    return f"Actor + Gaussian noise (σ={agent.sigma:.2f})" if training else "Actor (deterministic, test)"


# ---------------------------------------------------------------------------
# Single decision-step
# ---------------------------------------------------------------------------
def do_step():
    env, agent = st.session_state.env, st.session_state.agent
    agent_name = st.session_state.active_agent_name

    # Recomputed fresh on every single step (not once per batch) so that a
    # high Simulation Speed — which can pack up to 216 steps, i.e. a whole
    # episode, into one rerun — can never let a step "leak" into training
    # after the 150-episode budget is spent just because it happened to
    # land in the same batch as the episode-150 boundary.
    hist = st.session_state.history_cache[agent_name]
    training = agent.is_trainable and (hist['completed_episodes'] < TRAIN_EPISODES_TARGET)

    action, aux = agent.select_action(env, training=training)
    ns, reward, done, info = env.step(action)
    st.session_state.last_aux = aux

    if training:
        if isinstance(agent, (QLearningAgent, MCQLearningAgent)):
            next_s_idx = env._get_discrete_state()
            agent.observe(aux['state_idx'], action, reward, next_s_idx, done)
        elif isinstance(agent, (DDPGAgent, TD3Agent)):
            agent.observe(aux['state'], aux['a_cont'], reward, ns, done)

    raw = raw_components(env, action, info)
    st.session_state.raw_history.append(raw)
    st.session_state.weighted_history.append(weighted_terms(raw, default_weights(env)))

    st.session_state.last_action      = action
    st.session_state.last_reward      = reward
    st.session_state.total_reward_ep += reward
    st.session_state.delivered_ep    += info['n_delivered']
    st.session_state.dispatch_ep     += int(info['dispatched'])
    st.session_state.wait_time_accum += info['total_wait']
    st.session_state.step_in_episode += 1

    if done:
        agent.end_episode()
        hist = st.session_state.history_cache[agent_name]
        hist['reward_history'].append(st.session_state.total_reward_ep)
        hist['delivered_history'].append(st.session_state.delivered_ep)
        hist['dispatch_history'].append(st.session_state.dispatch_ep)
        hist['completed_episodes'] += 1
        if st.session_state.total_reward_ep > 0:
            hist['success_count'] += 1
        hist['episode'] += 1

        # Training-controller log line — fixed format, one entry per completed
        # episode, cumulative averages over all episodes since the last Reset.
        ep_num = hist['completed_episodes']
        ep_mode = 'TRAIN' if ep_num <= TRAIN_EPISODES_TARGET else 'TEST'
        avg_dispatches = sum(hist['dispatch_history']) / len(hist['dispatch_history'])
        avg_rewards = sum(hist['reward_history']) / len(hist['reward_history'])
        total_delivered = sum(hist['delivered_history'])
        hist['episode_log'].append(
            f"Episode: {ep_num} | Mode: {ep_mode}\n"
            f"Avg Dispatches: {avg_dispatches:.1f}\n"
            f"Avg Rewards: {avg_rewards:.2f}\n"
            f"Passengers Delivered (total): {total_delivered}"
        )
        if len(hist['episode_log']) > 300:  # bounded buffer for long headless runs
            hist['episode_log'] = hist['episode_log'][-300:]

        env.reset()
        agent.reset_episode()
        reset_episode_bookkeeping()

        if hist['episode'] > st.session_state.max_episodes:
            st.session_state.running = False


# ---------------------------------------------------------------------------
# Comparison mode — same demand seed(s) across all controllers
# ---------------------------------------------------------------------------
def run_comparison(n_seeds):
    rows = []
    for name in AGENT_NAMES:
        agent = get_or_create_agent(name)
        rewards, delivered, dispatches = [], [], []
        for seed in range(COMPARISON_BASE_SEED, COMPARISON_BASE_SEED + n_seeds):
            random.seed(seed)
            np.random.seed(seed)
            env_tmp = TramNetworkEnv()
            agent.reset_episode()
            ep_r = ep_d = ep_disp = 0
            for _ in range(TramNetworkEnv.TOTAL_STEPS):
                a, _ = agent.select_action(env_tmp, training=False)
                _, r, done, info = env_tmp.step(a)
                ep_r += r
                ep_d += info['n_delivered']
                ep_disp += int(info['dispatched'])
                if done:
                    break
            rewards.append(ep_r); delivered.append(ep_d); dispatches.append(ep_disp)
        row = {
            'Agent': name,
            'Avg Reward': round(sum(rewards) / n_seeds, 1),
            'Avg Delivered': round(sum(delivered) / n_seeds, 1),
            'Avg Dispatches': round(sum(dispatches) / n_seeds, 1),
        }
        if n_seeds > 1:
            mean_r = sum(rewards) / n_seeds
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / n_seeds) ** 0.5
            row['Std Reward'] = round(std_r, 1)
            row['Min Reward'] = round(min(rewards), 1)
            row['Max Reward'] = round(max(rewards), 1)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")

agent_choice = st.sidebar.selectbox(
    "Agent", AGENT_NAMES, index=AGENT_NAMES.index(st.session_state.active_agent_name)
)

st.sidebar.subheader("Training Controller")
st.sidebar.caption(
    f"Fixed schedule (per agent, since last Reset): episodes "
    f"1–{TRAIN_EPISODES_TARGET} run in **TRAIN** (learning enabled); "
    f"episode {TRAIN_EPISODES_TARGET + 1} onward runs in **TEST** "
    f"(parameters frozen) permanently, until Reset. Not user-adjustable — "
    f"this is a controller invariant, not a preference."
)
max_eps = st.sidebar.number_input(
    "Episodes to run (this run)", 1, 2000, st.session_state.max_episodes, step=1,
    help=f"Total episode budget since the last Reset. Set above "
         f"{TRAIN_EPISODES_TARGET} to actually reach the testing phase — "
         f"e.g. 300 = {TRAIN_EPISODES_TARGET} train + 150 test episodes."
)

st.sidebar.subheader("Playback")
speed_choice = st.sidebar.slider(
    "Simulation Speed", 1, 50, st.session_state.speed_choice, format="%dx",
    help="1x = full dwell animation, one decision step at a time (~2.5s/step). "
         "50x = one full 216-step episode per rerun with no pause (headless "
         "training speed). In between, the pause shortens first, then extra "
         "decision steps get batched into each redraw."
)
st.session_state.speed_choice = speed_choice
steps_per_frame, frame_delay = speed_to_params(speed_choice)

camera_mode, weather = camera.render_camera_and_weather_controls()

st.sidebar.subheader(" Scenario Preset")
scenario_choice = st.sidebar.selectbox("Jump to scenario", SCENARIO_OPTIONS, index=0)

col_a, col_b, col_c = st.sidebar.columns(3)
start_clicked = col_a.button("Start")
pause_clicked = col_b.button("Pause")
reset_clicked = col_c.button("Reset")

# Agent switch — keep learned progress, just start a fresh episode
if agent_choice != st.session_state.active_agent_name:
    st.session_state.active_agent_name = agent_choice
    st.session_state.env.reset()
    reset_episode_bookkeeping()
    st.session_state.running = False

# Scenario preset — apply only on change, never every rerun
if scenario_choice != st.session_state.selected_scenario:
    st.session_state.selected_scenario = scenario_choice
    if scenario_choice != LIVE_OPTION:
        spec = apply_scenario(st.session_state.env, scenario_choice)
        reset_episode_bookkeeping()
        st.toast(f"Applied: {scenario_choice} — {spec['period_label']}")

if reset_clicked:
    st.session_state.agents_cache.pop(st.session_state.active_agent_name, None)
    st.session_state.history_cache[st.session_state.active_agent_name] = blank_history()
    st.session_state.env = TramNetworkEnv()
    reset_episode_bookkeeping()
    st.session_state.running = False

st.session_state.max_episodes = max_eps
if start_clicked:
    st.session_state.running = True
if pause_clicked:
    st.session_state.running = False

st.session_state.agent = get_or_create_agent(st.session_state.active_agent_name)
agent = st.session_state.agent
env = st.session_state.env
hist = st.session_state.history_cache[st.session_state.active_agent_name]

# Effective mode — deterministic function of completed-episode count for the
# CURRENT agent since its last Reset. No manual override: episodes 1-150
# always TRAIN (if the algorithm is trainable), 151+ always TEST, permanently,
# until Reset re-zeros hist['completed_episodes']. This also gates
# do_step()'s `training` flag below, so no agent.observe() / optimizer step
# can ever fire once the 150-episode budget is spent.
effective_mode = 'Train' if hist['completed_episodes'] < TRAIN_EPISODES_TARGET else 'Test'
status = 'TRAIN — learning enabled' if effective_mode == 'Train' else 'TEST — parameters frozen'
st.sidebar.info(
    f"**Mode: {status}**\n\n"
    f"Episode {hist['completed_episodes']} completed "
    f"({min(hist['completed_episodes'], TRAIN_EPISODES_TARGET)}/{TRAIN_EPISODES_TARGET} train)"
)


# ---------------------------------------------------------------------------
# Advance simulation (one frame's worth of steps)
# ---------------------------------------------------------------------------
prev_queues_snapshot = st.session_state.prev_queues
prev_trams_snapshot = st.session_state.prev_trams

if st.session_state.running:
    for _ in range(steps_per_frame):
        if not st.session_state.running:
            break
        do_step()

ensure_tram_ids(env)

tram_states = [
    {'id': t['id'], 'position': int(t['position']), 'occupancy': sum(t['passengers'].values())}
    for t in env.trams
]

decision = None
if st.session_state.last_action is not None:
    decision = dashboard.build_decision_reason(
        st.session_state.active_agent_name, st.session_state.last_action, st.session_state.last_aux
    )

scene_state = {
    'minute': env.minute,
    'queues': [int(q) for q in env.queues],
    'prev_queues': prev_queues_snapshot,
    'trams': tram_states,
    'prev_trams': prev_trams_snapshot,
    'dispatch_event': st.session_state.dispatch_ep > 0 and st.session_state.last_action == 1,
    'decision': decision,
}
st.session_state.prev_queues = [int(q) for q in env.queues]
st.session_state.prev_trams = snapshot_trams(env)


# Main layout

st.title("Intelligent Tram Dispatching — 3D RL Simulator")
st.caption(
    "Same environment, reward function, and algorithms as the report — "
    "upgraded visual layer only."
)

scene_html = scene3d.build_scene_html(
    scene_state,
    camera_mode=camera_mode,
    weather=weather
)

scenario = cfg.SCENARIO_LABELS[cfg.get_scenario(env.minute)]
avg_reward = (sum(hist['reward_history']) / len(hist['reward_history'])) if hist['reward_history'] else 0.0
delivered_total = st.session_state.delivered_ep
waiting_total = sum(env.queues)
avg_wait = (st.session_state.wait_time_accum / st.session_state.step_in_episode) if st.session_state.step_in_episode else 0.0
efficiency = delivered_total / (delivered_total + waiting_total + 1) * 100
congestion_index = min(200.0, (waiting_total / TramNetworkEnv.TRAM_CAPACITY) * 100)
success_rate = (hist['success_count'] / hist['completed_episodes'] * 100) if hist['completed_episodes'] else 0.0
sim_speed = "Paused" if not st.session_state.running else f"{speed_choice}x"
op_cost_episode = TramNetworkEnv.C_OP * st.session_state.dispatch_ep
total_dispatches_session = sum(hist['dispatch_history']) + st.session_state.dispatch_ep
op_cost_session = TramNetworkEnv.C_OP * total_dispatches_session
avg_dispatch_count = (sum(hist['dispatch_history']) / len(hist['dispatch_history'])) if hist['dispatch_history'] else 0.0
session_total_reward = sum(hist['reward_history']) + st.session_state.total_reward_ep

# Tabbed layout — groups the simulator's ~9 previously stacked sections into
# 4 focused views so each screen only shows what's relevant to that task
# (watch the sim, inspect reward/policy internals, review training curves,
# or compare controllers) instead of one long scroll of everything at once.
tab_live, tab_diag, tab_history, tab_compare = st.tabs(
    ["🎬 Live Simulation", "🧠 Reward & Policy Diagnostics", "📈 Training History", "⚖️ Compare Algorithms"]
)

with tab_live:
    components.html(scene_html, height=cfg.SCENE_HEIGHT_PX + 10, scrolling=False)
    st.divider()

    st.subheader("Training Controller Log")
    if hist['episode_log']:
        st.code(hist['episode_log'][-1], language=None)
        with st.expander(f"Full log — {len(hist['episode_log'])} completed episode(s) this run (newest first)"):
            st.code("\n\n".join(reversed(hist['episode_log'])), language=None)
    else:
        st.caption("No completed episodes yet this run — press Start.")

    st.divider()
    dashboard.render_dashboard({
        'minute': env.minute, 'scenario': scenario, 'weather': weather,
        'agent_name': st.session_state.active_agent_name,
        'episode': min(hist['episode'], st.session_state.max_episodes),
        'max_episodes': st.session_state.max_episodes,
        'current_reward': st.session_state.last_reward,
        'total_reward': st.session_state.total_reward_ep,
        'avg_reward': avg_reward,
        'passengers_waiting': waiting_total,
        'passengers_delivered': delivered_total,
        'avg_wait_time': avg_wait,
        'dispatch_count': st.session_state.dispatch_ep,
        'queues': env.queues,
        'tram_occupancies': [sum(t['passengers'].values()) for t in env.trams],
        'active_trams': len(env.trams),
        'efficiency': efficiency,
        'sim_speed': sim_speed,
        'congestion_index': congestion_index,
        'success_rate': success_rate,
        'op_cost_episode': op_cost_episode,
        'op_cost_session': op_cost_session,
        'avg_dispatch_count': avg_dispatch_count,
        'session_total_reward': session_total_reward,
    })
    st.divider()
    dashboard.render_rl_panel({
        'state_repr': describe_state(st.session_state.active_agent_name, env, st.session_state.last_aux),
        'action_name': ACTION_NAMES.get(st.session_state.last_action, '-'),
        'reward': st.session_state.last_reward,
        'q_values': st.session_state.last_aux.get('q_values') if isinstance(agent, (QLearningAgent, MCQLearningAgent)) else None,
        'a_cont': st.session_state.last_aux.get('a_cont') if isinstance(agent, (DDPGAgent, TD3Agent)) else None,
        'policy_desc': describe_policy(agent, st.session_state.active_agent_name,
                                        (effective_mode == 'Train') and agent.is_trainable),
        'episode': min(hist['episode'], st.session_state.max_episodes),
        'demand_context': scenario,
        'elapsed_time': env.elapsed_dispatch,
        'queue_length': waiting_total,
        'tram_occupancy': sum(sum(t['passengers'].values()) for t in env.trams),
        'active_trams': len(env.trams),
    })

with tab_diag:
    reward_panel.render_reward_decomposition(st.session_state.weighted_history)
    reward_panel.render_reward_sandbox(st.session_state.raw_history, default_weights(env))
    st.divider()
    current_state_idx = (
        st.session_state.last_aux.get('state_idx')
        if isinstance(agent, (QLearningAgent, MCQLearningAgent)) else None
    )
    qtable_heatmap.render_qtable_heatmap(agent, current_state_idx, st.session_state.active_agent_name)

with tab_history:
    st.caption(
        "One section per algorithm, each from its own independent "
        "history_cache entry — nothing here is shared or averaged across "
        "algorithms. A section appears once that algorithm has completed "
        "at least one episode this run (since the last Reset)."
    )
    any_trained = False
    for name in AGENT_NAMES:
        h = st.session_state.history_cache[name]
        if not h['reward_history']:
            continue
        any_trained = True
        active_tag = "  ·  *(currently selected)*" if name == st.session_state.active_agent_name else ""
        st.markdown(f"#### {name} — {h['completed_episodes']} episode(s) completed{active_tag}")
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            st.caption("Total Reward per Episode")
            st.line_chart(h['reward_history'])
        with hc2:
            st.caption("Passengers Delivered per Episode")
            st.line_chart(h['delivered_history'])
        with hc3:
            st.caption("Dispatches per Episode")
            st.line_chart(h['dispatch_history'])
        st.divider()

    if not any_trained:
        st.info("No completed episodes yet for any algorithm this run — select an agent in the sidebar and press Start.")

with tab_compare:
    st.subheader("Algorithm Comparison Mode")
    n_seeds = st.slider("Test seeds to average (from seed 1000)", 1, 10, 3)
    if st.button("Run Comparison — same demand seed(s) for every controller"):
        with st.spinner(f"Running {len(AGENT_NAMES)} controllers x {n_seeds} seed(s)..."):
            st.session_state.comparison_results = run_comparison(n_seeds)
    if st.session_state.comparison_results:
        dashboard.render_comparison_table(st.session_state.comparison_results)
        dashboard.render_comparison_chart(st.session_state.comparison_results)


# ---------------------------------------------------------------------------
# Animation driver — one frame per script rerun (keeps Pause responsive)
# ---------------------------------------------------------------------------
if st.session_state.running:
    time.sleep(frame_delay)
    st.rerun()