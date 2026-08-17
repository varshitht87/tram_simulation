"""
ui/reward_panel.py
Two related panels, both built on simulation/reward_decomposer.py:

1. render_reward_decomposition — cumulative contribution of each of the
   report's 4 reward terms across the current episode, using the
   report's real weights (R_DELIVERED, C_OP, W_WAIT, W_TRAVEL).

2. render_reward_sandbox — "what if the weights were different?" tool.
   Recomputes total reward for the SAME already-taken trajectory under
   user-adjustable weights. Does not retrain any agent, does not touch
   tram_env.py, and does not affect the live run or the report's results
   — purely a sensitivity-analysis display.
"""

import streamlit as st
import plotly.graph_objects as go

from simulation.reward_decomposer import weighted_terms, weighted_reward

TERM_LABELS = {
    'delivered_term': 'Delivered (+R_delivered·N)',
    'dispatch_term': 'Dispatch cost (−C_op·A)',
    'wait_term': 'Wait penalty (−W_wait·ΣQ)',
    'travel_term': 'Travel penalty (−W_travel·ΣP)',
}
TERM_COLORS = {
    'delivered_term': '#2ecc71',
    'dispatch_term': '#e74c3c',
    'wait_term': '#f39c12',
    'travel_term': '#9b59b6',
}


def render_reward_decomposition(weighted_history):
    """weighted_history: list of dicts from reward_decomposer.weighted_terms(),
    one per step of the CURRENT episode, in order."""
    st.subheader(" Live Reward Decomposition")

    if not weighted_history:
        st.caption("No steps taken yet this episode.")
        return

    steps = list(range(1, len(weighted_history) + 1))
    fig = go.Figure()
    cumulative_total = [0.0] * len(weighted_history)

    for key, label in TERM_LABELS.items():
        series = [row[key] for row in weighted_history]
        cum = []
        running = 0.0
        for v in series:
            running += v
            cum.append(running)
        for i, v in enumerate(cum):
            cumulative_total[i] += v
        fig.add_trace(go.Scatter(
            x=steps, y=cum, mode='lines', name=label,
            line=dict(color=TERM_COLORS[key], width=2),
        ))

    fig.add_trace(go.Scatter(
        x=steps, y=cumulative_total, mode='lines', name='Total Reward',
        line=dict(color='#111111', width=3, dash='dot'),
    ))

    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Decision step (this episode)", yaxis_title="Cumulative contribution",
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        template='plotly_white',
    )
    st.plotly_chart(fig, width='stretch')
    st.caption("Each line is the running sum of that reward term's contribution so far this episode.")


def render_reward_sandbox(raw_history, report_weights):
    """raw_history: list of dicts from reward_decomposer.raw_components()
    for the CURRENT episode. report_weights: dict from default_weights()."""
    st.subheader("🎛 Reward-Weight Sandbox (what-if, does not affect training)")

    if not raw_history:
        st.caption("No steps taken yet this episode.")
        return

    c1, c2, c3, c4 = st.columns(4)
    r_delivered = c1.slider("R_delivered", 0.0, 5.0, float(report_weights['R_delivered']), 0.1)
    c_op = c2.slider("C_op", 0.0, 100.0, float(report_weights['C_op']), 5.0)
    w_wait = c3.slider("W_wait", 0.0, 1.0, float(report_weights['W_wait']), 0.05)
    w_travel = c4.slider("W_travel", 0.0, 0.5, float(report_weights['W_travel']), 0.01)
    sandbox_weights = {'R_delivered': r_delivered, 'C_op': c_op, 'W_wait': w_wait, 'W_travel': w_travel}

    report_total = sum(weighted_reward(raw, report_weights) for raw in raw_history)
    sandbox_total = sum(weighted_reward(raw, sandbox_weights) for raw in raw_history)

    m1, m2 = st.columns(2)
    m1.metric("Total Reward — Report Weights", f"{report_total:.1f}")
    m2.metric("Total Reward — Sandbox Weights", f"{sandbox_total:.1f}", delta=f"{sandbox_total - report_total:+.1f}")

    report_terms_totals = {k: 0.0 for k in TERM_LABELS}
    sandbox_terms_totals = {k: 0.0 for k in TERM_LABELS}
    for raw in raw_history:
        rt = weighted_terms(raw, report_weights)
        st_ = weighted_terms(raw, sandbox_weights)
        for k in TERM_LABELS:
            report_terms_totals[k] += rt[k]
            sandbox_terms_totals[k] += st_[k]

    fig = go.Figure()
    labels = [TERM_LABELS[k] for k in TERM_LABELS]
    fig.add_trace(go.Bar(name='Report Weights', x=labels,
                          y=[report_terms_totals[k] for k in TERM_LABELS], marker_color='#4e9af1'))
    fig.add_trace(go.Bar(name='Sandbox Weights', x=labels,
                          y=[sandbox_terms_totals[k] for k in TERM_LABELS], marker_color='#f4c542'))
    fig.update_layout(
        barmode='group', height=320, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Total contribution this episode", template='plotly_white',
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Recomputes reward for the SAME trajectory (same states/actions already taken) "
        "under different weights. No agent is retrained and tram_env.py is untouched — "
        "this is a sensitivity-analysis view only."
    )