"""
ui/dashboard.py
Renders the transport control-centre metrics panel and the live
agent-decision panel. Takes plain dicts/values from app.py — never
touches TramNetworkEnv or the agents directly.

v2 additions: Simulation Speed, Average Occupancy, Congestion Index,
Success Rate, and build_decision_reason() for the scene's WAIT/DISPATCH
banner. Congestion Index and System Efficiency are cosmetic UI-derived
metrics (not part of the report's reward function) — flagged inline.
"""

import streamlit as st
import plotly.graph_objects as go


def _minute_to_clock(minute):
    hh, mm = divmod(int(minute), 60)
    return f"{hh + 6:02d}:{mm:02d}"


def render_dashboard(ctx):
    """
    ctx keys: minute, scenario, weather, agent_name, episode, max_episodes,
    current_reward, total_reward, avg_reward, passengers_waiting,
    passengers_delivered, avg_wait_time, dispatch_count, queues (list[4]),
    tram_occupancies (list[int]), active_trams, efficiency, sim_speed (str),
    congestion_index, success_rate

    v3 layout: a slim always-visible status strip + two day/episode progress
    bars up top, the metrics that matter moment-to-moment (reward, queue,
    fleet) grouped right below, and everything session/cost-related tucked
    behind an expander so the tab isn't a 24-metric wall on first glance.
    """
    st.subheader("Transport Control Centre")

    top = st.columns(5)
    top[0].metric("Simulation Time", _minute_to_clock(ctx['minute']))
    top[1].metric("Scenario", ctx['scenario'])
    top[2].metric("Weather", ctx['weather'])
    top[3].metric("Agent", ctx['agent_name'])
    top[4].metric("Simulation Speed", ctx['sim_speed'])

    day_frac = max(0.0, min(1.0, ctx['minute'] / 1080))
    st.progress(day_frac, text=f"Service day — {_minute_to_clock(ctx['minute'])} of 24:00 ({day_frac * 100:.0f}%)")

    ep_frac = max(0.0, min(1.0, ctx['episode'] / ctx['max_episodes'])) if ctx['max_episodes'] else 0.0
    st.progress(ep_frac, text=f"Episode {ctx['episode']} / {ctx['max_episodes']}")

    st.markdown("**Reward & Passengers**")
    r1 = st.columns(4)
    r1[0].metric("Current Reward", f"{ctx['current_reward']:.2f}")
    reward_delta = (ctx['total_reward'] - ctx['avg_reward']) if ctx['avg_reward'] else None
    r1[1].metric(
        "Total Reward (Episode)", f"{ctx['total_reward']:.1f}",
        delta=(f"{reward_delta:+.1f} vs avg" if reward_delta is not None else None),
    )
    r1[2].metric("Passengers Waiting", int(ctx['passengers_waiting']))
    r1[3].metric("Passengers Delivered", int(ctx['passengers_delivered']))

    st.markdown("**Fleet & Congestion**")
    r2 = st.columns(4)
    r2[0].metric("Active Trams", int(ctx['active_trams']))
    r2[1].metric("Tram Occupancy (total)", int(sum(ctx['tram_occupancies'])))
    r2[2].metric("Congestion Index", f"{ctx['congestion_index']:.0f}%")
    r2[3].metric("System Efficiency", f"{ctx['efficiency']:.0f}%")

    with st.expander("Per-station queue, dispatch cost & session totals"):
        st.caption("Per-station queue")
        cols = st.columns(4)
        for i, q in enumerate(ctx['queues']):
            cols[i].metric(f"Station {i}", int(q))

        st.caption("This episode")
        r3 = st.columns(4)
        r3[0].metric("Avg Waiting Time (steps)", f"{ctx['avg_wait_time']:.1f}")
        r3[1].metric("Dispatch Count", int(ctx['dispatch_count']))
        r3[2].metric("Operational Cost", f"{ctx['op_cost_episode']:.0f}")
        r3[3].metric("Success Rate (session)", f"{ctx['success_rate']:.0f}%")

        st.caption("Session totals")
        r4 = st.columns(4)
        r4[0].metric("Average Reward (all episodes)", f"{ctx['avg_reward']:.1f}")
        r4[1].metric("Total Reward (Session)", f"{ctx['session_total_reward']:.1f}")
        r4[2].metric("Operational Cost (Session)", f"{ctx['op_cost_session']:.0f}")
        r4[3].metric("Avg Dispatch Count (Session)", f"{ctx['avg_dispatch_count']:.1f}")

    st.caption(
        "System Efficiency, Congestion Index and Simulation Speed are cosmetic "
        "dashboard metrics for the visual layer — they are not part of the "
        "report's reward function and have no effect on training. Operational "
        "Cost = C_op × dispatch count, using the report's own C_op=40 constant."
    )


def render_rl_panel(ctx):
    """
    ctx keys: state_repr, action_name, reward, q_values (list[2] or None),
    a_cont (float or None), policy_desc, episode, demand_context,
    elapsed_time, queue_length, tram_occupancy, active_trams
    """
    st.subheader(" Live Agent Decision")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Current State**\n\n`{ctx['state_repr']}`")
    c2.markdown(f"**Action**\n\n### {ctx['action_name']}")
    c3.markdown(f"**Reward**\n\n### {ctx['reward']:.2f}")

    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**Q-value / Actor Signal**")
        if ctx.get('q_values') is not None:
            st.write(f"Wait: {ctx['q_values'][0]:.2f}  |  Dispatch: {ctx['q_values'][1]:.2f}")
        elif ctx.get('a_cont') is not None:
            st.write(f"Actor output a = {ctx['a_cont']:.2f}  (Dispatch if a > 0)")
        else:
            st.write("N/A (fixed-time controller)")
    with c5:
        st.markdown("**Current Policy**")
        st.write(ctx['policy_desc'])

    r = st.columns(5)
    r[0].metric("Episode", ctx['episode'])
    r[1].metric("Demand Context", ctx['demand_context'])
    r[2].metric("Dispatch Timer", f"{ctx['elapsed_time']} min")
    r[3].metric("Queue Length", ctx['queue_length'])
    r[4].metric("Active Trams", ctx['active_trams'])


def render_comparison_table(rows):
    """rows: list of dicts, e.g. Agent/Avg Reward/Avg Delivered/Avg Dispatches
    (optionally also Std Reward/Min Reward/Max Reward — shown as extra columns
    automatically if present, no signature change needed)."""
    st.subheader(" Algorithm Comparison (identical demand seed(s))")
    st.table(rows)


def render_comparison_chart(rows):
    """rows: list of dicts with 'Agent', 'Avg Reward', 'Std Reward' (and
    optionally 'Min Reward'/'Max Reward'). Renders a bar chart with error
    bars so the comparison shows spread across seeds, not just a point
    estimate — only called when n_seeds > 1 (std is undefined for n=1)."""
    if not rows or 'Std Reward' not in rows[0]:
        return
    st.markdown("**Avg Reward with seed-to-seed variation (± 1 std dev)**")
    agents = [r['Agent'] for r in rows]
    means = [r['Avg Reward'] for r in rows]
    stds = [r['Std Reward'] for r in rows]

    fig = go.Figure(data=go.Bar(
        x=agents, y=means,
        error_y=dict(type='data', array=stds, visible=True, color='#333333'),
        marker_color='#4e9af1',
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Avg Reward", template='plotly_white',
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Error bars show ±1 standard deviation across the test seeds used. "
        "Overlapping error bars mean the difference between those agents "
        "may not be statistically meaningful at this sample size."
    )


def build_decision_reason(agent_name, action, aux):
    """
    Short human-readable justification for the WAIT/DISPATCH banner.
    Uses only what select_action() already returns in aux — no new
    RL computation, purely a text summary of the agent's own signal.
    """
    action_name = 'DISPATCH' if action == 1 else 'WAIT'

    if agent_name == 'Baseline':
        reason = "Fixed 15-min timer" if action == 1 else "Waiting for next 15-min slot"
        return {'action': action_name, 'reason': reason}

    if agent_name in ('Q-Learning', 'MC-Q-Learning'):
        q = aux.get('q_values')
        if q is not None:
            reason = f"Q(Wait)={q[0]:.1f}  Q(Dispatch)={q[1]:.1f}"
        else:
            reason = "-"
        return {'action': action_name, 'reason': reason}

    a_cont = aux.get('a_cont')
    if a_cont is not None:
        cmp = '>' if a_cont > 0 else '≤'
        reason = f"Actor output a={a_cont:.2f} ({cmp} 0)"
    else:
        reason = "-"
    return {'action': action_name, 'reason': reason}