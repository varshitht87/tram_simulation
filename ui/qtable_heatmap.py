"""
ui/qtable_heatmap.py
Interactive version of the report's Fig-5 (27-state greedy policy table)
for Q-Learning / MC-Q-Learning. Same state ordering as
TramNetworkEnv._get_discrete_state() (state = q_bin*9 + d_bin*3 + e_bin),
so row i here is exactly report state i. Read-only display — never
writes to agent.q_table.
"""

import streamlit as st
import plotly.graph_objects as go

Q_BINS = ['Low(<10)', 'Med(10-30)', 'High(>=30)']
D_BINS = ['Off-Peak', 'Eve-Peak', 'Morn-Peak']
E_BINS = ['Short(<10)', 'Med(10-20)', 'Long(>=20)']

STATE_LABELS = [
    f"{q} | {d} | {e}"
    for q in Q_BINS for d in D_BINS for e in E_BINS
]


def render_qtable_heatmap(agent, current_state_idx, agent_name):
    st.subheader(" Live Q-Table (27 states) — same layout as report Fig-5")

    if not hasattr(agent, 'q_table'):
        st.caption(f"{agent_name} is not a tabular agent — no Q-table to display.")
        return

    q = agent.q_table  # shape (27, 2)
    z = q.tolist()
    text = [[f"{v:.1f}" for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=['Wait', 'Dispatch'], y=STATE_LABELS,
        text=text, texttemplate='%{text}', textfont=dict(size=10),
        colorscale='RdYlGn', colorbar=dict(title='Q-value'),
    ))

    if current_state_idx is not None:
        fig.add_shape(
            type='rect', x0=-0.5, x1=1.5,
            y0=current_state_idx - 0.5, y1=current_state_idx + 0.5,
            line=dict(color='black', width=3), fillcolor='rgba(0,0,0,0)',
        )
        fig.add_annotation(
            x=1.5, y=current_state_idx, text='◀ current state',
            showarrow=False, xanchor='left', font=dict(size=11, color='black'),
        )

    fig.update_layout(
        height=620, margin=dict(l=10, r=110, t=20, b=10),
        yaxis=dict(autorange='reversed', tickfont=dict(size=9)),
        template='plotly_white',
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Rows follow the report's 27-state index (Queue bin | Demand context | Elapsed-since-dispatch bin). "
        "The black outline marks the agent's current state; hover any cell for its exact Q-value."
    )