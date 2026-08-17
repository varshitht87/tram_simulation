"""
simulation/scenario_manager.py
Named scenario presets that override TramNetworkEnv's clock/queues for
demo purposes. Pure state override — reward function, action space,
and transition dynamics in tram_env.py are never touched. Demand-window
classification (Morning/Evening/Off-Peak) reuses config.get_scenario(),
the same boundaries used everywhere else in the project.

Queue values are hand-picked to be internally consistent with
tram_env.py's smoothed demand ramp (see TramNetworkEnv.DEMAND_RAMP_MINUTES):
presets that sit just inside/outside a ramp window use queue sizes that
look like they arrived gradually rather than snapping to full peak volume.
"""

SCENARIOS = {
    'Early Morning Ramp-Up (07:15)': {
        'minute': 75, 'queues': [12, 5, 3, 2], 'elapsed': 8,
        'period_label': 'Pre-Peak ramp (demand rising toward 07:30 peak)',
    },
    'Morning Peak (08:00)': {
        'minute': 120, 'queues': [65, 28, 15, 10], 'elapsed': 20,
        'period_label': 'Morning Peak (07:30-09:30)',
    },
    'Morning Peak – Uneven Demand (08:45)': {
        'minute': 165, 'queues': [75, 5, 2, 1], 'elapsed': 10,
        'period_label': 'Morning Peak (07:30-09:30)',
    },
    'Midday Off-Peak (13:30)': {
        'minute': 450, 'queues': [5, 3, 2, 1], 'elapsed': 5,
        'period_label': 'Off-Peak (09:31-16:29)',
    },
    'Evening Peak (17:15)': {
        'minute': 675, 'queues': [40, 35, 22, 18], 'elapsed': 15,
        'period_label': 'Evening Peak (16:30-18:30)',
    },
    'Evening Peak – After Long Delay (18:00)': {
        'minute': 720, 'queues': [55, 48, 40, 30], 'elapsed': 35,
        'period_label': 'Evening Peak (16:30-18:30)',
    },
    'Late Evening Wind-Down (19:15)': {
        'minute': 795, 'queues': [10, 8, 5, 3], 'elapsed': 12,
        'period_label': 'Post-Peak ramp (demand falling after 18:30 peak)',
    },
    'Late Night (23:00)': {
        'minute': 1020, 'queues': [2, 1, 1, 0], 'elapsed': 5,
        'period_label': 'Off-Peak (low overnight demand)',
    },
    'Full Operating Day (06:00 start)': {
        'minute': 0, 'queues': [0, 0, 0, 0], 'elapsed': 0,
        'period_label': 'Full Day — dynamic transitions',
        'full_reset': True,
    },
}

SCENARIO_NAMES = list(SCENARIOS.keys())


def apply_scenario(env, name):
    """Mutate env in place to match a named preset. Returns the spec dict
    (useful for displaying period_label in the dashboard)."""
    spec = SCENARIOS[name]
    if spec.get('full_reset'):
        env.reset()
        return spec

    env.minute = spec['minute']
    env.elapsed_dispatch = spec['elapsed']
    for i, q in enumerate(spec['queues']):
        env.queues[i] = q
    env.trams = []  # clean track for a clear preset demo
    env.step_count = env.minute // env.STEP_DURATION
    return spec