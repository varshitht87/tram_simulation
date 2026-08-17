"""
simulation/reward_decomposer.py
Splits the scalar reward returned by TramNetworkEnv.step() back into the
4 named terms from the report's reward function:

    R_t = R_delivered*N_delivered - C_op*A_t - W_wait*sum(Q_i) - W_travel*sum(P_j)

tram_env.py only returns the SUM (by design, matching the report exactly).
This module reconstructs each term from values already public on env/info
— it reads env.R_DELIVERED/C_OP/W_WAIT/W_TRAVEL directly, so if those
constants are ever tuned, this stays in sync automatically. No RL logic
is duplicated or altered; this is a read-only recomputation for display.
"""


def default_weights(env):
    """Pull the report's actual weights straight from the env class."""
    return {
        'R_delivered': env.R_DELIVERED,
        'C_op': env.C_OP,
        'W_wait': env.W_WAIT,
        'W_travel': env.W_TRAVEL,
    }


def raw_components(env, action, info):
    """
    Un-weighted quantities for one step, captured right after env.step():
      n_delivered        - passengers delivered this step
      dispatched         - 1 if a tram was dispatched this step, else 0
      total_wait         - sum of station queues after this step
      travel_occupancy   - sum of in-tram passengers after this step
    These are exactly the quantities tram_env.py's own reward line uses.
    """
    travel_occupancy = sum(sum(t['passengers'].values()) for t in env.trams)
    return {
        'n_delivered': info['n_delivered'],
        'dispatched': int(info['dispatched']),
        'total_wait': info['total_wait'],
        'travel_occupancy': travel_occupancy,
    }


def weighted_terms(raw, weights):
    """Apply weights to raw components -> the 4 named reward terms."""
    return {
        'delivered_term': weights['R_delivered'] * raw['n_delivered'],
        'dispatch_term': -weights['C_op'] * raw['dispatched'],
        'wait_term': -weights['W_wait'] * raw['total_wait'],
        'travel_term': -weights['W_travel'] * raw['travel_occupancy'],
    }


def weighted_reward(raw, weights):
    terms = weighted_terms(raw, weights)
    return sum(terms.values())