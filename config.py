"""
config.py
All constants for the 3D visual layer (colors, timing, camera presets).
RL/environment constants (stations, capacity, reward weights) live in
simulation/tram_env.py and are NOT duplicated here — this file only
configures how things are drawn, never how the environment behaves.
"""

# ---------------------------------------------------------------------------
# Scenario windows — must match simulation/tram_env.py exactly (report Sec 2)
# ---------------------------------------------------------------------------
MORNING_PEAK_RANGE = (90, 210)    # 07:30-09:30, minutes since 06:00
EVENING_PEAK_RANGE = (630, 750)   # 16:30-18:30

SCENARIO_LABELS = {
    'morning': 'Morning Peak',
    'evening': 'Evening Peak',
    'offpeak': 'Off-Peak',
}


def get_scenario(minute):
    if MORNING_PEAK_RANGE[0] <= minute < MORNING_PEAK_RANGE[1]:
        return 'morning'
    if EVENING_PEAK_RANGE[0] <= minute < EVENING_PEAK_RANGE[1]:
        return 'evening'
    return 'offpeak'


# ---------------------------------------------------------------------------
# Day/night lighting — driven by minute-of-day (0-1080), purely cosmetic
# ---------------------------------------------------------------------------
# t_frac = minute / 1080 -> 0.0 (06:00) ... 1.0 (24:00)
SKY_KEYFRAMES = [
    # (t_frac, sky_hex, sun_hex, ambient_intensity, sun_intensity)
    (0.00, '0x9fc7e8', '0xfff2cc', 0.55, 0.9),   # 06:00 dawn
    (0.20, '0x7fc3f0', '0xffffff', 0.65, 1.1),   # 08:00 morning
    (0.45, '0x6fb8ef', '0xffffff', 0.70, 1.2),   # 11:30 midday
    (0.60, '0x5aa0e0', '0xffe9b3', 0.60, 1.0),   # 14:30 afternoon
    (0.75, '0xdd8a52', '0xffb066', 0.45, 0.8),   # 18:00 sunset
    (0.85, '0x2b2f4a', '0x8fa5ff', 0.25, 0.4),   # 19:30 dusk
    (1.00, '0x0b0e21', '0x3355aa', 0.15, 0.2),   # 24:00 night
]

STATION_COLOR   = '0x2ecc71'
DEPOT_COLOR     = '0x4e9af1'
TERMINUS_COLOR  = '0xe74c3c'
TRAM_BODY_COLOR = '0xf4c542'
TRACK_COLOR     = '0x555555'
GROUND_COLOR    = '0x3a3f2e'

MAX_FIGURES_PER_STATION = 8   # visual cap; overflow shown as "+N" label

# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------
CAMERA_MODES = ['Top View', 'Tracking', 'Driver View', 'Station View', 'Free Orbit']
DEFAULT_CAMERA = 'Tracking'

# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
WEATHER_OPTIONS = ['Clear', 'Rain', 'Fog']

# ---------------------------------------------------------------------------
# Scene layout (world units)
# ---------------------------------------------------------------------------
STATION_SPACING = 12          # distance between adjacent stations along X
SCENE_HEIGHT_PX = 520         # embedded iframe height

# Animation tween duration should roughly match app.py's frame delay
# so tram/door motion appears continuous across Streamlit reruns.
TWEEN_MS = 450