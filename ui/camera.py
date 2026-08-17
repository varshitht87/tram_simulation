"""
ui/camera.py
Sidebar controls for camera mode and weather. Pure UI — returns the
selected strings for scene3d.build_scene_html() to consume.
"""

import streamlit as st

import config as cfg


def render_camera_and_weather_controls():
    st.sidebar.subheader("🎥 View")
    camera_mode = st.sidebar.selectbox(
        "Camera", cfg.CAMERA_MODES,
        index=cfg.CAMERA_MODES.index(cfg.DEFAULT_CAMERA)
    )
    weather = st.sidebar.selectbox("Weather", cfg.WEATHER_OPTIONS, index=0)
    st.sidebar.caption("Free Orbit: drag to rotate, scroll to zoom.")
    return camera_mode, weather