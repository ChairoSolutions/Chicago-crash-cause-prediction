import random
import time
from datetime import datetime
import os

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL")

if not API_URL:
    try:
        API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000/predict")
    except FileNotFoundError:
        API_URL = "http://127.0.0.1:8000/predict"

WEATHER_OPTIONS = ["CLEAR", "RAIN", "SNOW", "CLOUDY/OVERCAST", "FOG/SMOKE/HAZE"]
LIGHTING_OPTIONS = ["DAYLIGHT", "DARKNESS", "DARKNESS, LIGHTED ROAD", "DAWN", "DUSK"]
ROAD_OPTIONS = ["DRY", "WET", "ICE", "SNOW OR SLUSH"]
TRAFFICWAY_OPTIONS = ["NOT DIVIDED", "ONE-WAY", "FOUR WAY", "DIVIDED - W/MEDIAN BARRIER"]
ALIGNMENT_OPTIONS = ["STRAIGHT AND LEVEL", "STRAIGHT ON GRADE", "CURVE, LEVEL"]
TRAFFIC_CONTROL_OPTIONS = ["NO CONTROLS", "TRAFFIC SIGNAL", "STOP SIGN/FLASHER"]
DEVICE_CONDITION_OPTIONS = ["NO CONTROLS", "FUNCTIONING PROPERLY", "UNKNOWN"]
MANEUVER_OPTIONS = ["STRAIGHT AHEAD", "TURNING LEFT", "TURNING RIGHT", "CHANGING LANES"]
SEX_OPTIONS = ["M", "F", "Unknown"]


def generate_event():
    return {
        "WEATHER_CONDITION": random.choice(WEATHER_OPTIONS),
        "LIGHTING_CONDITION": random.choice(LIGHTING_OPTIONS),
        "ROADWAY_SURFACE_COND": random.choice(ROAD_OPTIONS),
        "TRAFFICWAY_TYPE": random.choice(TRAFFICWAY_OPTIONS),
        "ALIGNMENT": random.choice(ALIGNMENT_OPTIONS),
        "TRAFFIC_CONTROL_DEVICE": random.choice(TRAFFIC_CONTROL_OPTIONS),
        "DEVICE_CONDITION": random.choice(DEVICE_CONDITION_OPTIONS),
        "dominant_maneuver": random.choice(MANEUVER_OPTIONS),
        "dominant_sex": random.choice(SEX_OPTIONS),
        "num_vehicle_types": random.randint(1, 4),
        "avg_age": random.randint(18, 80),
        "CRASH_HOUR": random.randint(0, 23),
        "CRASH_DAY_OF_WEEK": random.randint(1, 7),
        "CRASH_MONTH": random.randint(1, 12),
        "POSTED_SPEED_LIMIT": random.choice([20, 25, 30, 35, 40, 45, 55]),
        "num_people": random.randint(1, 8),
    }


def request_prediction(event):
    response = requests.post(API_URL, json=event, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Live Crash Simulation", layout="wide")

st.title("Live Chicago Crash Simulation")
st.caption("Simulated crash conditions are sent to your FastAPI model server in real time.")

if "history" not in st.session_state:
    st.session_state.history = []

left, right = st.columns([1, 2])

with left:
    st.subheader("Simulation Controls")
    events_to_run = st.slider("Events to simulate", min_value=1, max_value=50, value=10)
    delay_seconds = st.slider("Seconds between events", min_value=1, max_value=10, value=3)

    run_simulation = st.button("Run Live Simulation", type="primary")
    run_one_event = st.button("Generate One Event")
    clear_history = st.button("Clear History")

if clear_history:
    st.session_state.history = []

latest_placeholder = st.empty()
table_placeholder = st.empty()
chart_placeholder = st.empty()


def add_event_to_history(event, prediction):
    predicted_cause = prediction.get(
        "predicted_crash_cause",
        prediction.get("prediction", "Unknown"),
    )

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_cause": predicted_cause,
        **event,
    }
    st.session_state.history.insert(0, row)


def render_dashboard():
    if st.session_state.history:
        latest = st.session_state.history[0]

        with latest_placeholder.container():
            st.subheader("Latest Prediction")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Predicted Cause", latest["predicted_cause"])
            metric_cols[1].metric("Weather", latest["WEATHER_CONDITION"])
            metric_cols[2].metric("Road Surface", latest["ROADWAY_SURFACE_COND"])
            metric_cols[3].metric("Crash Hour", latest["CRASH_HOUR"])

        history_df = pd.DataFrame(st.session_state.history)

        with table_placeholder.container():
            st.subheader("Prediction History")
            st.dataframe(history_df, use_container_width=True, hide_index=True)

        with chart_placeholder.container():
            st.subheader("Most Frequent Predicted Causes")
            cause_counts = history_df["predicted_cause"].value_counts()
            st.bar_chart(cause_counts)
    else:
        with latest_placeholder.container():
            st.info("Run the simulation to see live model predictions here.")


if run_one_event:
    event = generate_event()
    prediction = request_prediction(event)
    add_event_to_history(event, prediction)

if run_simulation:
    progress = st.progress(0)
    for index in range(events_to_run):
        event = generate_event()
        prediction = request_prediction(event)
        add_event_to_history(event, prediction)
        render_dashboard()
        progress.progress((index + 1) / events_to_run)
        time.sleep(delay_seconds)

render_dashboard()
