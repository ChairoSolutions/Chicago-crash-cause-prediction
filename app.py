import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Chicago Crash Cause Predictor", layout="centered")

st.title("Chicago Crash Cause Predictor")
st.write("Predict the primary contributory cause of a crash using the tuned Random Forest model.")

deployment_dir = Path("deployment_artifacts")

model = joblib.load(deployment_dir / "tuned_random_forest_model.joblib")
training_columns = joblib.load(deployment_dir / "training_columns.joblib")
label_encoder = joblib.load(deployment_dir / "label_encoder.joblib")
selected_features = joblib.load(deployment_dir / "selected_features.joblib")

WEATHER_OPTIONS = [
    'BLOWING SAND, SOIL, DIRT', 'BLOWING SNOW', 'CLEAR', 'CLOUDY/OVERCAST',
    'FOG/SMOKE/HAZE', 'FREEZING RAIN/DRIZZLE', 'OTHER', 'RAIN',
    'SEVERE CROSS WIND GATE', 'SLEET/HAIL', 'SNOW', 'UNKNOWN'
]

LIGHTING_OPTIONS = [
    'DARKNESS', 'DARKNESS, LIGHTED ROAD', 'DAWN',
    'DAYLIGHT', 'DUSK', 'UNKNOWN'
]

ROADWAY_SURFACE_OPTIONS = [
    'DRY', 'ICE', 'OTHER', 'SAND, MUD, DIRT',
    'SNOW OR SLUSH', 'UNKNOWN', 'WET'
]

TRAFFICWAY_OPTIONS = [
    'ALLEY', 'CENTER TURN LANE', 'DIVIDED - W/MEDIAN (NOT RAISED)',
    'DIVIDED - W/MEDIAN BARRIER', 'DRIVEWAY', 'FIVE POINT, OR MORE',
    'FOUR WAY', 'L-INTERSECTION', 'NOT DIVIDED', 'NOT REPORTED',
    'ONE-WAY', 'OTHER', 'PARKING LOT', 'RAMP', 'ROUNDABOUT',
    'T-INTERSECTION', 'TRAFFIC ROUTE', 'UNKNOWN',
    'UNKNOWN INTERSECTION TYPE', 'Y-INTERSECTION'
]

ALIGNMENT_OPTIONS = [
    'CURVE ON GRADE', 'CURVE ON HILLCREST', 'CURVE, LEVEL',
    'STRAIGHT AND LEVEL', 'STRAIGHT ON GRADE', 'STRAIGHT ON HILLCREST'
]

TRAFFIC_CONTROL_OPTIONS = [
    'BICYCLE CROSSING SIGN', 'DELINEATORS', 'FLASHING CONTROL SIGNAL',
    'NO CONTROLS', 'NO PASSING', 'OTHER', 'OTHER RAILROAD CROSSING',
    'OTHER REG. SIGN', 'OTHER WARNING SIGN', 'PEDESTRIAN CROSSING SIGN',
    'POLICE/FLAGMAN', 'RAILROAD CROSSING GATE', 'RR CROSSING SIGN',
    'SCHOOL ZONE', 'STOP SIGN/FLASHER', 'TRAFFIC SIGNAL', 'UNKNOWN', 'YIELD'
]

DEVICE_CONDITION_OPTIONS = [
    'FUNCTIONING IMPROPERLY', 'FUNCTIONING PROPERLY', 'MISSING',
    'NO CONTROLS', 'NOT FUNCTIONING', 'OTHER', 'UNKNOWN',
    'WORN REFLECTIVE MATERIAL'
]

MANEUVER_OPTIONS = [
    'AVOIDING VEHICLES/OBJECTS',
    'BACKING',
    'CHANGING LANES',
    'DISABLED',
    'DIVERGING',
    'DRIVERLESS',
    'DRIVING WRONG WAY',
    'ENTER FROM DRIVE/ALLEY',
    'ENTERING TRAFFIC LANE FROM PARKING',
    'LEAVING TRAFFIC LANE TO PARK',
    'MERGING',
    'NEGOTIATING A CURVE',
    'OTHER',
    'PARKED',
    'PARKED IN TRAFFIC LANE',
    'PASSING/OVERTAKING',
    'SKIDDING/CONTROL LOSS',
    'SLOW/STOP IN TRAFFIC',
    'STARTING IN TRAFFIC',
    'STRAIGHT AHEAD',
    'TURNING LEFT',
    'TURNING ON RED',
    'TURNING RIGHT',
    'U-TURN',
    'UNKNOWN/NA',
    'Unknown'
]

SEX_OPTIONS = ['F', 'M', 'Unknown', 'Unkown', 'X']

st.subheader("Enter crash conditions")

col1, col2 = st.columns(2)

with col1:
    weather_condition = st.selectbox("WEATHER_CONDITION", WEATHER_OPTIONS, index=2)
    lighting_condition = st.selectbox("LIGHTING_CONDITION", LIGHTING_OPTIONS, index=3)
    roadway_surface_cond = st.selectbox("ROADWAY_SURFACE_COND", ROADWAY_SURFACE_OPTIONS, index=0)
    trafficway_type = st.selectbox("TRAFFICWAY_TYPE", TRAFFICWAY_OPTIONS, index=8)
    alignment = st.selectbox("ALIGNMENT", ALIGNMENT_OPTIONS, index=3)
    traffic_control_device = st.selectbox("TRAFFIC_CONTROL_DEVICE", TRAFFIC_CONTROL_OPTIONS, index=3)
    device_condition = st.selectbox("DEVICE_CONDITION", DEVICE_CONDITION_OPTIONS, index=3)
    dominant_maneuver = st.selectbox("dominant_maneuver", MANEUVER_OPTIONS, index=19)

with col2:
    dominant_sex = st.selectbox("dominant_sex", SEX_OPTIONS, index=1)
    num_vehicle_types = st.number_input("num_vehicle_types", min_value=1, max_value=20, value=1, step=1)
    avg_age = st.number_input("avg_age", min_value=16.0, max_value=100.0, value=35.0, step=1.0)
    crash_hour = st.number_input("CRASH_HOUR", min_value=0, max_value=23, value=12, step=1)
    crash_day_of_week = st.number_input("CRASH_DAY_OF_WEEK", min_value=1, max_value=7, value=3, step=1)
    crash_month = st.number_input("CRASH_MONTH", min_value=1, max_value=12, value=6, step=1)
    posted_speed_limit = st.number_input("POSTED_SPEED_LIMIT", min_value=0, max_value=120, value=30, step=5)
    num_people = st.number_input("num_people", min_value=1, max_value=20, value=2, step=1)

if st.button("Predict crash cause"):
    input_data = pd.DataFrame([{
        "WEATHER_CONDITION": weather_condition,
        "LIGHTING_CONDITION": lighting_condition,
        "ROADWAY_SURFACE_COND": roadway_surface_cond,
        "TRAFFICWAY_TYPE": trafficway_type,
        "ALIGNMENT": alignment,
        "TRAFFIC_CONTROL_DEVICE": traffic_control_device,
        "DEVICE_CONDITION": device_condition,
        "dominant_maneuver": dominant_maneuver,
        "num_vehicle_types": num_vehicle_types,
        "avg_age": avg_age,
        "dominant_sex": dominant_sex,
        "CRASH_HOUR": crash_hour,
        "CRASH_DAY_OF_WEEK": crash_day_of_week,
        "CRASH_MONTH": crash_month,
        "POSTED_SPEED_LIMIT": posted_speed_limit,
        "num_people": num_people
    }])

    st.write("### Input preview")
    st.dataframe(input_data)

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

    pred_encoded = model.predict(input_encoded)
    prediction = label_encoder.inverse_transform(pred_encoded)[0]

    st.success(f"Predicted Primary Contributory Cause: {prediction}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_encoded)[0]
        class_labels = label_encoder.inverse_transform(list(range(len(proba))))

        proba_df = pd.DataFrame({
            "Crash Cause": class_labels,
            "Probability": proba
        }).sort_values("Probability", ascending=False)

        st.write("### Top predicted probabilities")
        st.dataframe(proba_df.head(5).reset_index(drop=True))