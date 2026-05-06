import random
import time
import requests

API_URL = "http://127.0.0.1:8000/predict"

weather_options = ["CLEAR", "RAIN", "SNOW", "CLOUDY/OVERCAST", "FOG/SMOKE/HAZE"]
lighting_options = ["DAYLIGHT", "DARKNESS", "DARKNESS, LIGHTED ROAD", "DAWN", "DUSK"]
road_options = ["DRY", "WET", "ICE", "SNOW OR SLUSH"]
trafficway_options = ["NOT DIVIDED", "ONE-WAY", "FOUR WAY", "DIVIDED - W/MEDIAN BARRIER"]
alignment_options = ["STRAIGHT AND LEVEL", "STRAIGHT ON GRADE", "CURVE, LEVEL"]
traffic_control_options = ["NO CONTROLS", "TRAFFIC SIGNAL", "STOP SIGN/FLASHER"]
device_condition_options = ["NO CONTROLS", "FUNCTIONING PROPERLY", "UNKNOWN"]
maneuver_options = ["STRAIGHT AHEAD", "TURNING LEFT", "TURNING RIGHT", "CHANGING LANES"]
sex_options = ["M", "F", "Unknown"]


def generate_event():
    return {
        "WEATHER_CONDITION": random.choice(weather_options),
        "LIGHTING_CONDITION": random.choice(lighting_options),
        "ROADWAY_SURFACE_COND": random.choice(road_options),
        "TRAFFICWAY_TYPE": random.choice(trafficway_options),
        "ALIGNMENT": random.choice(alignment_options),
        "TRAFFIC_CONTROL_DEVICE": random.choice(traffic_control_options),
        "DEVICE_CONDITION": random.choice(device_condition_options),
        "dominant_maneuver": random.choice(maneuver_options),
        "dominant_sex": random.choice(sex_options),
        "num_vehicle_types": random.randint(1, 4),
        "avg_age": random.randint(18, 80),
        "CRASH_HOUR": random.randint(0, 23),
        "CRASH_DAY_OF_WEEK": random.randint(1, 7),
        "CRASH_MONTH": random.randint(1, 12),
        "POSTED_SPEED_LIMIT": random.choice([20, 25, 30, 35, 40, 45, 55]),
        "num_people": random.randint(1, 8)
    }


while True:
    event = generate_event()

    response = requests.post(API_URL, json=event)

    print("\nIncoming simulated crash event:")
    print(event)

    print("\nModel prediction:")
    print(response.json())

    print("-" * 80)

    time.sleep(3)
