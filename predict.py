"""Prediction helpers shared by the command-line and Flask clients."""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).parent / "models" / "smart_street_light_model.pkl"
FEATURES = [
    "Area_Type",
    "Hour",
    "Day_Type",
    "Vehicle_Count",
    "Pedestrian_Count",
    "Ambient_Lux",
    "Motion_Detected",
    "Traffic_Level",
]
VALID_AREAS = {"Residential", "Market", "Highway", "School"}
VALID_DAYS = {"Weekday", "Weekend"}
VALID_TRAFFIC = {"Low", "Medium", "High", "Very High"}
SAFE_BRIGHTNESS = 70
MIN_ACTIVE_BRIGHTNESS = 10
LIGHT_OFF_LUX = 800


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive guard for corrupted artifacts
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {exc}") from exc


model = load_model()


def adjust_for_ambient_lux(predicted: int, lux: float, motion: int) -> int:
    """Convert ambient light into a smooth maximum PWM intensity."""
    if float(lux) > LIGHT_OFF_LUX:
        return 0
    if int(motion) == 1:
        return 100
    if float(lux) <= 0:
        return 100
    ambient_cap = round(100 - (max(0.0, min(float(lux), LIGHT_OFF_LUX)) / LIGHT_OFF_LUX) * 100)
    return max(MIN_ACTIVE_BRIGHTNESS, min(100, min(int(predicted), max(MIN_ACTIVE_BRIGHTNESS, ambient_cap))))


def validate_input(values: dict) -> None:
    required = set(FEATURES)
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    if values["Area_Type"] not in VALID_AREAS:
        raise ValueError(f"Area_Type must be one of {sorted(VALID_AREAS)}")
    if values["Day_Type"] not in VALID_DAYS:
        raise ValueError(f"Day_Type must be one of {sorted(VALID_DAYS)}")
    if values["Traffic_Level"] not in VALID_TRAFFIC:
        raise ValueError(f"Traffic_Level must be one of {sorted(VALID_TRAFFIC)}")
    if not 0 <= float(values["Hour"]) <= 23:
        raise ValueError("Hour must be between 0 and 23")
    if not 0 <= float(values["Ambient_Lux"]) <= 100000:
        raise ValueError("Ambient_Lux must be between 0 and 100000")
    if not 0 <= int(values["Vehicle_Count"]) <= 10000:
        raise ValueError("Vehicle_Count is outside the allowed range")
    if not 0 <= int(values["Pedestrian_Count"]) <= 10000:
        raise ValueError("Pedestrian_Count is outside the allowed range")
    if int(values["Motion_Detected"]) not in (0, 1):
        raise ValueError("Motion_Detected must be 0 or 1")


def predict_brightness(
    area: str,
    hour: int,
    day: str,
    vehicles: int,
    pedestrians: int,
    lux: float,
    motion: int,
    traffic: str,
) -> dict:
    values = {
        "Area_Type": area,
        "Hour": hour,
        "Day_Type": day,
        "Vehicle_Count": vehicles,
        "Pedestrian_Count": pedestrians,
        "Ambient_Lux": lux,
        "Motion_Detected": motion,
        "Traffic_Level": traffic,
    }
    validate_input(values)
    frame = pd.DataFrame([values], columns=FEATURES)
    prediction = int(round(float(model.predict(frame)[0])))
    probabilities = model.predict_proba(frame)[0]
    confidence = float(max(probabilities))
    if confidence < 0.55:
        prediction = SAFE_BRIGHTNESS
        fallback = True
    else:
        fallback = False
    return {
        "brightness": adjust_for_ambient_lux(prediction, lux, motion),
        "confidence": round(confidence, 4),
        "fallback_used": fallback,
    }


if __name__ == "__main__":
    examples = [
        ("Market", 19, "Weekday", 70, 25, 15, 1, "High"),
        ("Residential", 2, "Weekday", 8, 2, 10, 0, "Low"),
        ("Highway", 20, "Weekend", 90, 10, 12, 1, "Very High"),
    ]
    for example in examples:
        print(predict_brightness(*example))
