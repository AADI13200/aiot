"""Create a reproducible, balanced training dataset for the street-light model.

The generated rows are synthetic augmentation. Replace them gradually with
ESP32 readings and operator-approved brightness labels for production use.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
OUTPUT_PATH = Path("traffic_data.csv")
ORIGINAL_PATH = Path("traffic_data_original.csv")
AUGMENTED_PATH = Path("traffic_data_augmented.csv")
TARGET_ROWS = 2_500
COLUMNS = [
    "Area_Type",
    "Hour",
    "Day_Type",
    "Vehicle_Count",
    "Pedestrian_Count",
    "Ambient_Lux",
    "Motion_Detected",
    "Traffic_Level",
    "Brightness_Level",
]


def traffic_level(vehicles: int, pedestrians: int) -> str:
    activity = vehicles + pedestrians * 1.5
    if activity < 25:
        return "Low"
    if activity < 75:
        return "Medium"
    if activity < 125:
        return "High"
    return "Very High"


def brightness_target(
    area: str,
    hour: int,
    vehicles: int,
    pedestrians: int,
    lux: float,
    motion: int,
) -> int:
    """Generate an operator-like label from safety and energy rules."""
    if lux >= 180 and motion == 0:
        return 30

    activity = vehicles + pedestrians * 1.5
    night_factor = 1 if hour >= 19 or hour <= 5 else 0

    if motion and night_factor and activity >= 70:
        return 100
    if motion and night_factor:
        return 75
    if night_factor and activity >= 80:
        return 75
    if area == "Highway" and night_factor:
        return 75
    if night_factor:
        return 50
    return 30


def generate_rows(count: int, rng: np.random.Generator) -> pd.DataFrame:
    areas = ["Residential", "Market", "Highway", "School"]
    area_weights = [0.30, 0.28, 0.22, 0.20]
    rows = []

    for _ in range(count):
        area = rng.choice(areas, p=area_weights)
        hour = int(rng.integers(0, 24))
        day = str(rng.choice(["Weekday", "Weekend"], p=[0.72, 0.28]))
        is_night = hour >= 19 or hour <= 5

        area_limits = {
            "Residential": (45, 20),
            "Market": (110, 55),
            "Highway": (140, 25),
            "School": (75, 45),
        }
        vehicle_limit, pedestrian_limit = area_limits[area]
        busy_factor = 1.35 if area == "Market" and 8 <= hour <= 21 else 1.0
        if area == "School" and 7 <= hour <= 16:
            busy_factor *= 1.35
        if not is_night:
            busy_factor *= 1.15

        vehicles = int(np.clip(rng.poisson(vehicle_limit / 3 * busy_factor), 0, vehicle_limit))
        pedestrians = int(np.clip(rng.poisson(pedestrian_limit / 3 * busy_factor), 0, pedestrian_limit))
        motion_probability = min(0.95, 0.10 + (vehicles + pedestrians) / 160)
        motion = int(rng.random() < motion_probability)

        daylight = max(0, np.sin((hour - 6) * np.pi / 12))
        lux = float(np.clip(10 + daylight * 480 + rng.normal(0, 20), 0, 550))
        if is_night:
            lux = float(np.clip(rng.normal(18, 10), 0, 70))

        label = brightness_target(area, hour, vehicles, pedestrians, lux, motion)
        if rng.random() < 0.08:
            label = int(rng.choice([30, 50, 75, 100]))

        rows.append(
            {
                "Area_Type": area,
                "Hour": hour,
                "Day_Type": day,
                "Vehicle_Count": vehicles,
                "Pedestrian_Count": pedestrians,
                "Ambient_Lux": round(lux, 2),
                "Motion_Detected": motion,
                "Traffic_Level": traffic_level(vehicles, pedestrians),
                "Brightness_Level": label,
            }
        )

    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    rng = np.random.default_rng(SEED)
    original = pd.read_csv(ORIGINAL_PATH)
    generated = generate_rows(TARGET_ROWS, rng)
    combined = pd.concat([original[COLUMNS], generated], ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined.to_csv(AUGMENTED_PATH, index=False)

    print(f"Original rows retained: {len(original)}")
    print(f"Synthetic rows generated: {len(generated)}")
    print(f"Augmented dataset written separately: {AUGMENTED_PATH}")
    print(f"Augmented dataset rows: {len(combined)}")
    print("Brightness distribution:")
    print(combined["Brightness_Level"].value_counts().sort_index().to_string())
    print("Synthetic augmentation is marked by this generator; validate with real ESP32 data before deployment.")


if __name__ == "__main__":
    main()
