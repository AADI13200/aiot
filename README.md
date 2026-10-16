# Smart Street Light ML Pilot

An ESP32-connected street-light controller that predicts brightness from ambient light, motion, traffic, area, and time. The system combines machine learning with safety rules so uncertain predictions do not directly control the lamp.

## Architecture

```text
ESP32 sensors -> smart_light.py -> trained model -> brightness command
                         |
                         +-> optional Flask API -> outputs/readings.csv

traffic_data.csv -> train_model.py -> models/*.pkl
                         ^
                         +-> generate_dataset.py -> traffic_data_augmented.csv (optional experiment)
```

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the model

The repository's training file contains the original 100 source rows. Synthetic rows are generated only into `traffic_data_augmented.csv` for comparison experiments; they do not overwrite or silently enter the source training file.

```powershell
python generate_dataset.py
python train_model.py
```

Training writes:

- `models/smart_street_light_model.pkl`
- `outputs/model_metrics.json`
- `outputs/confusion_matrix.csv`
- `traffic_data_augmented.csv` (optional synthetic experiment)

The trainer compares Decision Tree, Random Forest, and Extra Trees using accuracy, balanced accuracy, and macro-F1. Model selection prioritizes balanced accuracy and macro-F1 because the brightness classes are not equally common.

## Run the API

```powershell
python server.py
```

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /readings`
- `POST /predict` (API testing)
- `POST /sensor` (ESP32 live readings)

Example request:

```json
{
  "area": "Market",
  "hour": 19,
  "day": "Weekday",
  "vehicles": 70,
  "pedestrians": 25,
  "lux": 15,
  "motion": 1,
  "traffic": "High"
}
```

The response includes `brightness`, `confidence`, and `fallback_used`. Valid readings are logged to `outputs/readings.csv`. The browser dashboard does not ask for manual values; it displays the latest readings received from the ESP32 bridge.

## Run with the ESP32

1. Close Arduino Serial Monitor and any other serial application.
2. Confirm the ESP32 port in `.venv/smart_light.py` (`COM5` by default).
3. Start only one copy:

```powershell
python .venv\smart_light.py
```

Or configure the port without editing the script:

```powershell
$env:ESP32_PORT = "COM5"
$env:ESP32_BAUD = "115200"
python .venv\smart_light.py
```

The live controller turns lights off in daylight, sends sensor readings to the API, applies ML at night, forces full brightness for detected motion, and uses a 70% fallback when model confidence is below `0.55`.

The bridge asks for only the area. It automatically derives the current hour and day, estimates traffic context from the selected area plus lux and motion, and sends those values to the model.

For automatic USB detection, run the watcher once:

```powershell
python auto_start.py
```

The watcher detects `COM5`, starts the API if needed, launches one ESP32 bridge, and restarts it when the board is disconnected and reconnected. To run it automatically when Windows logs in, add `python auto_start.py` from this project folder to Windows Task Scheduler.

## Run tests

```powershell
python -m unittest -v test_project.py
```

## Pilot-quality next step

The current source dataset is small. Replace it gradually with several days of real readings and operator-approved brightness labels. Retrain periodically and compare energy saved against a fixed-brightness baseline before deploying the model permanently.
