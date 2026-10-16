import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_start import detect_esp32_port
from predict import predict_brightness
from server import app


class SmartStreetLightTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_prediction_has_safe_result_contract(self):
        result = predict_brightness(
            "Market", 19, "Weekday", 70, 25, 15, 1, "High"
        )
        self.assertIn(result["brightness"], {30, 50, 70, 75, 100})
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)
        self.assertIn("fallback_used", result)

    def test_invalid_sensor_input_is_rejected(self):
        response = self.client.post(
            "/predict",
            json={
                "area": "Market",
                "hour": 99,
                "day": "Weekday",
                "vehicles": 70,
                "pedestrians": 25,
                "lux": 15,
                "motion": 1,
                "traffic": "High",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["status"], "error")

    def test_health_and_metrics_endpoints(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertTrue(metrics.is_json)

    def test_readings_csv_includes_energy_consumption(self):
        response = self.client.get("/readings.csv")
        header = response.get_data(as_text=True).splitlines()[0].split(",")
        self.assertIn("energy_wh", header)

    def test_sensor_endpoint_applies_minimum_active_brightness(self):
        response = self.client.post(
            "/sensor",
            json={
                "area": "Market",
                "hour": 12,
                "day": "Weekday",
                "vehicles": 0,
                "pedestrians": 0,
                "lux": 250,
                "motion": 0,
                "traffic": "Low",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.get_json()["brightness"], 10)

    def test_lux_above_800_turns_light_off(self):
        response = self.client.post(
            "/sensor",
            json={
                "area": "Market",
                "hour": 12,
                "day": "Weekday",
                "vehicles": 20,
                "pedestrians": 5,
                "lux": 801,
                "motion": 1,
                "traffic": "Medium",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["brightness"], 0)

    def test_zero_lux_requires_full_brightness(self):
        response = self.client.post(
            "/sensor",
            json={
                "area": "Residential",
                "hour": 2,
                "day": "Weekday",
                "vehicles": 8,
                "pedestrians": 2,
                "lux": 0,
                "motion": 0,
                "traffic": "Low",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["brightness"], 100)

    def test_motion_overrides_minimum_before_light_cutoff(self):
        response = self.client.post(
            "/sensor",
            json={
                "area": "Residential",
                "hour": 12,
                "day": "Weekday",
                "vehicles": 8,
                "pedestrians": 2,
                "lux": 250,
                "motion": 1,
                "traffic": "Low",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["brightness"], 100)

    def test_connection_endpoint_detects_live_esp32_port(self):
        ports = [
            SimpleNamespace(
                device="COM5",
                description="Silicon Labs CP210x USB to UART Bridge (COM5)",
                hwid="USB VID:PID=10C4:EA60 SER=0001 LOCATION=1-1",
            )
        ]
        with patch("server.list_ports.comports", return_value=ports):
            response = self.client.get("/connection")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["port_connected"])

    def test_connection_endpoint_ignores_stale_watcher_heartbeat(self):
        import server

        with patch.object(server, "CONNECTION_PATH") as connection_path, patch(
            "server.list_ports.comports", return_value=[]
        ):
            connection_path.exists.return_value = True
            connection_path.read_text.return_value = (
                '{"watcher": true, "bridge_running": true, '
                '"api_running": true, "timestamp": 0}'
            )
            response = self.client.get("/connection")
            status = response.get_json()
            self.assertFalse(status["watcher"])
            self.assertFalse(status["bridge_running"])

    def test_model_artifacts_exist(self):
        self.assertTrue(Path("models/smart_street_light_model.pkl").exists())
        self.assertTrue(Path("outputs/model_metrics.json").exists())
        self.assertTrue(Path("outputs/confusion_matrix.csv").exists())

    def test_load_model_reports_missing_file_clearly(self):
        import predict

        with patch("predict.MODEL_PATH", Path("models/does_not_exist.pkl")):
            with self.assertRaises(FileNotFoundError):
                predict.load_model()

    def test_esp32_port_is_detected_from_available_serial_ports(self):
        ports = [
            SimpleNamespace(
                device="COM3",
                description="USB Serial Device",
                hwid="USB VID:PID=10C4:EA60",
            ),
            SimpleNamespace(
                device="COM5",
                description="Standard Serial over Bluetooth link",
                hwid="BTHENUM\\{00001101-0000-1000-8000-00805F9B34FB}",
            ),
        ]
        with patch("auto_start.list_ports.comports", return_value=ports):
            self.assertEqual(detect_esp32_port(), "COM3")

    def test_auto_start_port_check_accepts_any_valid_esp32_port(self):
        ports = [
            SimpleNamespace(
                device="COM3",
                description="Silicon Labs CP210x USB to UART Bridge (COM3)",
                hwid="USB VID:PID=10C4:EA60",
            )
        ]
        with patch("auto_start.PORT_NAME", "COM5"), patch(
            "auto_start.list_ports.comports", return_value=ports
        ):
            from auto_start import port_is_connected

            self.assertTrue(port_is_connected())

    def test_auto_start_port_check_uses_reconnected_port(self):
        ports = [
            SimpleNamespace(
                device="COM7",
                description="USB Serial Device",
                hwid="USB VID:PID=10C4:EA60",
            )
        ]
        with patch("auto_start.list_ports.comports", return_value=ports):
            from auto_start import port_is_connected

            self.assertTrue(port_is_connected("COM7"))


if __name__ == "__main__":
    unittest.main()
