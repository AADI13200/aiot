"""Keep the Smart Street Light API and ESP32 bridge connected automatically."""

import os
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from serial.tools import list_ports


ROOT = Path(__file__).parent


def detect_esp32_port() -> str:
    env_port = os.getenv("ESP32_PORT")
    if env_port:
        return env_port

    candidates = []
    for port in list_ports.comports():
        device = getattr(port, "device", "")
        description = (getattr(port, "description", "") or "").lower()
        hwid = (getattr(port, "hwid", "") or "").lower()
        combined = f"{device} {description} {hwid}"

        if any(token in combined for token in ["esp32", "cp210", "ch340", "usb serial", "silicon labs", "uart"]):
            candidates.append(device)

    if candidates:
        return candidates[0]

    ports = [port.device for port in list_ports.comports()]
    if ports:
        return ports[0]

    return "COM5"


PORT_NAME = detect_esp32_port()
POLL_SECONDS = 2
STATUS_PATH = ROOT / "outputs" / "connection_status.json"


def is_esp32_port(port, expected_port=None) -> bool:
    device = getattr(port, "device", "") or ""
    description = (getattr(port, "description", "") or "").lower()
    hwid = (getattr(port, "hwid", "") or "").lower()
    combined = f"{device} {description} {hwid}"
    tokens = [
        "esp32",
        "cp210",
        "ch340",
        "usb serial",
        "silicon labs",
        "uart",
    ]
    expected = expected_port or PORT_NAME
    return any(token in combined for token in tokens) or device.upper() == expected.upper()


def api_is_running() -> bool:
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=1)
        return response.ok
    except requests.RequestException:
        return False


def port_is_connected(expected_port=None) -> bool:
    return any(
        is_esp32_port(port, expected_port) for port in list_ports.comports()
    )


def stop_process(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def save_status(port_connected, bridge_running, api_running, watcher_running=True, port=None):
    STATUS_PATH.parent.mkdir(exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(
            {
                "port": port or PORT_NAME,
                "port_connected": bool(port_connected),
                "bridge_running": bool(bridge_running),
                "api_running": bool(api_running),
                "watcher": bool(watcher_running),
                "timestamp": time.time(),
            }
        ),
        encoding="utf-8",
    )


def main() -> None:
    api_process = None
    bridge_process = None
    print(f"Auto-start watcher active for {PORT_NAME}.")
    save_status(False, False, api_is_running())

    try:
        while True:
            if not api_is_running():
                if api_process is None or api_process.poll() is not None:
                    api_process = subprocess.Popen(
                        [sys.executable, str(ROOT / "server.py")],
                        cwd=ROOT,
                    )
                    print("API started automatically.")

            detected_port = detect_esp32_port()
            connected = port_is_connected(detected_port)
            bridge_running = bridge_process and bridge_process.poll() is None
            if connected and not bridge_running and api_is_running():
                environment = os.environ.copy()
                environment["ESP32_PORT"] = detected_port
                bridge_process = subprocess.Popen(
                    [sys.executable, str(ROOT / ".venv" / "smart_light.py")],
                    cwd=ROOT,
                    env=environment,
                )
                print(f"ESP32 detected on {detected_port}; bridge started.")
            elif not connected and bridge_running:
                stop_process(bridge_process)
                bridge_process = None
                print(f"{PORT_NAME} disconnected; bridge stopped.")

            save_status(
                connected,
                bridge_process and bridge_process.poll() is None,
                api_is_running(),
                port=detected_port,
            )

            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("Auto-start watcher stopped.")
    finally:
        stop_process(bridge_process)
        stop_process(api_process)
        save_status(False, False, api_is_running(), watcher_running=False)


if __name__ == "__main__":
    main()
