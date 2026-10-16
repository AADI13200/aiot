"""Flask API for the smart street-light ML model."""

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, request
from serial.tools import list_ports

from predict import MODEL_PATH, predict_brightness


app = Flask(__name__)


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


READINGS_PATH = Path(__file__).parent / "outputs" / "readings.csv"
LIVE_READINGS_PATH = Path(__file__).parent / "outputs" / "live_readings.csv"
CONNECTION_PATH = Path(__file__).parent / "outputs" / "connection_status.json"
CONFIG_PATH = Path(__file__).parent / "outputs" / "area_config.json"
VALID_AREAS = {"Residential", "Market", "Highway", "School"}
DEFAULT_AREA = "Residential"
RATED_LAMP_WATTS = float(os.getenv("SMART_LIGHT_LAMP_WATTS", "10"))
READING_COLUMNS = [
    "timestamp",
    "area",
    "hour",
    "day",
    "vehicles",
    "pedestrians",
    "lux",
    "motion",
    "traffic",
    "brightness",
    "confidence",
    "fallback_used",
    "energy_wh",
]


def load_area():
    if not CONFIG_PATH.exists():
        return DEFAULT_AREA
    try:
        area = json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("area")
        return area if area in VALID_AREAS else DEFAULT_AREA
    except (OSError, json.JSONDecodeError):
        return DEFAULT_AREA


def save_area(area):
    CONFIG_PATH.parent.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({"area": area}), encoding="utf-8")


def save_reading(payload: dict, result: dict, readings_path: Path | None = None) -> None:
    if readings_path is None:
        readings_path = READINGS_PATH
    readings_path.parent.mkdir(exist_ok=True)
    previous_rows = []
    if readings_path.exists():
        with readings_path.open(newline="", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            previous_rows = list(reader)

    timestamp = datetime.now(timezone.utc)
    elapsed_seconds = 1.0
    if previous_rows:
        try:
            previous_timestamp = datetime.fromisoformat(previous_rows[-1]["timestamp"])
            elapsed_seconds = max(0.0, min((timestamp - previous_timestamp).total_seconds(), 60.0))
        except (KeyError, TypeError, ValueError):
            pass

    brightness = float(result.get("brightness", 0))
    energy_wh = RATED_LAMP_WATTS * max(0.0, min(brightness, 100.0)) / 100 * elapsed_seconds / 3600
    new_file = not readings_path.exists()
    needs_migration = bool(previous_rows and "energy_wh" not in previous_rows[0])
    with readings_path.open("a", newline="", encoding="utf-8") as readings_file:
        writer = csv.DictWriter(readings_file, fieldnames=READING_COLUMNS)
        if new_file:
            writer.writeheader()
        elif needs_migration:
            readings_file.seek(0)
            readings_file.truncate()
            writer.writeheader()
            for previous_row in previous_rows:
                previous_row["energy_wh"] = ""
                writer.writerow(previous_row)
        writer.writerow(
            {
                "timestamp": timestamp.isoformat(),
                **payload,
                **result,
                "energy_wh": round(energy_wh, 6),
            }
        )


def port_is_connected() -> bool:
    for port in list_ports.comports():
        device = getattr(port, "device", "") or ""
        description = (getattr(port, "description", "") or "").lower()
        hwid = (getattr(port, "hwid", "") or "").lower()
        combined = f"{device} {description} {hwid}"
        if any(token in combined for token in ["esp32", "cp210", "ch340", "usb serial", "silicon labs", "uart"]):
            return True
        if os.getenv("ESP32_PORT") and device.upper() == os.getenv("ESP32_PORT", "").upper():
            return True
    return False


@app.get("/")
def home():
        return render_template_string(
                """
                <!doctype html>
                <html lang="en">
                <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <title>Smart Street Light Control Room</title>
                    <style>
                        :root { --ink: #17231f; --muted: #63736c; --paper: #f4f1e8; --panel: #fffdf7; --line: #d9ded5; --green: #2d6a4f; --lime: #c9e4ca; --amber: #d87941; }
                        * { box-sizing: border-box; }
                        body { margin: 0; color: var(--ink); background: var(--paper); font-family: Georgia, 'Times New Roman', serif; }
                        main { max-width: 1180px; margin: 0 auto; padding: 34px 22px 56px; }
                        header { display: flex; justify-content: space-between; align-items: end; gap: 24px; padding-bottom: 28px; border-bottom: 1px solid var(--line); }
                        h1 { margin: 0; font-size: clamp(2rem, 5vw, 4.5rem); line-height: .95; letter-spacing: 0; max-width: 680px; }
                        .eyebrow { margin: 0 0 12px; color: var(--green); font: 700 12px/1.2 Arial, sans-serif; letter-spacing: 1.5px; text-transform: uppercase; }
                        .status { display: flex; align-items: center; gap: 9px; color: var(--green); font: 700 13px Arial, sans-serif; white-space: nowrap; }
                        .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 5px var(--lime); }
                        .grid { display: grid; grid-template-columns: 1.2fr .8fr; gap: 18px; margin-top: 24px; }
                        .panel { background: var(--panel); border: 1px solid var(--line); padding: 22px; box-shadow: 0 10px 30px rgba(23,35,31,.05); }
                        .panel h2 { margin: 0 0 18px; font: 700 13px Arial, sans-serif; letter-spacing: 1px; text-transform: uppercase; }
                        .section-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
                        .section-heading h2 { margin-bottom: 18px; }
                        .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
                        .card { border-left: 3px solid var(--green); background: #f0f5ed; padding: 15px; min-height: 92px; }
                        .value { display: block; margin-top: 8px; font-size: 30px; }
                        .label { color: var(--muted); font: 11px Arial, sans-serif; text-transform: uppercase; letter-spacing: .7px; }
                        form { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
                        label { color: var(--muted); font: 11px Arial, sans-serif; text-transform: uppercase; letter-spacing: .5px; }
                        input, select { width: 100%; margin-top: 6px; border: 1px solid var(--line); background: white; color: var(--ink); padding: 11px; font: 15px Georgia, serif; }
                        button { border: 0; background: var(--green); color: white; padding: 11px 14px; cursor: pointer; font: 700 12px Arial, sans-serif; text-transform: uppercase; letter-spacing: .7px; }
                        button:hover { background: #1f4f3a; }
                        .area-controls { display: flex; gap: 8px; margin-bottom: 16px; }
                        .area-controls select { flex: 1; margin-top: 0; }
                        .reset { background: var(--amber); }
                        .result { margin-top: 16px; padding: 14px; background: var(--lime); min-height: 52px; font: 14px Arial, sans-serif; }
                        .importance { display: grid; gap: 8px; font: 12px Arial, sans-serif; }
                        .importance-row { display: grid; grid-template-columns: 150px 1fr 45px; gap: 8px; align-items: center; }
                        .bar { height: 8px; background: var(--lime); }
                        .bar span { display: block; height: 100%; background: var(--green); }
                        .metric-table { width: 100%; border-collapse: collapse; font: 12px Arial, sans-serif; }
                        .metric-table th, .metric-table td { padding: 8px; border-bottom: 1px solid var(--line); text-align: left; }
                        .wide { grid-column: 1 / -1; }
                        table { width: 100%; border-collapse: collapse; font: 13px Arial, sans-serif; }
                        th, td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }
                        th { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .7px; }
                        .empty { color: var(--muted); font: 14px Arial, sans-serif; }
                        .download { color: var(--green); font: 700 11px Arial, sans-serif; letter-spacing: .5px; text-transform: uppercase; }
                        .analytics { display: grid; grid-template-columns: 1fr 220px; gap: 20px; align-items: stretch; }
                        .chart-wrap { min-width: 0; }
                        canvas { display: block; width: 100%; height: 260px; background: #f7faf4; border: 1px solid var(--line); }
                        .analytics-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
                        .analytics-stat { background: #f0f5ed; padding: 14px; border-left: 3px solid var(--green); }
                        .analytics-stat strong { display: block; margin-top: 7px; font-size: 24px; font-weight: 400; }
                        .legend { display: flex; gap: 14px; margin-top: 9px; color: var(--muted); font: 11px Arial, sans-serif; }
                        .legend span::before { content: ''; display: inline-block; width: 18px; height: 3px; margin: 0 5px 3px 0; background: var(--green); }
                        .legend .brightness::before { background: var(--amber); }
                        .legend .motion::before { width: 9px; height: 9px; border-radius: 50%; background: #b23a48; }
                        @media (max-width: 760px) { header, .grid { display: block; } .status { margin-top: 24px; } .panel { margin-top: 16px; } .cards { grid-template-columns: 1fr; } }
                        @media (max-width: 760px) { .analytics { grid-template-columns: 1fr; } }
                    </style>
                </head>
                <body>
                    <main>
                        <header>
                            <div><p class="eyebrow">IoT + Machine Learning</p><h1>Smart street light control room</h1></div>
                            <div class="status"><span class="dot"></span><span id="health">Checking model...</span></div>
                        </header>
                        <section class="grid">
                            <div class="panel">
                                <h2>Model snapshot</h2>
                                <div class="cards">
                                    <div class="card"><span class="label">Model</span><span class="value" id="model">-</span></div>
                                    <div class="card"><span class="label">Source training rows</span><span class="value" id="rows">-</span></div>
                                    <div class="card"><span class="label">Holdout balanced accuracy</span><span class="value" id="accuracy">-</span></div>
                                    <div class="card"><span class="label">Cross-validation</span><span class="value" id="cv-score">-</span></div>
                                    <div class="card"><span class="label">Macro-F1</span><span class="value" id="f1-score">-</span></div>
                                </div>
                            </div>
                            <div class="panel">
                                <h2>What the model learned</h2>
                                <div id="classes" class="result">Target classes: -</div>
                                <div id="importance" class="importance">Loading feature importance...</div>
                            </div>
                            <div class="panel wide">
                                <h2>Model comparison</h2>
                                <div id="comparison" class="empty">Loading evaluation results...</div>
                            </div>
                            <div class="panel wide">
                                <h2>Confusion matrix</h2>
                                <div id="confusion" class="empty">Loading class performance...</div>
                            </div>
                            <div class="panel">
                                <h2>Live ESP32 feed</h2>
                                <div class="area-controls">
                                    <select id="area-select" aria-label="Area"><option value="">Select area</option><option>Residential</option><option>Market</option><option>Highway</option><option>School</option></select>
                                    <button id="set-area">Set area</button>
                                    <button id="reset-area" class="reset">Reset</button>
                                </div>
                                <div class="cards">
                                    <div class="card"><span class="label">Ambient lux</span><span class="value" id="live-lux">-</span></div>
                                    <div class="card"><span class="label">Motion</span><span class="value" id="live-motion">-</span></div>
                                    <div class="card"><span class="label">Brightness</span><span class="value" id="live-brightness">-</span></div>
                                </div>
                                <div class="result" id="result">Waiting for the ESP32 to send a sensor reading...</div>
                            </div>
                            <div class="panel wide">
                                <h2>Live data analytics</h2>
                                <div class="analytics">
                                    <div class="chart-wrap">
                                        <canvas id="analytics-chart" aria-label="Live lux, brightness, and energy trends"></canvas>
                                        <div class="legend"><span>Ambient lux</span><span class="brightness">Brightness</span><span class="energy">Cumulative energy (Wh)</span><span class="motion">Motion detected</span></div>
                                    </div>
                                    <div class="analytics-stats">
                                        <div class="analytics-stat"><span class="label">Average lux</span><strong id="average-lux">-</strong></div>
                                        <div class="analytics-stat"><span class="label">Average brightness</span><strong id="average-brightness">-</strong></div>
                                        <div class="analytics-stat"><span class="label">Motion events</span><strong id="motion-events">-</strong></div>
                                        <div class="analytics-stat"><span class="label">Detected samples</span><strong id="detected-samples">-</strong></div>
                                        <div class="analytics-stat"><span class="label">Non-detected samples</span><strong id="clear-samples">-</strong></div>
                                        <div class="analytics-stat"><span class="label">Samples shown</span><strong id="sample-count">-</strong></div>
                                    </div>
                                </div>
                            </div>
                            <div class="panel wide">
                                <div class="section-heading"><h2>Latest system readings</h2><a class="download" href="/readings.csv" download="smart-light-readings.csv">Download CSV</a></div>
                                <div id="readings" class="empty">Loading readings...</div>
                            </div>
                        </section>
                    </main>
                    <script>
                        const text = (id, value) => document.getElementById(id).textContent = value;
                        let analyticsAnimationFrame = null;
                        let analyticsScrollOffset = 0;
                        let analyticsLastFrame = 0;
                        let analyticsLastPaint = 0;
                        let analyticsAnimating = false;
                        let analyticsFrameReadings = [];
                        let analyticsTargetPoints = [];
                        let analyticsCanvasMetrics = null;
                        let analyticsCanvas = null;
                        let analyticsSmoothedLux = [];
                        let analyticsSmoothedBrightness = [];
                        let analyticsSmoothedEnergy = [];
                        let analyticsMaximum = 100;
                        let analyticsEnergyMaximum = 0.001;
                        function startAnalyticsLoop() {
                            const animateAnalytics = now => {
                                if (!analyticsLastFrame) analyticsLastFrame = now;
                                analyticsScrollOffset += (now - analyticsLastFrame) / 1000 * 12;
                                analyticsLastFrame = now;
                                if (analyticsScrollOffset >= 14) {
                                    analyticsScrollOffset = 14;
                                    analyticsAnimating = false;
                                }
                                if (analyticsFrameReadings.length && now - analyticsLastPaint >= 16) {
                                    analyticsLastPaint = now;
                                    drawAnalytics(analyticsFrameReadings, analyticsScrollOffset, true);
                                }
                                if (analyticsAnimating) {
                                    analyticsAnimationFrame = requestAnimationFrame(animateAnalytics);
                                } else {
                                    analyticsAnimationFrame = null;
                                }
                            };
                            analyticsAnimationFrame = requestAnimationFrame(animateAnalytics);
                        }
                        function drawAnalytics(readings, scrollOffset = 0, renderOnly = false) {
                            if (!renderOnly) {
                                analyticsFrameReadings = readings;
                                analyticsScrollOffset = 0;
                                analyticsLastFrame = 0;
                                analyticsAnimating = false;
                            }
                            if (!analyticsCanvas) analyticsCanvas = document.getElementById('analytics-chart');
                            const canvas = analyticsCanvas;
                            if (!renderOnly || !analyticsCanvasMetrics) {
                                const context = canvas.getContext('2d', { alpha: false, desynchronized: true });
                                const bounds = canvas.getBoundingClientRect();
                                const ratio = window.devicePixelRatio || 1;
                                canvas.width = Math.max(1, Math.floor(bounds.width * ratio));
                                canvas.height = Math.max(1, Math.floor(bounds.height * ratio));
                                context.setTransform(ratio, 0, 0, ratio, 0, 0);
                                analyticsCanvasMetrics = { context, width: bounds.width, height: bounds.height };
                            }
                            const { context, width, height } = analyticsCanvasMetrics;
                            context.clearRect(0, 0, width, height);
                            if (!renderOnly) {
                                let cumulativeEnergy = 0;
                                analyticsTargetPoints = readings.map(reading => {
                                    const energy = Number(reading.energy_wh);
                                    cumulativeEnergy += Number.isFinite(energy) ? Math.max(0, energy) : 0;
                                    return {
                                        lux: Number(reading.lux),
                                        brightness: Number(reading.brightness),
                                        energy: cumulativeEnergy,
                                        motion: reading.motion === '1',
                                    };
                                }).filter(point => Number.isFinite(point.lux) && Number.isFinite(point.brightness));
                            }
                            const targetPoints = analyticsTargetPoints;
                            const points = targetPoints;
                            if (!renderOnly) {
                                const averageLux = points.length ? points.reduce((sum, point) => sum + point.lux, 0) / points.length : 0;
                                const averageBrightness = points.length ? points.reduce((sum, point) => sum + point.brightness, 0) / points.length : 0;
                                text('average-lux', points.length ? averageLux.toFixed(1) : '-');
                                text('average-brightness', points.length ? averageBrightness.toFixed(1) + '%' : '-');
                                const detectedSamples = points.filter(point => point.motion).length;
                                const motionEvents = points.reduce((count, point, index) => count + (point.motion && (index === 0 || !points[index - 1].motion) ? 1 : 0), 0);
                                text('motion-events', points.length ? motionEvents : '-');
                                text('detected-samples', points.length ? detectedSamples : '-');
                                text('clear-samples', points.length ? points.length - detectedSamples : '-');
                                text('sample-count', points.length || '-');
                                const smooth = key => points.map((point, index) => {
                                    const start = Math.max(0, index - 4);
                                    const values = points.slice(start, index + 1).map(item => item[key]);
                                    return values.reduce((sum, value) => sum + value, 0) / values.length;
                                });
                                analyticsSmoothedLux = smooth('lux');
                                analyticsSmoothedBrightness = smooth('brightness');
                                analyticsSmoothedEnergy = smooth('energy');
                                analyticsMaximum = Math.max(100, ...points.map(point => point.lux));
                                analyticsEnergyMaximum = Math.max(0.001, ...points.map(point => point.energy));
                            }
                            if (!points.length) {
                                context.fillStyle = '#63736c';
                                context.font = '13px Arial';
                                context.fillText('Waiting for live ESP32 data...', 18, height / 2);
                                return;
                            }
                            const padding = { top: 18, right: 16, bottom: 28, left: 42 };
                            const chartWidth = width - padding.left - padding.right;
                            const chartHeight = height - padding.top - padding.bottom;
                            const maximum = analyticsMaximum;
                            const x = index => padding.left + (points.length === 1 ? chartWidth / 2 : index / (points.length - 1) * chartWidth) - scrollOffset;
                            const y = value => padding.top + chartHeight - value / maximum * chartHeight;
                            const yEnergy = value => padding.top + chartHeight - value / analyticsEnergyMaximum * chartHeight;
                            context.strokeStyle = '#d9ded5';
                            context.lineWidth = 1;
                            context.font = '10px Arial';
                            context.fillStyle = '#63736c';
                            [0, maximum / 2, maximum].forEach(value => {
                                const yPosition = y(value);
                                context.beginPath();
                                context.moveTo(padding.left, yPosition);
                                context.lineTo(width - padding.right, yPosition);
                                context.stroke();
                                context.fillText(Math.round(value), 8, yPosition + 3);
                            });
                            context.textAlign = 'right';
                            [0, analyticsEnergyMaximum / 2, analyticsEnergyMaximum].forEach(value => {
                                context.fillText(value.toFixed(3), width - 2, yEnergy(value) + 3);
                            });
                            context.textAlign = 'left';
                            context.strokeStyle = '#b23a48';
                            context.setLineDash([4, 4]);
                            context.beginPath();
                            context.moveTo(width - padding.right, padding.top);
                            context.lineTo(width - padding.right, height - padding.bottom);
                            context.stroke();
                            context.setLineDash([]);
                            context.fillStyle = '#b23a48';
                            context.font = '700 10px Arial';
                            context.fillText('LIVE', width - padding.right - 28, padding.top - 5);
                            const drawLine = (key, color) => {
                                const values = key === 'lux' ? analyticsSmoothedLux : analyticsSmoothedBrightness;
                                const valueY = y;
                                context.strokeStyle = color;
                                context.lineWidth = 2;
                                context.beginPath();
                                values.forEach((value, index) => {
                                    if (index === 0) {
                                        context.moveTo(x(index), valueY(value));
                                        return;
                                    }
                                    const middleX = (x(index - 1) + x(index)) / 2;
                                    context.quadraticCurveTo(middleX, valueY(values[index - 1]), x(index), valueY(value));
                                });
                                context.lineTo(width - padding.right, valueY(values[values.length - 1]));
                                context.stroke();
                            };
                            const drawEnergyLine = () => {
                                context.strokeStyle = '#4f6d7a';
                                context.lineWidth = 2;
                                context.beginPath();
                                analyticsSmoothedEnergy.forEach((value, index) => {
                                    if (index === 0) {
                                        context.moveTo(x(index), yEnergy(value));
                                        return;
                                    }
                                    const middleX = (x(index - 1) + x(index)) / 2;
                                    context.quadraticCurveTo(middleX, yEnergy(analyticsSmoothedEnergy[index - 1]), x(index), yEnergy(value));
                                });
                                context.lineTo(width - padding.right, yEnergy(analyticsSmoothedEnergy[analyticsSmoothedEnergy.length - 1]));
                                context.stroke();
                            };
                            drawLine('lux', '#2d6a4f');
                            drawLine('brightness', '#d87941');
                            drawEnergyLine();
                            points.forEach((point, index) => {
                                if (!point.motion) return;
                                context.fillStyle = '#b23a48';
                                context.beginPath();
                                context.arc(x(index), y(point.lux), 4, 0, Math.PI * 2);
                                context.fill();
                            });
                        }
                        let dashboardLoading = false;
                        async function loadDashboard() {
                            if (dashboardLoading) return;
                            dashboardLoading = true;
                            try {
                            const [health, metrics, readings, config, connection] = await Promise.all([fetch('/health').then(r => r.json()), fetch('/metrics').then(r => r.json()), fetch('/readings').then(r => r.json()), fetch('/config').then(r => r.json()), fetch('/connection').then(r => r.json())]);
                            const watcherFresh = connection.timestamp && (Date.now() / 1000 - connection.timestamp) < 6;
                            const portLive = Boolean(connection.port_connected || connection.bridge_running);
                            text('health', portLive ? 'ESP32 connected' : 'ESP32 disconnected');
                            document.getElementById('area-select').value = config.area || '';
                            text('model', metrics.model || '-');
                            text('rows', metrics.dataset_rows || '-');
                            text('accuracy', ((metrics.evaluations[metrics.model]?.balanced_accuracy || 0) * 100).toFixed(1) + '%');
                            text('cv-score', ((metrics.evaluations[metrics.model]?.cv_balanced_accuracy || 0) * 100).toFixed(1) + '%');
                            text('f1-score', ((metrics.evaluations[metrics.model]?.macro_f1 || 0) * 100).toFixed(1) + '%');
                            text('classes', 'Target classes: ' + (metrics.classes || []).join(', '));
                            const importance = metrics.feature_importance || [];
                            const maximum = importance[0]?.importance || 1;
                            document.getElementById('importance').innerHTML = importance.map(item => `<div class="importance-row"><span>${item.feature.replace('categorical__', '').replace('numerical__', '')}</span><div class="bar"><span style="width:${(item.importance / maximum * 100).toFixed(1)}%"></span></div><span>${(item.importance * 100).toFixed(1)}%</span></div>`).join('');
                            document.getElementById('comparison').innerHTML = '<table class="metric-table"><thead><tr><th>Model</th><th>Accuracy</th><th>CV balanced accuracy</th><th>Macro-F1</th></tr></thead><tbody>' + Object.entries(metrics.evaluations || {}).map(([name, value]) => `<tr><td>${name}${name === metrics.model ? ' (selected)' : ''}</td><td>${(value.accuracy * 100).toFixed(1)}%</td><td>${(value.cv_balanced_accuracy * 100).toFixed(1)}%</td><td>${(value.macro_f1 * 100).toFixed(1)}%</td></tr>`).join('') + '</tbody></table>';
                            const matrix = metrics.confusion_matrix || [];
                            const classes = metrics.classes || [];
                            document.getElementById('confusion').innerHTML = '<table class="metric-table"><thead><tr><th>Actual / Predicted</th>' + classes.map(item => `<th>${item}</th>`).join('') + '</tr></thead><tbody>' + matrix.map((row, index) => `<tr><th>${classes[index]}</th>${row.map(value => `<td>${value}</td>`).join('')}</tr>`).join('') + '</tbody></table>';
                            const latest = readings[readings.length - 1];
                            drawAnalytics(readings);
                            const fresh = latest && (Date.now() - Date.parse(latest.timestamp)) < 5000;
                            if (fresh) {
                                text('live-lux', latest.lux);
                                text('live-motion', latest.motion === '1' ? 'Detected' : 'Clear');
                                text('live-brightness', latest.brightness + '%');
                                text('health', portLive ? 'ESP32 connected / Model online' : 'ESP32 disconnected');
                                text('result', `Fresh ESP32 reading. ML confidence: ${(Number(latest.confidence) * 100).toFixed(1)}%`);
                            } else {
                                text('live-lux', '-');
                                text('live-motion', '-');
                                text('live-brightness', '-');
                                text('health', portLive ? 'ESP32 connected' : 'ESP32 disconnected');
                                text('result', portLive ? 'ESP32 connected. Waiting for the next sensor reading...' : 'No fresh model reading. Reconnecting...');
                            }
                            const recent = readings.slice(-8).reverse();
                            document.getElementById('readings').innerHTML = fresh ? '<table><thead><tr><th>Time</th><th>Area</th><th>Lux</th><th>Motion</th><th>Brightness</th><th>Energy (Wh)</th><th>Confidence</th></tr></thead><tbody>' + recent.map(r => `<tr><td>${r.timestamp?.replace('T', ' ').slice(0, 19) || '-'}</td><td>${r.area}</td><td>${r.lux}</td><td>${r.motion === '1' ? 'Yes' : 'No'}</td><td>${r.brightness}%</td><td>${r.energy_wh ? Number(r.energy_wh).toFixed(4) : '-'}</td><td>${(Number(r.confidence) * 100).toFixed(1)}%</td></tr>`).join('') + '</tbody></table>' : '<p class="empty">Model offline. No readings displayed.</p>';
                            } finally {
                                dashboardLoading = false;
                            }
                        }
                        document.getElementById('set-area').addEventListener('click', async () => {
                            const area = document.getElementById('area-select').value;
                            if (!area) return;
                            await fetch('/config', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({area})});
                            loadDashboard();
                        });
                        document.getElementById('reset-area').addEventListener('click', async () => {
                            await fetch('/config/reset', {method: 'POST'});
                            document.getElementById('area-select').value = '';
                            text('result', 'Area reset. Select an area to resume the ESP32.');
                        });
                        loadDashboard().catch(() => text('health', 'Unable to reach API'));
                        setInterval(() => loadDashboard().catch(() => text('health', 'ESP32 feed unavailable')), 1000);
                    </script>
                </body>
                </html>
                """
        )


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model": str(MODEL_PATH.name)})


@app.get("/connection")
def connection():
    live_connected = port_is_connected()
    live_reading_fresh = False
    if LIVE_READINGS_PATH.exists():
        try:
            with LIVE_READINGS_PATH.open(newline="", encoding="utf-8") as readings_file:
                latest_reading = list(csv.DictReader(readings_file))[-1]
            latest_timestamp = datetime.fromisoformat(latest_reading["timestamp"])
            live_reading_fresh = (datetime.now(timezone.utc) - latest_timestamp).total_seconds() < 5
        except (IndexError, KeyError, OSError, TypeError, ValueError):
            pass
    status = {
        "watcher": False,
        "port_connected": live_connected,
        "bridge_running": live_connected,
        "live_reading_fresh": live_reading_fresh,
        "api_running": True,
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }
    if CONNECTION_PATH.exists():
        try:
            saved = json.loads(CONNECTION_PATH.read_text(encoding="utf-8"))
            saved_timestamp = float(saved.get("timestamp", 0))
            heartbeat_fresh = datetime.now(timezone.utc).timestamp() - saved_timestamp < 6
            status["watcher"] = bool(saved.get("watcher", False)) and heartbeat_fresh
            status["bridge_running"] = (
                bool(saved.get("bridge_running", False)) and heartbeat_fresh
            ) or live_connected
            status["api_running"] = bool(saved.get("api_running", True)) and heartbeat_fresh
            if heartbeat_fresh:
                status["timestamp"] = saved_timestamp
        except (OSError, json.JSONDecodeError):
            pass
    return jsonify(status)


@app.get("/config")
def config():
    return jsonify({"area": load_area()})


@app.post("/config")
def set_config():
    data = request.get_json(silent=True) or {}
    area = data.get("area")
    if area not in VALID_AREAS:
        return jsonify({"status": "error", "message": "Select a valid area"}), 400
    save_area(area)
    return jsonify({"status": "success", "area": area})


@app.post("/config/reset")
def reset_config():
    save_area(DEFAULT_AREA)
    return jsonify({"status": "success", "area": DEFAULT_AREA})


@app.get("/metrics")
def metrics():
    metrics_path = Path(__file__).parent / "outputs" / "model_metrics.json"
    if not metrics_path.exists():
        return jsonify({"status": "error", "message": "Metrics are not available"}), 404
    return jsonify(json.loads(metrics_path.read_text(encoding="utf-8")))


@app.get("/readings")
def readings():
    if not LIVE_READINGS_PATH.exists():
        return jsonify([])
    with LIVE_READINGS_PATH.open(newline="", encoding="utf-8") as readings_file:
        rows = list(csv.DictReader(readings_file))
    return jsonify(rows[-50:])


@app.get("/readings.csv")
def readings_csv():
    columns = [
        "timestamp",
        "area",
        "hour",
        "day",
        "vehicles",
        "pedestrians",
        "lux",
        "motion",
        "traffic",
        "brightness",
        "confidence",
        "fallback_used",
        "energy_wh",
    ]
    output = io.StringIO(newline="")
    if LIVE_READINGS_PATH.exists():
        output.write(LIVE_READINGS_PATH.read_text(encoding="utf-8"))
    else:
        csv.writer(output).writerow(columns)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=smart-light-readings.csv"},
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Request body must be JSON"}), 400

    try:
        payload = {
            "area": data["area"],
            "hour": data["hour"],
            "day": data["day"],
            "vehicles": data["vehicles"],
            "pedestrians": data["pedestrians"],
            "lux": data["lux"],
            "motion": data["motion"],
            "traffic": data["traffic"],
        }
        result = predict_brightness(
            area=payload["area"],
            hour=payload["hour"],
            day=payload["day"],
            vehicles=payload["vehicles"],
            pedestrians=payload["pedestrians"],
            lux=payload["lux"],
            motion=payload["motion"],
            traffic=payload["traffic"],
        )
        save_reading(payload, result)
        return jsonify({"status": "success", **result})
    except KeyError as error:
        return jsonify({"status": "error", "message": f"Missing field: {error.args[0]}"}), 400
    except (TypeError, ValueError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"status": "error", "message": str(error)}), 500


@app.post("/sensor")
def sensor():
    """Accept a real ESP32 reading and return the ML brightness command."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Request body must be JSON"}), 400
    try:
        result = predict_brightness(
            area=data["area"],
            hour=data["hour"],
            day=data["day"],
            vehicles=data["vehicles"],
            pedestrians=data["pedestrians"],
            lux=data["lux"],
            motion=data["motion"],
            traffic=data["traffic"],
        )
        payload = {
            "area": data["area"],
            "hour": data["hour"],
            "day": data["day"],
            "vehicles": data["vehicles"],
            "pedestrians": data["pedestrians"],
            "lux": data["lux"],
            "motion": data["motion"],
            "traffic": data["traffic"],
        }
        save_reading(payload, result, LIVE_READINGS_PATH)
        return jsonify({"status": "success", **result})
    except KeyError as error:
        return jsonify({"status": "error", "message": f"Missing field: {error.args[0]}"}), 400
    except (TypeError, ValueError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400


if __name__ == "__main__":
    print("Smart Street Light ML Server running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
