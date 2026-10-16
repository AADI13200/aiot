# Wokwi Smart Street Light Simulation

1. Open `wokwi/diagram.json` in Wokwi.
2. Use `wokwi/wokwi_simulation.ino` as the sketch.
3. Start the simulation and open the Serial Monitor at `115200` baud.
4. Adjust the BH1750 lux control and PIR motion control.
5. Send commands such as `BRIGHTNESS:70` to test the lamp output.

The simulated ESP32 emits the same sensor protocol used by `.venv/smart_light.py`:

```text
LUX:25.00,MOTION:1
```
