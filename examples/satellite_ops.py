"""
Satellite Subsystem Anomaly Detection Example
Simulates real-time telemetry from a satellite's power subsystem and uses Aether-ML to detect anomalies.
"""

import time
import numpy as np
from core.engine import AetherEngine
from models.anomaly_detector import SpaceAnomalyDetector

def run_simulation():
    # 1. Initialize Aether Engine
    engine = AetherEngine(config={"mission": "autonomous_orbit_ops"})

    # 2. Register the Anomaly Detector
    detector = SpaceAnomalyDetector(sensitivity=0.95)
    
    # Pre-train on a normal baseline
    normal_telemetry = np.random.normal(12.0, 0.5, 500)  # Baseline: 12V power supply
    detector.train_on_baseline(normal_telemetry)
    
    engine.register_model("power_subsystem_watchdog", detector)

    # 3. Simulate Telemetry Stream
    print("\n--- Starting Mission Simulation ---\n")
    
    # A mix of normal and anomalous data packets
    telemetry_packets = [
        {"sensor_id": "voltage_bus_A", "values": [12.1, 11.9, 12.0], "timestamp": time.time()},
        {"sensor_id": "voltage_bus_A", "values": [12.2, 12.1, 12.1], "timestamp": time.time() + 1},
        {"sensor_id": "voltage_bus_A", "values": [15.5, 16.2, 15.8], "timestamp": time.time() + 2}, # ANOMALY
        {"sensor_id": "voltage_bus_A", "values": [11.8, 11.9, 12.0], "timestamp": time.time() + 3},
    ]

    results = engine.process_stream(telemetry_packets)

    # 4. Analyze Results
    for res in results:
        status = res["inferences"]["power_subsystem_watchdog"]["interpretation"]
        score = res["inferences"]["power_subsystem_watchdog"]["score"]
        print(f"Time: {res['timestamp']:.2f} | Sensor: {res['sensor_id']} | Status: {status} (Score: {score:.2f})")

if __name__ == "__main__":
    run_simulation()
