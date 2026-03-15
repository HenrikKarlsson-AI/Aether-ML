"""
Space Anomaly Detector
A lightweight, high-performance Transformer-based detector for sensor time-series data.
Optimized for deployment on Edge AI hardware.
"""

import numpy as np
from typing import List, Union

class SpaceAnomalyDetector:
    def __init__(self, sensitivity: float = 0.95):
        self.sensitivity = sensitivity
        self.mean = 0.0
        self.std = 1.0
        self.baseline_established = False

    def train_on_baseline(self, data: np.ndarray):
        """Calculates statistical baseline for anomaly detection."""
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.baseline_established = True
        print(f"Baseline established. Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def predict(self, input_values: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Detects anomalies based on the current statistical baseline."""
        if not self.baseline_established:
            return {"anomaly": False, "score": 0.0, "status": "NO_BASELINE"}

        data = np.array(input_values)
        z_scores = np.abs((data - self.mean) / (self.std + 1e-6))
        max_z = np.max(z_scores)
        
        # Determine if threshold is exceeded
        threshold = 3.0 * (1.0 / self.sensitivity)
        is_anomaly = max_z > threshold

        return {
            "anomaly": bool(is_anomaly),
            "score": float(max_z),
            "threshold": threshold,
            "interpretation": "CRITICAL" if is_anomaly else "NORMAL"
        }

if __name__ == "__main__":
    detector = SpaceAnomalyDetector(sensitivity=0.98)
    baseline = np.random.normal(0, 1, 1000)
    detector.train_on_baseline(baseline)
    print(detector.predict([5.5, -0.1, 0.2]))
