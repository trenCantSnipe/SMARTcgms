from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from cgm_intelligence import (
    AlertDecisionEngine,
    DecisionLogger,
    GlucoseReading,
    ewma_smooth,
    is_probable_compression_low,
    predict_low_within_minutes,
    process_new_reading,
)


class CGMIntelligenceTests(unittest.TestCase):
    def test_ewma_smooth(self):
        out = ewma_smooth([100, 90, 80], alpha=0.5)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[-1], 87.5)

    def test_compression_low_detected(self):
        base = datetime(2025, 1, 1, 1, 0)
        readings = [
            GlucoseReading(base + timedelta(minutes=i * 5), g, motion=0.01)
            for i, g in enumerate([110, 93, 76, 60])
        ]
        self.assertTrue(is_probable_compression_low(readings))

    def test_predictive_low(self):
        base = datetime(2025, 1, 1, 12, 0)
        readings = [
            GlucoseReading(base + timedelta(minutes=0), 100),
            GlucoseReading(base + timedelta(minutes=5), 95),
            GlucoseReading(base + timedelta(minutes=10), 90, trend_mg_dl_per_min=-1.0),
        ]
        self.assertTrue(predict_low_within_minutes(readings))

    def test_process_new_reading_alert_callback(self):
        base = datetime(2025, 1, 1, 12, 0)
        readings = [
            GlucoseReading(base + timedelta(minutes=0), 100),
            GlucoseReading(base + timedelta(minutes=5), 85),
            GlucoseReading(base + timedelta(minutes=10), 72, trend_mg_dl_per_min=-0.3),
        ]
        engine = AlertDecisionEngine(cooldown_minutes=1)
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(str(Path(tmp) / "decisions.log"))
            reasons = []
            result = process_new_reading(readings, engine, logger, reasons.append)
            self.assertTrue(result.should_alert)
            self.assertEqual(reasons, ["Predictive low in next 30 minutes"])


if __name__ == "__main__":
    unittest.main()
