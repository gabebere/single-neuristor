from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import neuristor.model as model
from neuristor.model import HysteresisArray, YuanhangResistParams
from neuristor.sample_library import (
    compute_rt_fit_metrics,
    list_samples,
    load_experimental_rt_path,
    params_from_dict,
)


class HysteresisReversalTests(unittest.TestCase):
    def setUp(self) -> None:
        model._TORCH_HYSTERESIS_AVAILABLE = False
        self.params = YuanhangResistParams(reversal_threshold_K=0.01)

    def _run_path(self, values: np.ndarray) -> HysteresisArray:
        h = HysteresisArray(
            self.params,
            size=1,
            start_branch="insulator",
        )
        h.initialize(np.asarray([values[0]], dtype=np.float32))
        for value in values[1:]:
            h.evaluate(np.asarray([value], dtype=np.float32))
        return h

    def test_accumulates_subthreshold_steps(self) -> None:
        h = HysteresisArray(
            self.params,
            size=1,
            start_branch="insulator",
        )
        h.initialize(np.asarray([325.0], dtype=np.float32))
        for value in (325.004, 325.008):
            h.evaluate(np.asarray([value], dtype=np.float32))
            self.assertAlmostEqual(float(h.T_last[0]), 325.0, places=5)
            self.assertEqual(float(h.reversed[0]), 0.0)
        h.evaluate(np.asarray([325.012], dtype=np.float32))
        self.assertAlmostEqual(float(h.T_last[0]), 325.012, places=4)
        self.assertEqual(float(h.reversed[0]), 0.0)

    def test_records_accumulated_detection_point(self) -> None:
        path = np.asarray([325.0, 330.0, 334.0, 333.995, 333.989], dtype=float)
        h = self._run_path(path)
        self.assertEqual(float(h.delta[0]), -1.0)
        self.assertAlmostEqual(float(h.Tr[0]), 333.989, places=3)

    def test_saturated_reversal_endpoint_stays_finite(self) -> None:
        params = YuanhangResistParams(reversal_threshold_K=0.01, beta=1.5, w=7.0, Tc_K=333.0)
        h = HysteresisArray(params, size=1, start_branch="insulator")
        h.initialize(np.asarray([305.0], dtype=np.float32))
        tpr = h._solve_Tpr(
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([305.0], dtype=np.float32),
        )
        self.assertTrue(np.isfinite(float(tpr[0])))

    def test_matches_upstream_reference_trace(self) -> None:
        if model.torch is None:
            self.skipTest("Torch is required for the float32 reference replay.")
        model._TORCH_HYSTERESIS_AVAILABLE = True
        path = np.asarray([324.9, 325.0, 360.0, 325.0, 360.0, 330.0, 360.0, 335.0, 360.0])
        expected_ohm = np.asarray(
            [
                52101.7695,
                51843.3750,
                1286.3866,
                51843.3750,
                1286.3185,
                37773.2617,
                1286.3191,
                9628.7910,
                1286.3302,
            ]
        )
        h = HysteresisArray(
            self.params,
            size=1,
            start_branch="insulator",
        )
        h.initialize(np.asarray([path[0]], dtype=np.float32))
        actual = []
        for value in path:
            resistance, _ = h.evaluate(np.asarray([value], dtype=np.float32))
            actual.append(float(resistance[0]))
        np.testing.assert_allclose(actual, expected_ohm, rtol=2e-6, atol=0.02)

    def test_saved_sample_fits_replay_to_stored_metrics(self) -> None:
        checked = 0
        for sample in list_samples():
            source_path = Path(str(sample.get("source_path", "")))
            if not source_path.is_file():
                continue
            stored_rmse = float(sample["fit_metrics"]["rmse_log10"])
            df = load_experimental_rt_path(source_path)
            metrics, _, _ = compute_rt_fit_metrics(
                df,
                params_from_dict(sample["resist_params"]),
                start_branch=str(sample["start_branch"]),
            )
            self.assertAlmostEqual(metrics["rmse_log10"], stored_rmse, places=7, msg=str(sample["sample_id"]))
            checked += 1
        self.assertGreaterEqual(checked, 6)


if __name__ == "__main__":
    unittest.main()
