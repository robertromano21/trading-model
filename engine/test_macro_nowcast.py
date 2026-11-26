"""Tests for engine/macro_nowcast.py: outside-forecast separation, point-in-time Data, panel loader, refresh lock."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import fred_macro as FM
import macro_nowcast as MN
import outside_forecasts as OF

ENGINE = Path(__file__).resolve().parent


class SeparationTests(unittest.TestCase):
    """Hard rule: outside projections (consensus, GDPNow, NY Fed, SPF, Cleveland Fed) are never model inputs."""

    def test_feature_series_has_no_outside_ids(self):
        bases = {i.split(":")[0] for i in MN.FEATURE_SERIES} | {t["src"].split(":")[0] for t in MN.TARGETS.values()}
        self.assertFalse(bases & set(FM.OUTSIDE_FORECASTS))
        self.assertFalse(bases & set(OF.OUTSIDE_IDS))
        for sid in bases:
            self.assertNotEqual(FM.meta(sid)[3], "outside_forecast", sid)

    def test_guard_rejects_outside_ids(self):
        for bad in ("GDPNOW", "CONSENSUS", "SPF_RGDP", "CLEVELAND_NOWCAST:pch", "NYFED_NOWCAST"):
            with self.assertRaises(RuntimeError):
                MN.assert_no_outside_inputs({"CPIAUCSL", bad})

    def test_data_refuses_outside_feature(self):
        with mock.patch.object(MN, "FEATURE_SERIES", MN.FEATURE_SERIES | {"GDPNOW"}):
            with self.assertRaises(RuntimeError):
                MN.Data(frames={})

    def test_model_code_never_reads_outside_module_before_publishing_numbers(self):
        """outside_forecasts may only be imported inside publish()/research() after our numbers exist."""
        tree = ast.parse((ENGINE / "macro_nowcast.py").read_text(encoding="utf-8"))
        allowed = {"assert_no_outside_inputs", "publish", "research", "refresh"}
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                    if "outside_forecasts" in names:
                        self.assertIn(fn.name, allowed, f"{fn.name} imports outside_forecasts")
        src = (ENGINE / "macro_nowcast.py").read_text(encoding="utf-8")
        pub = src[src.index("def publish("):src.index("def _json_default")]
        self.assertLess(pub.index("nowcast_one("), pub.index("import outside_forecasts"))
        for node in tree.body:   # no module-level import of the outside module
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertNotIn("outside_forecasts", [a.name for a in node.names] + [getattr(node, "module", "") or ""])


class DataTests(unittest.TestCase):
    def test_known_hides_unreleased(self):
        idx = pd.to_datetime(["2020-01-01", "2020-02-01"])
        frames = {"CPIAUCSL:pch": pd.DataFrame({"value": [0.2, 0.4],
                                                "available": pd.to_datetime(["2020-02-13", "2020-03-11"])}, index=idx)}
        D = MN.Data(frames=frames)
        self.assertEqual(D.latest("CPIAUCSL:pch", "2020-03-10")[0], 0.2)
        self.assertEqual(D.latest("CPIAUCSL:pch", "2020-03-11")[0], 0.4)
        self.assertTrue(np.isnan(D.latest("CPIAUCSL:pch", "2020-02-12")[0]))
        self.assertTrue(np.isnan(D.value_at("CPIAUCSL:pch", "2020-02-01", "2020-03-10")))


class LoaderTests(unittest.TestCase):
    def test_load_point_in_time_has_asof(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nc.csv"
            p.write_text(",NC_CPI_MOM,NC_CPI_MOM_VS_LAST\n2026-01-02,0.2,-0.1\n2026-01-05,0.25,-0.05\n")
            f = MN.load_point_in_time(p)
        self.assertEqual(list(f["asof"].dt.strftime("%Y-%m-%d")), ["2026-01-02", "2026-01-05"])
        self.assertIn("NC_CPI_MOM", f.columns)
        self.assertIsNone(MN.load_point_in_time(Path(d) / "missing.csv"))

    def test_refresh_clears_lock_even_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "nowcast_refresh.lock").write_text("{}")
            (out / "nowcast_daily.csv").write_text("x\n")
            with mock.patch.object(MN, "OUT_DIR", out), mock.patch.object(MN, "PANEL_PATH", out / "nowcast_daily.csv"), \
                    mock.patch.object(FM, "update_all", return_value={"mode": "t", "errors": {}}), \
                    mock.patch.object(OF, "update_all", return_value={}), \
                    mock.patch.object(MN, "publish", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    MN.refresh(log=None)
            self.assertFalse((out / "nowcast_refresh.lock").exists())


@unittest.skipUnless(MN.LATEST_PATH.parent.exists() and (FM.FRED_LONG_DIR / "CPIAUCSL.csv").exists(),
                     "needs local FRED data")
class PublishIsolationTests(unittest.TestCase):
    """Changing the outside estimates must not change a single one of our numbers."""

    def test_outside_numbers_do_not_move_nowcasts(self):
        D = MN.Data()

        def run(val):
            fake = {t: {"consensus": {"value": val, "period": None}} for t in MN.TARGETS}
            fake["_note"] = "test"
            with mock.patch.object(OF, "latest_for_display", return_value=fake):
                return MN.publish(write=False, D=D, asof="2026-06-30")
        a, b = run(-99.0), run(99.0)
        self.assertEqual([n.get("nowcast") for n in a["nowcasts"]], [n.get("nowcast") for n in b["nowcasts"]])
        self.assertEqual([n.get("band80") for n in a["nowcasts"]], [n.get("band80") for n in b["nowcasts"]])


if __name__ == "__main__":
    unittest.main()
