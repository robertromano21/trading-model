"""Offline tests for engine/outside_forecasts.py (display-only outside estimates)."""
import io
import unittest
import zipfile

import pandas as pd

import outside_forecasts as OF


def _cleveland_doc():
    cats = [{"label": "10/01"}, {"label": "10/15"}, {"label": "CPI Oct"}, {"label": "11/14"}]
    return [{"chart": {"subcaption": "2026-10"},
             "categories": [{"category": cats}],
             "dataset": [{"seriesname": "CPI Inflation", "data": [{"value": "0.20"}, {"value": "0.25"}, {}, {"value": "0.31"}]},
                         {"seriesname": "Something else", "data": [{"value": "9"}]}]}]


class ClevelandTests(unittest.TestCase):
    def test_parse_and_final_before_release(self):
        f = OF.parse_cleveland(_cleveland_doc())
        self.assertEqual(set(f["target"]), {"CPI_MOM"})
        self.assertEqual(len(f), 3)
        self.assertEqual(f["period"].iloc[0], pd.Timestamp("2026-10-01"))
        final = f[f["finalBeforeRelease"]]
        self.assertEqual(len(final), 1)
        self.assertEqual(final["value"].iloc[0], 0.25)            # last value before the CPI Oct release line
        self.assertEqual(final["date"].iloc[0], pd.Timestamp("2026-10-15"))

    def test_bad_entries_ignored(self):
        self.assertTrue(OF.parse_cleveland([{"chart": {}}, {}]).empty)


class XlsxTests(unittest.TestCase):
    def test_stdlib_xlsx_reader(self):
        ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        shared = f'<sst {ns}><si><t>YEAR</t></si><si><t>DRGDP2</t></si></sst>'
        sheet = (f'<worksheet {ns}><sheetData>'
                 '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                 '<row r="2"><c r="A2"><v>2026</v></c><c r="B2"><v>2.3</v></c></row>'
                 '</sheetData></worksheet>')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("xl/sharedStrings.xml", shared)
            z.writestr("xl/worksheets/sheet1.xml", sheet)
        f = OF.read_xlsx_first_sheet(buf.getvalue())
        self.assertEqual(list(f.columns), ["YEAR", "DRGDP2"])
        self.assertEqual(f.iloc[0].tolist(), ["2026", "2.3"])

    def test_outside_ids_cover_named_sources(self):
        for sid in ("GDPNOW", "CONSENSUS", "SPF_RGDP", "CLEVELAND_NOWCAST", "NYFED_NOWCAST"):
            self.assertIn(sid, OF.OUTSIDE_IDS)


if __name__ == "__main__":
    unittest.main()
