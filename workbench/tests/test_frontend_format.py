"""A number on screen must name its unit and its denominator.

The served data mixes three units that all look like small decimals, and the previous UI printed
them through one helper:
  * nDCG is a fraction in [0,1];
  * recall@K is already a percentage (30.06 means 30.06 %);
  * the composition gap is a dimensionless fraction, not a percentage-point delta;
  * coverage is a fraction of a stated denominator.
An absent value must read as absent (em dash), never as 0, because 0 is a measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("format.ts", Path("/tmp/cir_frontend_format"))
SRC = json.dumps(str(MODULE))


def _fmt(expr: str) -> object:
    return support.run_json(f"import * as F from {SRC};\nconsole.log('__R__' + JSON.stringify({expr}));")


def test_the_three_units_render_differently() -> None:
    out = _fmt("""[
      F.fmtNdcg(0.2908449),
      F.fmtPercentValue(30.06458),
      F.fmtFractionPercent(0.9801765),
      F.fmtGap(0.5417445),
      F.fmtPp(30.06, 29.11),
    ]""")
    assert out[0] == "0.2908", out
    assert out[1] == "30.1%", out
    assert out[2] == "98.0%", out
    assert out[3] == "+0.542", out
    assert out[4] == "+0.95 pp", out


def test_a_negative_gap_keeps_its_sign() -> None:
    assert _fmt("F.fmtGap(-0.42)") == "-0.420"


def test_an_absent_value_is_absent_not_zero() -> None:
    out = _fmt("""[F.fmtNdcg(null), F.fmtPercentValue(undefined), F.fmtFractionPercent(null),
                   F.fmtGap(null), F.fmtPp(null, 1), F.fmtInt(NaN)]""")
    assert out == ["—", "—", "—", "—", "—", "—"], out


def test_a_rank_is_one_based_and_unscored_is_named() -> None:
    out = _fmt("[F.fmtRank(1), F.fmtRank(70016), F.fmtRank(null)]")
    assert out == ["#1", "#70,016", "unscored"], out


def test_a_metric_carries_its_denominator_and_coverage() -> None:
    out = _fmt("[F.fmtDenominator(5884, 6003), F.fmtDenominator(0, 4069), F.fmtDenominator(null, null)]")
    assert out[0] == "5,884 / 6,003 (98.0%)", out
    assert out[1] == "0 / 4,069 (0.0%)", out
    assert "—" in out[2], out
