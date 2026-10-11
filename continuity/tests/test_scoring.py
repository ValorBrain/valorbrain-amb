import datetime as dt

from continuity_bench.reader import parse_value
from continuity_bench.runner import render
from continuity_bench.scoring import context_ok, leaked, summarize, wilson


def test_context_ok_requires_gold_and_no_forbidden():
    assert context_ok("policy HERON-902", "HERON-902", ["KESTREL-411"])
    assert not context_ok("HERON-902 and KESTREL-411", "HERON-902", ["KESTREL-411"])
    assert not context_ok("nothing here", "HERON-902", [])
    assert context_ok("nothing here", "NONE", ["KESTREL-411"])
    assert not context_ok("old KESTREL-411", "NONE", ["KESTREL-411"])


def test_leak():
    assert leaked("my locker code is LARK-118", ["LARK-118"])
    assert not leaked("budget FINCH-224", ["LARK-118"])


def test_parse_value():
    assert parse_value('{"value": "heron-902", "reason": "x"}') == "HERON-902"
    assert parse_value('```json\n{"value": "NONE", "reason": "expired"}\n```') == "NONE"
    assert parse_value("The answer is WREN-733.") == "WREN-733"
    assert parse_value("I cannot tell") == "UNPARSED"


def test_wilson_bounds():
    lo, hi = wilson(10, 10)
    assert 0.69 < lo < 0.73 and hi == 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_render_dates_relative_to_run_day():
    assert render("from {date:-40} until {date:90}", dt.date(2026, 10, 11)) == "from 2026-09-01 until 2027-01-09"


def test_summarize_pools_runs_and_counts_leaks():
    rows = []
    for run in (1, 2):
        rows.append({"run": run, "system": "rag", "category": "permission", "variant": "private-other",
                     "forbidden": ["LARK-118"], "leak": True, "context_ok": False, "correct": run == 2})
        rows.append({"run": run, "system": "rag", "category": "permission", "variant": "shared",
                     "forbidden": [], "leak": False, "context_ok": True, "correct": True})
    s = summarize(rows)
    cell = s["rag"]["permission"]["*"]
    assert cell["n"] == 4 and cell["runs"] == 2
    assert cell["correct"]["k"] == 3 and cell["correct"]["per_run"] == [0.5, 1.0]
    assert cell["leak"] == {"k": 2, "n": 2, "rate": 1.0}
