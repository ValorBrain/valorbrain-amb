"""README guard: one metric per line, and every published number matches its run file.

Run with `python3 -m unittest discover -s tests -v` (standard library only).

What it rejects:
- a line (or table) that shows a percentage without exactly one metric label, or with the
  ambiguous word "accuracy" (AMB's `accuracy` field is avg-score; "accuracy" next to a
  number has been read as binary more than once);
- "X% (a/b)" where a/b is not X% (the classic avg-score-next-to-binary-count mix);
- a run block, decomposition table, per-category table or scoreboard row whose number differs
  from what `scripts/beam_summary.py` recomputes from `results/` or `results-manifest.json`.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import beam_summary  # noqa: E402

README = ROOT / "README.md"
APPENDIX_HEADING = "## Appendix"

LABELS = {
    "avg-score": re.compile(r"avg-score", re.I),
    "binary": re.compile(r"\bbinary\b", re.I),
    "recall@10": re.compile(r"recall@10", re.I),
    "answer accuracy": re.compile(r"answer accuracy", re.I),
}
AMBIGUOUS = re.compile(r"(?<!answer )\baccuracy\b", re.I)
CI = re.compile(r"95% CI")
PERCENT = re.compile(r"\d+(?:\.\d+)?%")
PCT_WITH_COUNT = re.compile(r"(\d+(?:\.\d+)?)%[^()\n|]*?\((\d+)/(\d+)\)")
FILE_REF = re.compile(r"`(results/[^`]+)`")


def scoped_lines(text: str) -> list[str]:
    """README lines before the upstream appendix, with fenced code blocks blanked."""
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith(APPENDIX_HEADING):
            break
        if line.strip().startswith("```"):
            fenced = not fenced
            out.append("")
            continue
        out.append("" if fenced else line)
    return out


def labels_in(line: str) -> set[str]:
    return {name for name, rx in LABELS.items() if rx.search(line)}


def is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def mixing_errors(lines: list[str]) -> list[str]:
    """Every line or table row with a metric percentage must carry exactly one label."""
    errors: list[str] = []
    header: str | None = None
    for no, line in enumerate(lines, 1):
        if not is_table_row(line):
            header = None
        elif header is None:
            header = line
        has_pct = bool(PERCENT.search(CI.sub("", line)))
        if not has_pct:
            continue
        if AMBIGUOUS.search(line):
            errors.append(f"line {no}: ambiguous 'accuracy' next to a number: {line.strip()}")
            continue
        scope = header if is_table_row(line) and header is not None else line
        found = labels_in(scope)
        if len(found) != 1:
            what = "table header" if is_table_row(line) else "line"
            errors.append(f"line {no}: {what} must name exactly one metric, found {sorted(found) or 'none'}: {line.strip()}")
    return errors


def count_errors(lines: list[str]) -> list[str]:
    errors: list[str] = []
    for no, line in enumerate(lines, 1):
        for m in PCT_WITH_COUNT.finditer(line):
            shown, a, b = float(m.group(1)), int(m.group(2)), int(m.group(3))
            if beam_summary.pct(a / b) != f"{shown:.1f}%":
                errors.append(f"line {no}: {shown}% shown next to {a}/{b} ({beam_summary.pct(a / b)}): {line.strip()}")
    return errors


def sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.startswith("### "):
            out.append((line[4:].strip(), []))
        elif out:
            out[-1][1].append(line)
    return out


def table_rows(body: list[str]) -> list[list[str]]:
    rows = [l for l in body if is_table_row(l)]
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    return [c for c in cells if not all(set(x) <= set("-: ") for x in c)]


def pct_value(cell: str) -> str:
    m = PERCENT.search(cell)
    if not m:
        raise AssertionError(f"no percentage in cell {cell!r}")
    return f"{float(m.group(0)[:-1]):.1f}%"


def pp(value: float) -> str:
    return f"{value:+.1f} pp"


OLD_README_LINES = [
    "This is the reader that scored **70.9% on BEAM-100K (312/400)** — 5 of 10 category wins vs Hindsight.",
    "Headline: **80.8% on BEAM-100K** (323/400)",
    "**BEAM-10M**: **51.2%** runner accuracy (112/200 binary) on the 10M-token split.",
    "58% of our total improvement came from reader-agnostic memory engineering.",
]


class CheckerRejectsMixedReadme(unittest.TestCase):
    def test_each_old_line_is_rejected(self) -> None:
        for line in OLD_README_LINES:
            with self.subTest(line=line):
                self.assertTrue(mixing_errors([line]) or count_errors([line]), line)

    def test_a_clean_line_passes(self) -> None:
        line = "- binary: 80.8% (323/400) (95% CI 78.0%–83.5%)"
        self.assertEqual(mixing_errors([line]) + count_errors([line]), [])

    def test_table_without_metric_in_header_is_rejected(self) -> None:
        table = ["| Category | Score |", "|---|---:|", "| Abstention | 65.0% |"]
        self.assertTrue(mixing_errors(table))

    def test_table_with_two_metrics_in_header_is_rejected(self) -> None:
        table = ["| Run | avg-score | binary |", "|---|---:|---:|", "| x | 75.5% | 80.8% |"]
        self.assertTrue(mixing_errors(table))


class ReadmeMatchesResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lines = scoped_lines(README.read_text(encoding="utf-8"))
        cls.sections = sections(cls.lines)
        cls.cache: dict[str, beam_summary.RunSummary] = {}

    def summary(self, rel: str) -> beam_summary.RunSummary:
        if rel not in self.cache:
            path = ROOT / rel
            self.assertTrue(path.exists(), f"README cites missing file {rel}")
            self.cache[rel] = beam_summary.summarize(path)
        return self.cache[rel]

    def section(self, prefix: str) -> list[str]:
        for title, body in self.sections:
            if title.startswith(prefix):
                return body
        self.fail(f"README has no section starting with {prefix!r}")

    def test_one_metric_per_line(self) -> None:
        self.assertEqual(mixing_errors(self.lines), [])

    def test_counts_match_percentages(self) -> None:
        self.assertEqual(count_errors(self.lines), [])

    def test_every_results_file_is_cited(self) -> None:
        cited = {m for line in self.lines for m in FILE_REF.findall(line)}
        on_disk = {f.relative_to(ROOT).as_posix() for f in beam_summary.run_files()}
        self.assertEqual(on_disk - cited, set(), "run files in results/ that the README never cites")

    def test_run_blocks_match_their_file(self) -> None:
        checked = 0
        for title, body in self.sections:
            files = [m for line in body if line.startswith("- File:") for m in FILE_REF.findall(line)]
            if not files:
                continue
            s = self.summary(files[0])
            for line in body:
                if line.startswith("- avg-score:"):
                    expected = (f"- avg-score: {beam_summary.pct(s.avg_score)} (95% CI {beam_summary.pct(s.avg_score_ci[0])}"
                                f"–{beam_summary.pct(s.avg_score_ci[1])}; {s.scored_queries} scored questions)")
                    self.assertEqual(line.strip(), expected, title)
                    checked += 1
                if line.startswith("- binary:"):
                    expected = (f"- binary: {beam_summary.pct(s.binary)} ({s.binary_correct}/{s.total_queries}) "
                                f"(95% CI {beam_summary.pct(s.binary_ci[0])}–{beam_summary.pct(s.binary_ci[1])})")
                    self.assertEqual(line.strip(), expected, title)
                    checked += 1
                if line.startswith("- Dataset:"):
                    self.assertIn(f"{s.conversations} conversations", line, title)
                    self.assertIn(f"{s.total_queries} questions", line, title)
                if line.startswith("- Judge:"):
                    self.assertIn(f"`{s.judge}`", line, title)
                if line.startswith("- Reader:"):
                    self.assertIn(f"`{s.reader}`", line, title)
        self.assertGreaterEqual(checked, 6)

    def test_decomposition_adds_up(self) -> None:
        rows = [r for r in table_rows(self.section("Where the BEAM-100K headline came from")) if r[0] != "Step"]
        self.assertEqual(len(rows), 8, "two 4-row tables: avg-score then binary")
        avg_rows, bin_rows = rows[:4], rows[4:]
        shown = []
        for step, file_cell, _reader, value, _change in avg_rows[:3]:
            s = self.summary(FILE_REF.findall(file_cell)[0])
            self.assertEqual(pct_value(value), beam_summary.pct(s.avg_score), step)
            shown.append(float(pct_value(value)[:-1]))
        self.assertEqual(avg_rows[1][4], pp(shown[1] - shown[0]))
        self.assertEqual(avg_rows[2][4], pp(shown[2] - shown[1]))
        self.assertEqual(avg_rows[3][4], pp(shown[2] - shown[0]))
        counts = []
        for step, file_cell, _reader, value, gained in bin_rows[:3]:
            s = self.summary(FILE_REF.findall(file_cell)[0])
            self.assertEqual(value, f"{beam_summary.pct(s.binary)} ({s.binary_correct}/{s.total_queries})", step)
            counts.append(s.binary_correct)
        self.assertEqual(bin_rows[1][4], f"+{counts[1] - counts[0]}")
        self.assertEqual(bin_rows[2][4], f"+{counts[2] - counts[1]}")
        self.assertEqual(bin_rows[3][4], f"+{counts[2] - counts[0]}")

    def test_per_category_table_matches_files(self) -> None:
        body = self.section("Per category")
        files = [m for line in body if line.startswith("- Files:") for m in FILE_REF.findall(line)]
        self.assertEqual(len(files), 2)
        left, right = (self.summary(f) for f in files)
        names = {name.lower(): name for name in left.categories}
        rows = [r for r in table_rows(body) if r[0] != "Category"]
        self.assertEqual(len(rows), len(left.categories))
        for category, a, b, delta in rows:
            key = names[re.sub(r"[ -]", "_", category.lower())]
            va, vb = left.categories[key]["binary"], right.categories[key]["binary"]
            self.assertEqual(pct_value(a), beam_summary.pct(va), category)
            self.assertEqual(pct_value(b), beam_summary.pct(vb), category)
            self.assertEqual(delta, f"{(float(pct_value(b)[:-1]) - float(pct_value(a)[:-1])):+.1f}".replace("+0.0", "0.0"), category)

    def test_scoreboard_context_matches_manifest(self) -> None:
        manifest = json.loads((ROOT / "results-manifest.json").read_text(encoding="utf-8"))
        rows = [r for r in table_rows(self.section("Context from the AMB scoreboard")) if r[0] != "Split"]
        self.assertTrue(rows)
        for split, system, _reader, _judge, value in rows:
            mode = re.search(r"\(([^)]+)\)", system).group(1)
            entry = next(e for e in manifest if e["dataset"] == "beam" and e["memory"] == "hindsight"
                         and e["split"] == split.lower() and e["mode"] == mode)
            self.assertEqual(pct_value(value), beam_summary.pct(entry["accuracy"]), f"{split} {system}")


if __name__ == "__main__":
    unittest.main()
