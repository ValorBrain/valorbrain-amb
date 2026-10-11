import json
import re
from collections import Counter
from pathlib import Path

from continuity_bench import dataset as ds

CODE = re.compile(r"\b[A-Z]{3,12}-\d{3}\b")


def test_same_seed_same_timeline():
    assert ds.generate(seed=7, per_cell=4) == ds.generate(seed=7, per_cell=4)
    assert ds.generate(seed=7, per_cell=4)["sha256"] != ds.generate(seed=8, per_cell=4)["sha256"]


def test_committed_dataset_matches_generator():
    committed = json.loads((Path(__file__).parents[1] / "data" / "continuity-v1.json").read_text())
    assert committed == ds.generate(seed=committed["seed"], per_cell=committed["per_cell"])


def test_every_written_code_is_unique():
    data = ds.generate(per_cell=10)
    written = [c for s in data["steps"] if s["op"] in ("doc", "fact", "handoff", "memory") for c in CODE.findall(s["text"])]
    assert written and len(written) == len(set(written))


def test_questions_point_at_written_codes_and_never_leak_them():
    data = ds.generate(per_cell=10)
    written = {c for s in data["steps"] if "text" in s for c in CODE.findall(s["text"])}
    for q in ds.questions(data):
        assert not CODE.search(q["question"]), q["id"]
        assert q["gold"] == "NONE" or q["gold"] in written, q["id"]
        assert q["gold"] not in q["forbidden"], q["id"]
        assert set(q["forbidden"]) <= written, q["id"]


def test_rule_and_decision_subjects_do_not_collide():
    data = ds.generate(per_cell=10)
    keys = [s["key"] for s in data["steps"] if s["op"] == "fact"]
    assert len(keys) == len(set(keys))
    asks = Counter(q["question"] for q in ds.questions(data) if q["category"] in ("supersession", "expiry"))
    assert max(asks.values()) == 1


def test_cell_sizes():
    data = ds.generate(per_cell=10)
    cells = Counter((q["category"], q["variant"]) for q in ds.questions(data))
    assert all(n == 10 for n in cells.values()), cells
    assert {c for c, _ in cells} == {"handoff", "supersession", "expiry", "permission"}


def test_superseded_versions_are_written_before_their_successor():
    data = ds.generate(per_cell=6)
    seen: set[str] = set()
    for s in data["steps"]:
        if s["op"] == "doc":
            if s["supersedes"]:
                assert s["supersedes"] in seen
            seen.add(s["path"])
