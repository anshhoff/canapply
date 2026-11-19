"""Eval gate for CI: re-judge the frozen 120-posting sample with the current eligibility code.

It needs no database and no API key, because the requirements readings are frozen in the fixture. What it
protects is the decision logic: if someone changes a threshold or a rule and accuracy drops, the build fails.
Floors sit a little under the measured values (accuracy 0.942, can-apply recall 1.0 on 18 Nov 2026).
"""
import json
from pathlib import Path

from canapply.eligibility import Profile, judge
from canapply.requirements import Requirements

FIXTURE = json.loads((Path(__file__).parent.parent / "evals" / "fixtures" / "reading_pool.json").read_text())


def _scores():
    profile = Profile.model_validate(FIXTURE["profile"])
    pred = {i["id"]: judge(Requirements.model_validate(i["requirements"]), profile).bucket for i in FIXTURE["items"]}
    gold = {i["id"]: i["gold"] for i in FIXTURE["items"]}
    accuracy = sum(pred[k] == gold[k] for k in gold) / len(gold)
    said = [k for k in gold if pred[k] == "can_apply"]
    real = [k for k in gold if gold[k] == "can_apply"]
    hits = [k for k in said if gold[k] == "can_apply"]
    return accuracy, len(hits) / len(said), len(hits) / len(real)


def test_fixture_is_the_full_sample():
    assert len(FIXTURE["items"]) == 120


def test_verdict_accuracy_does_not_regress():
    accuracy, _, _ = _scores()
    assert accuracy >= 0.92, f"verdict accuracy fell to {accuracy:.3f}"


def test_can_apply_precision_and_recall_do_not_regress():
    _, precision, recall = _scores()
    assert recall >= 0.85, f"can-apply recall fell to {recall:.3f}"
    assert precision >= 0.70, f"can-apply precision fell to {precision:.3f}"
