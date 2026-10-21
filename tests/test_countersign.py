import asyncio
import io

import numpy as np
import pytest

from countersign.evaluate import at_budget, rules_score, run
from countersign.features import NAMES, FeatureEngine, haversine_km, replay
from countersign.gen import DAY, Txn, generate
from countersign.model import Model, scores, train
from countersign.stream import Dedupe, Scorer, run_stream


@pytest.fixture(scope="module")
def small():
    return generate(seed=3, cards=300, days=10, merchants=200)


@pytest.fixture(scope="module")
def trained():
    return run(seed=7)


def txn(card=1, t=1000.0, amount=20.0, merchant=1, category="grocery", lat=39.74, lon=-104.99, device=None, tid="t1"):
    return Txn(tid, card, t, t + 1, amount, merchant, category, "Denver" if lat else None, lat, lon, device, False, "")


# ---- data ------------------------------------------------------------------


def test_generation_is_deterministic():
    a = generate(seed=5, cards=50, days=3)
    b = generate(seed=5, cards=50, days=3)
    assert [e.to_dict() for e in a.events] == [e.to_dict() for e in b.events]


def test_stream_is_messy_like_real_traffic(small):
    ids = [e.txn_id for e in small.events]
    assert len(ids) > len(set(ids)), "expected duplicate deliveries"
    assert any(e.arrival_time - e.event_time > 30 for e in small.events), "expected late events"
    arrivals = [e.arrival_time for e in small.events]
    assert arrivals == sorted(arrivals)
    events = [e.event_time for e in small.events]
    assert events != sorted(events), "expected out-of-order event times"


# ---- features ----------------------------------------------------------------


def test_training_features_are_exactly_the_served_features(small, trained):
    """The point of the project: no training/serving skew, checked bit for bit."""
    model, _ = trained
    training = {e.txn_id: x for e, x in replay(small.events, small.homes, "arrival")}
    scorer = Scorer(model, small.homes)
    checked = 0
    for e in small.events:
        if not scorer.dedupe.first_time(e.txn_id):
            continue
        served = scorer.engine.features(e)
        scorer.engine.update(e)
        assert served == training[e.txn_id]
        checked += 1
    assert checked == len(training)


def test_tidy_rebuild_does_disagree(small):
    live = {e.txn_id: x for e, x in replay(small.events, small.homes, "arrival")}
    tidy = {e.txn_id: x for e, x in replay(small.events, small.homes, "event")}
    assert sum(live[k] != tidy[k] for k in live) > 0


def test_features_only_use_the_past():
    eng = FeatureEngine({1: (39.74, -104.99)})
    first = txn(t=1000)
    eng.update(first)
    later = txn(t=5000, tid="t2")
    eng.update(later)
    # A late event from before both: nothing in its 10-minute or 1-hour window.
    late = txn(t=900, tid="t3")
    f = dict(zip(NAMES, eng.features(late)))
    assert f["txns_last_10m"] == 0 and f["txns_last_1h"] == 0


def test_card_testing_burst_is_visible():
    eng = FeatureEngine({1: (39.74, -104.99)})
    for i in range(5):
        e = txn(t=1000 + i * 30, amount=1.5, category="digital", lat=None, lon=None, device="x", tid=f"b{i}")
        eng.update(e)
    f = dict(zip(NAMES, eng.features(txn(t=1200, amount=2.0, category="digital", lat=None, lon=None, device="x", tid="b9"))))
    assert f["tiny_txns_last_10m"] == 5 and f["new_device"] == 0


def test_impossible_travel():
    eng = FeatureEngine({1: (39.74, -104.99)})
    eng.update(txn(t=0))
    f = dict(zip(NAMES, eng.features(txn(t=3600, lat=40.71, lon=-74.01, tid="ny"))))  # NYC an hour later
    assert np.expm1(f["travel_speed"]) > 2000


def test_haversine_denver_to_new_york():
    assert 2610 < haversine_km(39.74, -104.99, 40.71, -74.01) < 2640


# ---- model and evaluation ------------------------------------------------------


def test_model_beats_rules_on_held_out_days(trained):
    _, report = trained
    assert report["model"]["recall"] > report["rules"]["recall"] + 0.2
    assert report["model"]["pr_auc"] > 0.7


def test_declines_are_mostly_fraud(trained):
    _, report = trained
    d = report["decisions"]["decline"]
    assert d["fraud"] / d["count"] > 0.75


def test_scalar_and_vector_scores_agree(trained, small):
    model, _ = trained
    rows = replay(small.events[:500], small.homes)
    X = np.array([x for _, x in rows])
    assert np.allclose(scores(model, X), [model.score(x) for _, x in rows], atol=1e-12)


def test_model_round_trips_through_json(trained):
    model, _ = trained
    again = Model.from_json(model.to_json())
    x = [1.0] * len(NAMES)
    assert again.score(x) == model.score(x)


def test_reasons_name_the_cause(trained):
    model, _ = trained
    eng = FeatureEngine({1: (39.74, -104.99)})
    for i in range(6):
        eng.update(txn(t=1000 + i * 20, amount=1.0, category="digital", merchant=9, lat=None, lon=None, device="x", tid=f"r{i}"))
    x = eng.features(txn(t=1150, amount=1.2, category="digital", merchant=9, lat=None, lon=None, device="x", tid="r9"))
    labels = [r["feature"] for r in model.reasons(x)]
    assert "tiny_txns_last_10m" in labels


def test_budget_metric():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0, 0, 0, 0, 0, 0, 0])
    r = at_budget(y, s, np.ones(10), budget=0.2)
    assert r["reviewed"] == 2 and r["fraud_caught"] == 1 and r["recall"] == 0.5


def test_rules_score_shape(small):
    X = np.array([x for _, x in replay(small.events[:100], small.homes)])
    assert rules_score(X).shape == (len(X),)


# ---- streaming --------------------------------------------------------------------


def test_dedupe_is_bounded():
    d = Dedupe(capacity=3)
    assert all(d.first_time(k) for k in "abcd")
    assert not d.first_time("d")
    assert d.first_time("a"), "oldest id should have been forgotten"


def test_stream_scores_every_transaction_once(small, trained):
    model, _ = trained
    events = small.events[:3000]
    out = io.StringIO()
    stats = asyncio.run(run_stream(events, model, small.homes, rate=None, queue_size=64, log=out))
    unique = len({e.txn_id for e in events})
    assert stats.scored == unique
    assert stats.duplicates == len(events) - unique
    assert stats.max_queue <= 64
    assert sum(stats.actions.values()) == unique


# ---- real data (runs only when the public dataset is present) ---------------------

import os

REAL = os.environ.get("CREDITCARD_CSV", "data/creditcard.csv")


@pytest.mark.skipif(not os.path.exists(REAL), reason="creditcard.csv not present; see README")
def test_real_dataset_holds_up():
    from countersign.realdata import run as run_real

    r = run_real(REAL)
    assert r["transactions"] == 284807 and r["fraud"] == 492
    assert r["model"]["recall"] > 0.75
    d = r["decisions"]["decline"]
    assert d["fraud"] / max(1, d["count"]) > 0.8
