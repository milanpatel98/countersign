"""The live scorer: events in, decisions out, each within a latency budget.

Shape of the pipeline:

    source ──► bounded queue ──► scorer ──► decision log
               (backpressure)     │
                                  ├─ dedupe by txn id (at-least-once delivery)
                                  ├─ features as of arrival (same engine as training)
                                  ├─ score, decide, explain
                                  └─ flag late arrivals, never drop them

A payment cannot wait for a watermark: the card is at the till. So late events
are scored immediately with what is known, and marked late in the log so
analysts and retraining can see them. That is also why training replays the
stream in arrival order.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .features import FeatureEngine
from .model import Model

LATE_AFTER = 30.0  # seconds between purchase and arrival


@dataclass
class Decision:
    txn_id: str
    card: int
    amount: float
    score: float
    action: str
    reasons: list
    late: bool
    latency_us: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


class Dedupe:
    """Remembers recent transaction ids in bounded memory."""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.seen: OrderedDict[str, None] = OrderedDict()

    def first_time(self, key: str) -> bool:
        if key in self.seen:
            self.seen.move_to_end(key)
            return False
        self.seen[key] = None
        if len(self.seen) > self.capacity:
            self.seen.popitem(last=False)
        return True


@dataclass
class Stats:
    scored: int = 0
    duplicates: int = 0
    late: int = 0
    actions: dict = field(default_factory=lambda: {"approve": 0, "review": 0, "decline": 0})
    latencies: list = field(default_factory=list)
    max_queue: int = 0

    def percentile(self, q: float) -> float:
        if not self.latencies:
            return 0.0
        xs = sorted(self.latencies)
        return xs[min(len(xs) - 1, int(q * len(xs)))]


class Scorer:
    def __init__(self, model: Model, homes: dict):
        self.model = model
        self.engine = FeatureEngine(homes)
        self.dedupe = Dedupe()
        self.stats = Stats()

    def handle(self, e, received_ns: int) -> Decision | None:
        if not self.dedupe.first_time(e.txn_id):
            self.stats.duplicates += 1
            return None
        x = self.engine.features(e)
        p = self.model.score(x)
        action = self.model.decide(p)
        reasons = self.model.reasons(x) if action != "approve" else []
        self.engine.update(e)
        late = (e.arrival_time - e.event_time) > LATE_AFTER
        latency = (time.perf_counter_ns() - received_ns) / 1000
        self.stats.scored += 1
        self.stats.late += late
        self.stats.actions[action] += 1
        self.stats.latencies.append(latency)
        return Decision(e.txn_id, e.card, e.amount, round(p, 6), action, reasons, late, round(latency, 1))


async def run_stream(events, model: Model, homes: dict, rate: float | None = None, queue_size: int = 1024, log=None) -> Stats:
    """Push events through the pipeline. rate=None means as fast as possible."""
    scorer = Scorer(model, homes)
    q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

    async def source():
        interval = 1.0 / rate if rate else 0.0
        start = time.perf_counter()
        for i, e in enumerate(events):
            if interval:
                delay = start + i * interval - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            await q.put((e, time.perf_counter_ns()))  # blocks when the scorer falls behind
            scorer.stats.max_queue = max(scorer.stats.max_queue, q.qsize())
        await q.put(None)

    async def score():
        while True:
            item = await q.get()
            if item is None:
                return
            d = scorer.handle(*item)
            if d is not None and log is not None and d.action != "approve":
                log.write(d.to_json() + "\n")

    await asyncio.gather(source(), score())
    return scorer.stats


def benchmark(events, model: Model, homes: dict) -> dict:
    """Score latency per event (features + model + decision), single core."""
    scorer = Scorer(model, homes)
    t0 = time.perf_counter()
    for e in events:
        scorer.handle(e, time.perf_counter_ns())
    elapsed = time.perf_counter() - t0
    s = scorer.stats
    return {
        "events": len(events),
        "scored": s.scored,
        "duplicates_dropped": s.duplicates,
        "late_scored": s.late,
        "seconds": round(elapsed, 2),
        "events_per_second": round(len(events) / elapsed),
        "p50_us": round(s.percentile(0.50), 1),
        "p99_us": round(s.percentile(0.99), 1),
        "p999_us": round(s.percentile(0.999), 1),
        "actions": s.actions,
    }
