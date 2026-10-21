"""Features: one engine, used for training and for live scoring.

The classic way fraud models fail in production is training/serving skew:
features for training are rebuilt later from a tidy, time-sorted warehouse,
while the live scorer only knows what has arrived so far. Late events make the
two disagree, and the model quietly learns from information it never has at
decision time.

Here there is one engine. It sees events in arrival order and answers "what did
we know about this card when this transaction arrived?". Training replays the
same arrival-ordered stream through the same engine, so the vectors the model
learns from are, bit for bit, the vectors it is served. A test proves it, and
the evaluation measures what the naive time-sorted rebuild would have cost.

The engine is plain Python on purpose: web/src/features.js is a line-by-line
port, and a parity test checks the two agree on thousands of real events.
"""

from __future__ import annotations

import math
from bisect import bisect_right

NAMES = [
    "log_amount",
    "amount_vs_usual",
    "txns_last_10m",
    "txns_last_1h",
    "spend_last_1h",
    "tiny_txns_last_10m",
    "new_merchant",
    "new_device",
    "travel_speed",
    "distance_from_home",
    "night",
    "card_not_present",
    "digital_goods",
    "gift_card",
    "electronics",
    "history_length",
    "hours_since_last",
]

LABELS = {
    "log_amount": "Large amount",
    "amount_vs_usual": "Much bigger than this card usually spends",
    "txns_last_10m": "Many purchases in 10 minutes",
    "txns_last_1h": "Many purchases in the last hour",
    "spend_last_1h": "High spend in the last hour",
    "tiny_txns_last_10m": "Burst of tiny charges (card testing)",
    "new_merchant": "First purchase at this merchant",
    "new_device": "New device",
    "travel_speed": "Impossible travel since the last purchase",
    "distance_from_home": "Far from the billing address",
    "night": "Middle of the night",
    "card_not_present": "Online purchase",
    "digital_goods": "Digital goods",
    "gift_card": "Gift cards",
    "electronics": "Electronics",
    "history_length": "Card history",
    "hours_since_last": "Long gap since the last purchase",
}

EARTH_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


class CardState:
    __slots__ = ("times", "items", "n", "mean", "m2", "merchants", "devices")

    def __init__(self):
        self.times: list[float] = []  # event times, sorted
        self.items: list[tuple] = []  # (event_time, amount, lat, lon), same order
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.merchants: set[int] = set()
        self.devices: set[str] = set()


class FeatureEngine:
    """Feed transactions in arrival order. `features` reads, `update` learns."""

    def __init__(self, homes: dict[int, tuple[float, float]]):
        self.homes = homes
        self.cards: dict[int, CardState] = {}

    def features(self, e) -> list[float]:
        s = self.cards.get(e.card) or CardState()
        t = e.event_time
        amount = e.amount
        la = math.log1p(amount)

        # Amount against this card's own history (Welford mean and variance).
        if s.n >= 3:
            var = s.m2 / (s.n - 1)
            z = (la - s.mean) / math.sqrt(var + 0.01)
            z = max(-5.0, min(8.0, z))
        else:
            z = 0.0

        # Windows anchored at this event's time, over what is known now.
        hi = bisect_right(s.times, t)
        c10 = c60 = tiny = 0
        sum60 = 0.0
        i = hi - 1
        while i >= 0:
            et, amt, _, _ = s.items[i]
            age = t - et
            if age >= 3600:
                break
            c60 += 1
            sum60 += amt
            if age < 600:
                c10 += 1
                if amt < 5:
                    tiny += 1
            i -= 1

        # Speed from the previous located purchase, if both have a place.
        speed = 0.0
        dist_home = 0.0
        if e.lat is not None:
            j = hi - 1
            while j >= 0:
                pt, _, plat, plon = s.items[j]
                if plat is not None:
                    hours = max((t - pt) / 3600.0, 1.0 / 60.0)
                    speed = math.log1p(min(5000.0, haversine_km(plat, plon, e.lat, e.lon) / hours))
                    break
                j -= 1
            hlat, hlon = self.homes[e.card]
            dist_home = math.log1p(haversine_km(hlat, hlon, e.lat, e.lon))

        if hi > 0:
            gap = math.log1p(max(0.0, t - s.times[hi - 1]) / 3600.0)
        else:
            gap = math.log1p(720.0)

        hour = (t % 86400.0) / 3600.0
        cat = e.category
        return [
            la,
            z,
            float(c10),
            float(c60),
            math.log1p(sum60),
            float(tiny),
            0.0 if e.merchant in s.merchants else 1.0,
            1.0 if (e.device is not None and e.device not in s.devices) else 0.0,
            speed,
            dist_home,
            1.0 if hour < 5.0 else 0.0,
            1.0 if e.lat is None else 0.0,
            1.0 if cat == "digital" else 0.0,
            1.0 if cat == "giftcard" else 0.0,
            1.0 if cat == "electronics" else 0.0,
            math.log1p(s.n),
            gap,
        ]

    def update(self, e) -> None:
        s = self.cards.get(e.card)
        if s is None:
            s = self.cards[e.card] = CardState()
        idx = bisect_right(s.times, e.event_time)
        s.times.insert(idx, e.event_time)
        s.items.insert(idx, (e.event_time, e.amount, e.lat, e.lon))
        la = math.log1p(e.amount)
        s.n += 1
        delta = la - s.mean
        s.mean += delta / s.n
        s.m2 += delta * (la - s.mean)
        s.merchants.add(e.merchant)
        if e.device is not None:
            s.devices.add(e.device)


def replay(events, homes, order: str = "arrival"):
    """Compute one feature vector per unique transaction.

    order="arrival": what the live scorer knows (the correct way to train).
    order="event":   a tidy rebuild sorted by event time (the tempting, skewed way).
    """
    seen = set()
    stream = events if order == "arrival" else sorted(events, key=lambda e: (e.event_time, e.txn_id))
    engine = FeatureEngine(homes)
    rows = []
    for e in stream:
        if e.txn_id in seen:
            continue  # at-least-once delivery: score each transaction once
        seen.add(e.txn_id)
        rows.append((e, engine.features(e)))
        engine.update(e)
    return rows

