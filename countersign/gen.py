"""Synthetic card transactions with realistic fraud and realistic delivery.

Everything is fictional: invented cardholders, invented merchants, real US
city coordinates. One seed produces the same 30 days every time.

Normal behaviour is per cardholder: a home city, a spending level, a daily
rhythm, a handful of favourite merchants, one or two devices.

Fraud comes in four shapes, each built the way it shows up in practice:

* card testing     bursts of tiny online charges to see if a stolen number works
* account takeover a new device in a far-off city buying electronics and gift cards
* skimmed card     the physical card used in another city soon after a real swipe
* quiet misuse     a slightly-too-big purchase at an unfamiliar shop at an odd hour
                   (deliberately hard; a model that catches all of these is lying)

Delivery is messy too: most events arrive within two seconds, some arrive
minutes late, and a few are delivered twice.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

DAY = 86_400

CITIES = [
    ("Denver", 39.74, -104.99), ("Golden", 39.76, -105.22), ("Aurora", 39.73, -104.83),
    ("Boulder", 40.01, -105.27), ("Colorado Springs", 38.83, -104.82), ("Salt Lake City", 40.76, -111.89),
    ("Phoenix", 33.45, -112.07), ("Los Angeles", 34.05, -118.24), ("San Francisco", 37.77, -122.42),
    ("Seattle", 47.61, -122.33), ("Chicago", 41.88, -87.63), ("Dallas", 32.78, -96.80),
    ("Houston", 29.76, -95.37), ("Atlanta", 33.75, -84.39), ("Miami", 25.76, -80.19),
    ("New York", 40.71, -74.01), ("Boston", 42.36, -71.06), ("Minneapolis", 44.98, -93.27),
    ("Las Vegas", 36.17, -115.14), ("Portland", 45.52, -122.68),
]
HOME_CITIES = CITIES[:5]  # cardholders live around Colorado; fraud often doesn't

CATEGORIES = {
    #  name          amount scale  card present  share of merchants
    "grocery": (45, True, 0.20),
    "fuel": (40, True, 0.10),
    "restaurant": (32, True, 0.20),
    "pharmacy": (25, True, 0.06),
    "retail": (60, True, 0.14),
    "online": (48, False, 0.14),
    "electronics": (240, True, 0.05),
    "travel": (320, False, 0.04),
    "digital": (12, False, 0.04),
    "giftcard": (100, False, 0.03),
}

NAME_A = ["Cedar", "Summit", "Juniper", "Copper", "Aspen", "Granite", "Willow", "Harbor", "Prairie", "Maple",
          "Canyon", "Silver", "Birch", "North", "Red Rock", "Blue Sky", "Pine", "Clear Creek", "Mesa", "Alpine"]
NAME_B = {
    "grocery": ["Market", "Grocers", "Foods", "Pantry"], "fuel": ["Fuel", "Gas & Go", "Petro"],
    "restaurant": ["Kitchen", "Diner", "Noodle Bar", "Taqueria", "Bistro"], "pharmacy": ["Pharmacy", "Drug"],
    "retail": ["Outfitters", "Goods", "Supply", "Home"], "online": ["Shop Online", "Direct", ".shop"],
    "electronics": ["Electronics", "Tech", "Digital Depot"], "travel": ["Air", "Travel", "Lodging"],
    "digital": ["Apps", "Streaming", "Games"], "giftcard": ["Gift Cards", "eGift"],
}


@dataclass
class Txn:
    txn_id: str
    card: int
    event_time: float  # when the purchase happened
    arrival_time: float  # when the event reached the scorer
    amount: float
    merchant: int
    category: str
    city: str | None  # None for card-not-present
    lat: float | None
    lon: float | None
    device: str | None  # set for online purchases
    fraud: bool
    pattern: str  # "" for legitimate

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Merchant:
    id: int
    name: str
    category: str
    city: str | None
    lat: float | None
    lon: float | None


@dataclass
class Card:
    id: int
    home: tuple
    scale: float  # personal spending multiplier
    rate: float  # purchases per day
    peak_hour: float
    favourites: list
    devices: list


def _merchants(rng: random.Random, n: int) -> list[Merchant]:
    out = []
    cats = list(CATEGORIES)
    weights = [CATEGORIES[c][2] for c in cats]
    for i in range(n):
        cat = rng.choices(cats, weights)[0]
        present = CATEGORIES[cat][1]
        name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B[cat])}"
        if present:
            city = rng.choice(HOME_CITIES) if rng.random() < 0.75 else rng.choice(CITIES)
            out.append(Merchant(i, name, cat, city[0], city[1], city[2]))
        else:
            out.append(Merchant(i, name, cat, None, None, None))
    return out


@dataclass
class Dataset:
    events: list  # Txn, sorted by arrival time: the order a scorer sees them
    homes: dict  # card id -> (lat, lon) of the billing address
    merchants: list
    days: int


def generate(seed: int = 7, cards: int = 3000, days: int = 30, merchants: int = 700) -> Dataset:
    rng = random.Random(seed)
    shops = _merchants(rng, merchants)
    by_cat: dict[str, list[Merchant]] = {}
    for m in shops:
        by_cat.setdefault(m.category, []).append(m)

    holders = []
    for c in range(cards):
        home = rng.choice(HOME_CITIES)
        local = [m for m in shops if m.city is None or m.city == home[0]]
        favs = rng.sample(local, k=min(len(local), rng.randint(6, 16)))
        holders.append(
            Card(
                id=c,
                home=home,
                scale=math.exp(rng.gauss(0, 0.45)),
                rate=rng.uniform(0.6, 5.5),
                peak_hour=rng.gauss(15, 3),
                favourites=favs,
                devices=[f"d{c}-{k}" for k in range(rng.choice([1, 1, 2]))],
            )
        )

    events: list[Txn] = []
    seq = 0

    def emit(card: Card, t: float, amount: float, m: Merchant, fraud: bool, pattern: str, device: str | None = None):
        nonlocal seq
        seq += 1
        if m.city is None and device is None:
            device = rng.choice(card.devices)
        events.append(
            Txn(
                txn_id=f"t{seq:07d}",
                card=card.id,
                event_time=round(t, 3),
                arrival_time=0.0,
                amount=round(max(0.5, amount), 2),
                merchant=m.id,
                category=m.category,
                city=m.city,
                lat=m.lat,
                lon=m.lon,
                device=device if m.city is None else None,
                fraud=fraud,
                pattern=pattern,
            )
        )

    # ---- normal life -------------------------------------------------------
    for card in holders:
        n = _poisson(rng, card.rate * days)
        for _ in range(n):
            day = rng.randrange(days)
            hour = min(23.99, max(0.0, rng.gauss(card.peak_hour, 3.2)))
            t = day * DAY + hour * 3600 + rng.uniform(0, 60)
            m = rng.choice(card.favourites) if rng.random() < 0.82 else rng.choice(
                [s for s in shops if s.city in (None, card.home[0])]
            )
            base = CATEGORIES[m.category][0]
            emit(card, t, base * card.scale * math.exp(rng.gauss(0, 0.55)), m, False, "")

    # ---- fraud ---------------------------------------------------------------
    victims = rng.sample(holders, k=int(cards * 0.07))
    far = [c for c in CITIES if c not in HOME_CITIES]
    for card in victims:
        start = rng.uniform(2 * DAY, (days - 0.5) * DAY)
        kind = rng.choices(["testing", "takeover", "skimmed", "quiet"], [0.3, 0.25, 0.25, 0.2])[0]
        if kind == "testing":
            dev = f"x{card.id}-{rng.randrange(1000)}"
            t = start
            for _ in range(rng.randint(4, 12)):
                t += rng.uniform(5, 90)
                emit(card, t, rng.uniform(0.5, 3.5), rng.choice(by_cat["digital"]), True, kind, dev)
            if rng.random() < 0.6:
                emit(card, t + rng.uniform(60, 1800), rng.uniform(250, 1500), rng.choice(by_cat["giftcard"] + by_cat["online"]), True, kind, dev)
        elif kind == "takeover":
            dev = f"x{card.id}-{rng.randrange(1000)}"
            t = start
            for _ in range(rng.randint(2, 6)):
                t += rng.uniform(120, 1500)
                m = rng.choice(by_cat["giftcard"] + by_cat["online"] + by_cat["digital"])
                emit(card, t, rng.uniform(150, 1200), m, True, kind, dev)
        elif kind == "skimmed":
            city = rng.choice(far)
            local = [m for m in shops if m.city == city[0] and m.category in ("electronics", "retail", "fuel", "grocery")]
            if not local:
                continue
            # a genuine swipe at home shortly before
            home_shop = rng.choice([m for m in card.favourites if m.city] or card.favourites)
            emit(card, start - rng.uniform(600, 5400), CATEGORIES[home_shop.category][0] * card.scale, home_shop, False, "")
            t = start
            for _ in range(rng.randint(1, 5)):
                t += rng.uniform(300, 2400)
                m = rng.choice(local)
                emit(card, t, CATEGORIES[m.category][0] * rng.uniform(2, 8), m, True, kind)
        else:  # quiet
            m = rng.choice([s for s in shops if s.city in (None, card.home[0]) and s not in card.favourites])
            hour = rng.choice([1, 2, 3, 4, 23])
            t = int(start // DAY) * DAY + hour * 3600 + rng.uniform(0, 3600)
            emit(card, t, CATEGORIES[m.category][0] * card.scale * rng.uniform(1.8, 4.5), m, True, kind)

    # ---- delivery ------------------------------------------------------------
    delivered: list[Txn] = []
    for e in events:
        r = rng.random()
        if r < 0.03:
            delay = rng.uniform(30, 600)  # late: a slow acquirer or a retry queue
        else:
            delay = rng.expovariate(1.5)  # usually well under two seconds
        e.arrival_time = round(e.event_time + delay, 3)
        delivered.append(e)
        if rng.random() < 0.004:  # at-least-once delivery: the same event, again
            dup = Txn(**e.to_dict())
            dup.arrival_time = round(e.arrival_time + rng.uniform(0.5, 120), 3)
            delivered.append(dup)
    delivered.sort(key=lambda e: (e.arrival_time, e.txn_id))
    homes = {c.id: (c.home[1], c.home[2]) for c in holders}
    return Dataset(delivered, homes, [asdict(m) for m in shops], days)


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth's method is fine for the small rates here; use a normal approximation above 60.
    if lam > 60:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    l, k, p = math.exp(-lam), 0, 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1
