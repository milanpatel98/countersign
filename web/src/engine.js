// A line-by-line port of countersign/features.py and model.py.
// test/parity.mjs replays the demo world and checks every score against Python.

const EARTH_KM = 6371.0088;
const rad = (d) => (d * Math.PI) / 180;

export function haversineKm(lat1, lon1, lat2, lon2) {
  const p1 = rad(lat1);
  const p2 = rad(lat2);
  const dp = p2 - p1;
  const dl = rad(lon2 - lon1);
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * EARTH_KM * Math.asin(Math.min(1, Math.sqrt(a)));
}

function bisectRight(arr, x) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x < arr[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

function newState() {
  return { times: [], items: [], n: 0, mean: 0, m2: 0, merchants: new Set(), devices: new Set() };
}

export class FeatureEngine {
  constructor(homes) {
    this.homes = homes;
    this.cards = new Map();
  }

  features(e) {
    const s = this.cards.get(e.card) || newState();
    const t = e.eventTime;
    const la = Math.log1p(e.amount);

    let z = 0;
    if (s.n >= 3) {
      const v = s.m2 / (s.n - 1);
      z = (la - s.mean) / Math.sqrt(v + 0.01);
      z = Math.max(-5, Math.min(8, z));
    }

    const hi = bisectRight(s.times, t);
    let c10 = 0;
    let c60 = 0;
    let tiny = 0;
    let sum60 = 0;
    for (let i = hi - 1; i >= 0; i--) {
      const [et, amt] = s.items[i];
      const age = t - et;
      if (age >= 3600) break;
      c60 += 1;
      sum60 += amt;
      if (age < 600) {
        c10 += 1;
        if (amt < 5) tiny += 1;
      }
    }

    let speed = 0;
    let distHome = 0;
    if (e.lat !== null) {
      for (let j = hi - 1; j >= 0; j--) {
        const [pt, , plat, plon] = s.items[j];
        if (plat !== null) {
          const hours = Math.max((t - pt) / 3600, 1 / 60);
          speed = Math.log1p(Math.min(5000, haversineKm(plat, plon, e.lat, e.lon) / hours));
          break;
        }
      }
      const [hlat, hlon] = this.homes[e.card];
      distHome = Math.log1p(haversineKm(hlat, hlon, e.lat, e.lon));
    }

    const gap = hi > 0 ? Math.log1p(Math.max(0, t - s.times[hi - 1]) / 3600) : Math.log1p(720);
    const hour = (t % 86400) / 3600;
    const cat = e.category;
    return [
      la,
      z,
      c10,
      c60,
      Math.log1p(sum60),
      tiny,
      s.merchants.has(e.merchant) ? 0 : 1,
      e.device !== null && !s.devices.has(e.device) ? 1 : 0,
      speed,
      distHome,
      hour < 5 ? 1 : 0,
      e.lat === null ? 1 : 0,
      cat === "digital" ? 1 : 0,
      cat === "giftcard" ? 1 : 0,
      cat === "electronics" ? 1 : 0,
      Math.log1p(s.n),
      gap,
    ];
  }

  update(e) {
    let s = this.cards.get(e.card);
    if (!s) {
      s = newState();
      this.cards.set(e.card, s);
    }
    const idx = bisectRight(s.times, e.eventTime);
    s.times.splice(idx, 0, e.eventTime);
    s.items.splice(idx, 0, [e.eventTime, e.amount, e.lat, e.lon]);
    const la = Math.log1p(e.amount);
    s.n += 1;
    const delta = la - s.mean;
    s.mean += delta / s.n;
    s.m2 += delta * (la - s.mean);
    s.merchants.add(e.merchant);
    if (e.device !== null) s.devices.add(e.device);
  }
}

export function score(model, x) {
  let z = model.bias;
  for (let i = 0; i < x.length; i++) z += (model.weights[i] * (x[i] - model.mean[i])) / model.std[i];
  return z > -60 ? 1 / (1 + Math.exp(-z)) : 0;
}

export function decide(model, p) {
  if (p >= model.decline_at) return "decline";
  if (p >= model.review_at) return "review";
  return "approve";
}

export function reasons(model, x, labels, k = 3) {
  const c = x.map((v, i) => [(model.weights[i] * (v - model.mean[i])) / model.std[i], i]);
  c.sort((a, b) => b[0] - a[0]);
  return c
    .slice(0, k)
    .filter(([w]) => w > 0.25)
    .map(([w, i]) => ({ feature: model.features[i], label: labels[model.features[i]] || model.features[i], weight: w }));
}

const PATTERNS = ["", "testing", "takeover", "skimmed", "quiet"];

// Turn the compact world file into events, in arrival order, as the scorer sees them.
export function hydrate(world) {
  const homes = {};
  for (const [k, v] of Object.entries(world.homes)) homes[k] = v;
  const events = world.events.map(([id, card, eventTime, arrivalTime, amount, merchant, device, fraud, pattern]) => {
    const [name, category, city, lat, lon] = world.merchants[merchant];
    return {
      id,
      card,
      eventTime,
      arrivalTime,
      amount,
      merchant,
      merchantName: name.replace(" .", "."),
      category,
      city,
      lat,
      lon,
      device: device >= 0 ? device : null,
      fraud: fraud === 1,
      pattern: PATTERNS[pattern],
    };
  });
  return { homes, events };
}

// Score the whole stream once: dedupe, features as of arrival, decision, reasons.
export function scoreStream(world, model, labels) {
  const { homes, events } = hydrate(world);
  const engine = new FeatureEngine(homes);
  const seen = new Set();
  const out = [];
  for (const e of events) {
    if (seen.has(e.id)) {
      out.push({ ...e, duplicate: true });
      continue;
    }
    seen.add(e.id);
    const x = engine.features(e);
    const p = score(model, x);
    const action = decide(model, p);
    engine.update(e);
    out.push({
      ...e,
      duplicate: false,
      late: e.arrivalTime - e.eventTime > 30,
      score: p,
      action,
      reasons: action === "approve" ? [] : reasons(model, x, labels),
    });
  }
  return out;
}
