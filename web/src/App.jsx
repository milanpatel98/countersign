import { useEffect, useMemo, useState } from "react";
import { scoreStream } from "./engine.js";

const LABELS = {
  log_amount: "Large amount",
  amount_vs_usual: "Bigger than this card usually spends",
  txns_last_10m: "Many purchases in 10 minutes",
  txns_last_1h: "Many purchases in the last hour",
  spend_last_1h: "High spend in the last hour",
  tiny_txns_last_10m: "Burst of tiny charges",
  new_merchant: "First time at this merchant",
  new_device: "New device",
  travel_speed: "Impossible travel",
  distance_from_home: "Far from home",
  night: "Middle of the night",
  card_not_present: "Online purchase",
  digital_goods: "Digital goods",
  gift_card: "Gift cards",
  electronics: "Electronics",
  history_length: "Card history",
  hours_since_last: "Long gap since last purchase",
};

const PATTERNS = {
  testing: ["Card testing", "Bursts of tiny online charges to check a stolen number."],
  skimmed: ["Skimmed card", "The physical card used in another city soon after a real swipe."],
  takeover: ["Account takeover", "A new device buying gift cards and digital goods."],
  quiet: ["Quiet misuse", "One slightly-too-big purchase at an odd hour. Hard on purpose."],
};

const DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud";
const REPO_URL = "https://github.com/milanpatel98/countersign";

const money = (n) => n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: n < 100 ? 2 : 0 });
const int = (n) => (n ?? 0).toLocaleString("en-US");
const pct = (x) => `${Math.round(x * 100)}%`;

function clock(t) {
  const s = t % 86400;
  return `Day ${Math.floor(t / 86400) + 1}, ${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;
}

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all(["world", "model", "report", "bench", "real"].map((f) => fetch(`data/${f}.json`).then((r) => (r.ok ? r.json() : null))))
      .then(([world, model, report, bench, real]) => {
        if (!world || !model) throw new Error("missing");
        const t0 = performance.now();
        const stream = scoreStream(world, model, LABELS);
        setData({ stream, model, report, bench, real, browserMs: performance.now() - t0 });
      })
      .catch(() => setError("The demo data is missing. Run “countersign train && countersign export” and reload."));
  }, []);

  return (
    <>
      <header className="hero">
        <div className="hero-grid-lines" aria-hidden="true" />
        <div className="wrap bar">
          <span>Countersign – 03</span>
          <a href={REPO_URL}>Source</a>
        </div>
        <div className="wrap hero-body">
          <h1 className="word">Countersign</h1>
          <p className="lede">Real-time card fraud scoring. Every payment checked in microseconds, every decline explained.</p>
        </div>
        <div className="wrap hero-foot">
          <span className="numeral">{data?.report ? pct(data.report.model.recall) : "–"}</span>
          <p className="hero-caption">
            of fraud caught while reviewing one payment in two hundred, on days the model never saw.
          </p>
        </div>
      </header>

      <section className="wrap live" aria-labelledby="live-title">
        <div className="live-text">
          <h2 id="live-title">Live</h2>
          <p>
            The model running in your browser on a stream of invented payments. Reviews and declines show their reasons.
          </p>
          <dl className="stats">
            <div>
              <dd>{data?.real ? pct(data.real.model.recall) : "–"}</dd>
              <dt>of fraud caught on real payments from a public dataset</dt>
            </div>
            <div>
              <dd>{data?.bench ? `${Math.round(data.bench.p50_us)} µs` : "–"}</dd>
              <dt>to score one payment on one CPU core</dt>
            </div>
          </dl>
        </div>
        {error ? <p className="notice">{error}</p> : <Feed data={data} />}
      </section>

      <main>
        {data?.report && <Results report={data.report} />}
        {data?.real && <Real real={data.real} />}
        {data?.report && <How report={data.report} bench={data.bench} browserMs={data.browserMs} count={data.stream.length} />}
      </main>

      <footer className="wrap foot">
        <p>© 2026 Milan Patel</p>
        <p>
          <a href={`${REPO_URL}/blob/main/LICENSE`}>MIT License</a>
        </p>
      </footer>
    </>
  );
}

/* ---- live feed --------------------------------------------------------------- */

function Feed({ data }) {
  const stream = data?.stream;
  const start = useMemo(() => (stream ? Math.max(0, stream.findIndex((e) => e.arrivalTime >= 4 * 86400)) : 0), [stream]);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(() => !window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  useEffect(() => setPos(start + 8), [start]);

  useEffect(() => {
    if (!stream || !playing) return undefined;
    const t = setInterval(() => setPos((p) => (p + 1 >= stream.length ? start : p + 1)), 900);
    return () => clearInterval(t);
  }, [stream, playing, start]);

  const rows = useMemo(() => {
    if (!stream) return [];
    const out = [];
    for (let i = Math.min(pos, stream.length - 1); i >= start && out.length < 5; i--) if (!stream[i].duplicate) out.push(stream[i]);
    return out;
  }, [stream, start, pos]);

  const tally = useMemo(() => {
    const t = { scored: 0, review: 0, decline: 0 };
    if (!stream) return t;
    for (let i = start; i <= pos && i < stream.length; i++) {
      const e = stream[i];
      if (e.duplicate) continue;
      t.scored++;
      if (e.action !== "approve") t[e.action]++;
    }
    return t;
  }, [stream, start, pos]);

  const nextAlert = () => {
    for (let i = pos + 1; i < stream.length; i++) {
      if (!stream[i].duplicate && stream[i].action !== "approve") return setPos(i);
    }
    return setPos(start);
  };

  return (
    <section className="feed" aria-label="Live payments">
      <div className="feed-head">
        <span className={`live-dot ${playing ? "on" : ""}`} aria-hidden="true" />
        <span className="feed-title">Live</span>
        <span className="feed-clock">{stream ? clock(stream[Math.min(pos, stream.length - 1)].arrivalTime) : "Loading"}</span>
      </div>
      <ol className="feed-list">
        {rows.map((e) => (
          <li key={e.id} className={`item item-${e.action}`}>
            <div className="item-main">
              <span className="item-merchant">{e.merchantName}</span>
              <span className="item-amount">{money(e.amount)}</span>
            </div>
            <div className="item-sub">
              <span>
                {e.city || "Online"}
                {e.late ? " · arrived late" : ""}
              </span>
              <span className={`pill pill-${e.action}`}>{e.action === "approve" ? "Approved" : e.action === "review" ? "Review" : "Declined"}</span>
            </div>
            {e.reasons.length > 0 && <p className="item-why">{e.reasons.map((r) => r.label).join(" · ")}</p>}
          </li>
        ))}
        {!stream && Array.from({ length: 5 }, (_, i) => <li key={i} className="item item-skeleton" />)}
      </ol>
      <div className="feed-foot">
        <span>
          {int(tally.scored)} scored · {int(tally.review)} review · {int(tally.decline)} declined
        </span>
        <span className="feed-controls">
          <button type="button" onClick={() => setPlaying((p) => !p)} disabled={!stream}>
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" onClick={nextAlert} disabled={!stream}>
            Next alert
          </button>
        </span>
      </div>
    </section>
  );
}

/* ---- sections ------------------------------------------------------------------ */

function Results({ report }) {
  const m = report.model;
  const r = report.rules;
  return (
    <section id="results" className="wrap section">
      <div className="section-head">
        <h2>Results</h2>
        <p>
          {int(report.transactions)} synthetic payments over 30 days. Trained on days 1 to 20, tested on days 25 to 30, which the model never saw.
          Analysts can review 0.5% of payments.
        </p>
      </div>
      <div className="compare">
        <Meter label="Countersign" value={m.recall} note={`${m.fraud_caught} of ${m.fraud_total} frauds · ${money(m.dollars_caught)} of ${money(m.dollars_total)}`} strong />
        <Meter label="Hand-written rules" value={r.recall} note={`${r.fraud_caught} of ${r.fraud_total} frauds · ${money(r.dollars_caught)}`} />
      </div>
      <table className="table">
        <thead>
          <tr>
            <th scope="col">Fraud pattern</th>
            <th scope="col" className="num">
              Caught
            </th>
            <th scope="col" className="hide-sm">
              What it looks like
            </th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(report.by_pattern).map(([k, v]) => (
            <tr key={k}>
              <td>{PATTERNS[k][0]}</td>
              <td className="num">
                {v.caught} / {v.total}
              </td>
              <td className="hide-sm muted">{PATTERNS[k][1]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Real({ real }) {
  const m = real.model;
  const a = real.largest_amounts;
  const d = real.decisions.decline;
  return (
    <section id="real" className="wrap section">
      <div className="section-head">
        <h2>Real data</h2>
        <p>
          Tested on the public <a href={DATASET_URL}>Credit Card Fraud Detection dataset</a> by Worldline and the Machine Learning Group of ULB:{" "}
          {int(real.transactions)} card payments from September 2013, {real.fraud} of them fraud. Its columns are anonymised, so this checks the model
          and evaluation rather than the per-card features. Trained on the first 60% by time, tested on the last {int(real.split.test)}.
        </p>
      </div>
      <div className="compare">
        <Meter label="Countersign" value={m.recall} note={`${m.fraud_caught} of ${m.fraud_total} frauds · reviewing ${m.reviewed} payments`} strong />
        <Meter label="Flag the largest payments" value={a.recall} note={`${a.fraud_caught} of ${a.fraud_total} frauds`} />
      </div>
      <p className="footnote">
        {d.fraud} of the {d.count} payments it would decline were fraud. Dataset: <a href={DATASET_URL}>kaggle.com/datasets/mlg-ulb/creditcardfraud</a>,
        Database Contents License (DbCL) v1.0.
      </p>
    </section>
  );
}

function How({ report, bench, browserMs, count }) {
  return (
    <section id="how" className="wrap section">
      <div className="section-head">
        <h2>How it works</h2>
      </div>
      <div className="cols">
        <article>
          <h3>Same features in training and live</h3>
          <p>
            One feature engine sees payments in the order they arrive, for training and for scoring. A tidy rebuild would have changed{" "}
            {int(report.skewed_vectors)} feature vectors; a test checks Countersign's match bit for bit.
          </p>
        </article>
        <article>
          <h3>Built for messy streams</h3>
          <p>
            {int(report.late_events)} payments arrived late and {int(report.duplicates)} were delivered twice. Late ones are still scored straight away;
            duplicates are dropped.
          </p>
        </article>
        <article>
          <h3>Fast and explainable</h3>
          <p>
            {bench ? `${int(bench.events_per_second)} payments a second on one core, p99 ${bench.p99_us} µs. ` : ""}A linear model gives exact reasons for
            every decline. This page scored {int(count)} payments in {Math.round(browserMs)} ms.
          </p>
        </article>
      </div>
    </section>
  );
}

function Meter({ label, value, note, strong }) {
  return (
    <div className={`meter ${strong ? "meter-strong" : ""}`}>
      <div className="meter-top">
        <span>{label}</span>
        <span className="meter-value">{pct(value)}</span>
      </div>
      <div className="meter-track">
        <span style={{ width: `${Math.max(1, value * 100)}%` }} />
      </div>
      <p className="meter-note">{note}</p>
    </div>
  );
}
