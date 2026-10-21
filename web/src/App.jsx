import { useEffect, useMemo, useRef, useState } from "react";
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

const PATTERN_TEXT = {
  testing: ["Card testing", "Bursts of tiny online charges to see if a stolen number works."],
  skimmed: ["Skimmed card", "The physical card used in another city soon after a real swipe."],
  takeover: ["Account takeover", "A new device buying gift cards and digital goods."],
  quiet: ["Quiet misuse", "One slightly-too-big purchase at an odd hour. Hard on purpose."],
};

const money = (n) => n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: n < 100 ? 2 : 0 });
const int = (n) => (n ?? 0).toLocaleString("en-US");
const pct = (x) => `${Math.round(x * 100)}%`;

function clock(t) {
  const day = Math.floor(t / 86400) + 1;
  const s = t % 86400;
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  return { day, time: `${h}:${m}` };
}

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all(["world", "model", "report", "bench"].map((f) => fetch(`data/${f}.json`).then((r) => (r.ok ? r.json() : null))))
      .then(([world, model, report, bench]) => {
        if (!world || !model) throw new Error("missing");
        const t0 = performance.now();
        const stream = scoreStream(world, model, LABELS);
        setData({ stream, model, report, bench, browserMs: performance.now() - t0 });
      })
      .catch(() => setError("The demo data is missing. Run “countersign train && countersign export” and reload."));
  }, []);

  return (
    <div className="shell">
      <Hero report={data?.report} />
      {error && <p className="notice">{error}</p>}
      {data && <Live data={data} />}
      {data?.report && <Results report={data.report} bench={data.bench} browserMs={data.browserMs} count={data.stream.length} />}
      <footer className="foot">
        <p>© 2026 Milan Patel</p>
        <p>
          <a href="https://github.com/milanpatel98/countersign/blob/main/LICENSE">MIT License</a>
        </p>
      </footer>
    </div>
  );
}

function Guilloche() {
  // The interlaced wave pattern printed on cheques to make forgery hard.
  const lines = [];
  for (let k = 0; k < 26; k++) {
    let d = "";
    for (let x = 0; x <= 1200; x += 12) {
      const y = 300 + Math.sin(x / 70 + k * 0.42) * (90 + k * 4) + Math.sin(x / 23 + k) * 12 + (k - 13) * 9;
      d += `${x === 0 ? "M" : "L"}${x} ${y.toFixed(1)}`;
    }
    lines.push(<path key={k} d={d} />);
  }
  return (
    <svg className="guilloche" viewBox="0 0 1200 600" preserveAspectRatio="none" aria-hidden="true">
      {lines}
    </svg>
  );
}

function Hero({ report }) {
  const m = report?.model;
  return (
    <header className="hero">
      <Guilloche />
      <div className="bar">
        <span>Countersign – 03</span>
        <a href="https://github.com/milanpatel98/countersign">Source</a>
      </div>
      <h1 className="word">Countersign</h1>
      <p className="hero-line">Scores every card payment in microseconds, and says why it said no.</p>
      <div className="hero-foot">
        <span className="numeral">{m ? pct(m.recall) : "–"}</span>
        <p className="hero-caption">
          of fraud caught while reviewing one payment in two hundred, on days the model never saw.
        </p>
      </div>
    </header>
  );
}

const SPEEDS = [
  { label: "Slow", n: 1 },
  { label: "Fast", n: 6 },
  { label: "Faster", n: 30 },
];

function Live({ data }) {
  const { stream, model } = data;
  const start = useMemo(() => Math.max(0, stream.findIndex((e) => e.arrivalTime >= 4 * 86400)), [stream]);
  const [pos, setPos] = useState(start);
  const [playing, setPlaying] = useState(() => !window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [speed, setSpeed] = useState(1);
  const timer = useRef(null);

  useEffect(() => {
    if (!playing) return undefined;
    timer.current = setInterval(() => {
      setPos((p) => (p + SPEEDS[speed].n >= stream.length ? start : p + SPEEDS[speed].n));
    }, 420);
    return () => clearInterval(timer.current);
  }, [playing, speed, stream.length, start]);

  const seen = useMemo(() => {
    const s = { scored: 0, approve: 0, review: 0, decline: 0, dupes: 0, late: 0, stopped: 0, missed: 0, stoppedN: 0, missedN: 0 };
    for (let i = start; i <= pos && i < stream.length; i++) {
      const e = stream[i];
      if (e.duplicate) {
        s.dupes++;
        continue;
      }
      s.scored++;
      s[e.action]++;
      if (e.late) s.late++;
      if (e.fraud && e.action !== "approve") {
        s.stopped += e.amount;
        s.stoppedN++;
      }
      if (e.fraud && e.action === "approve") {
        s.missed += e.amount;
        s.missedN++;
      }
    }
    return s;
  }, [stream, start, pos]);

  const tape = useMemo(() => {
    const out = [];
    for (let i = Math.min(pos, stream.length - 1); i >= start && out.length < 9; i--) if (!stream[i].duplicate) out.push(stream[i]);
    return out;
  }, [stream, start, pos]);

  const now = clock(stream[Math.min(pos, stream.length - 1)].arrivalTime);

  const nextAlert = () => {
    for (let i = pos + 1; i < stream.length; i++) {
      const e = stream[i];
      if (!e.duplicate && (e.action !== "approve" || e.fraud)) {
        setPos(i);
        return;
      }
    }
    setPos(start);
  };

  return (
    <section className="live" aria-labelledby="live-title">
      <div className="live-head">
        <div>
          <h2 id="live-title">Live</h2>
          <p className="live-sub">
            The model running in your browser on a synthetic day of payments. Day {now.day}, {now.time}.
          </p>
        </div>
        <div className="controls">
          <button type="button" className="btn btn-solid" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <div className="seg" role="group" aria-label="Speed">
            {SPEEDS.map((s, i) => (
              <button key={s.label} type="button" aria-pressed={i === speed} onClick={() => setSpeed(i)}>
                {s.label}
              </button>
            ))}
          </div>
          <button type="button" className="btn" onClick={nextAlert}>
            Next alert
          </button>
        </div>
      </div>

      <dl className="tally">
        <div>
          <dt>Scored</dt>
          <dd>{int(seen.scored)}</dd>
        </div>
        <div>
          <dt>Approved</dt>
          <dd>{int(seen.approve)}</dd>
        </div>
        <div>
          <dt>Sent to review</dt>
          <dd>{int(seen.review)}</dd>
        </div>
        <div className="tally-decline">
          <dt>Declined</dt>
          <dd>{int(seen.decline)}</dd>
        </div>
        <div>
          <dt>Fraud stopped</dt>
          <dd>{money(seen.stopped)}</dd>
        </div>
        <div>
          <dt>Fraud missed</dt>
          <dd>{money(seen.missed)}</dd>
        </div>
      </dl>

      <ol className="tape" aria-live="off">
        {tape.map((e) => (
          <Row key={e.id} e={e} model={model} />
        ))}
      </ol>
      <p className="tape-note">
        {int(seen.dupes)} duplicate deliveries dropped and {int(seen.late)} late arrivals scored so far. Everything here is invented: no real people,
        cards or shops.
      </p>
    </section>
  );
}

function Row({ e, model }) {
  const place = e.city || "Online";
  const t = clock(e.eventTime);
  const missed = e.fraud && e.action === "approve";
  const bar = Math.min(1, e.score / Math.max(model.decline_at, 1e-9));
  return (
    <li className={`row row-${e.action}${missed ? " row-missed" : ""}`}>
      <span className="row-time">{t.time}</span>
      <span className="row-who">
        <span className="row-merchant">{e.merchantName}</span>
        <span className="row-place">
          {place}
          {e.late && " · arrived late"}
        </span>
      </span>
      <span className="row-amount">{money(e.amount)}</span>
      <span className="row-risk" aria-label={`Risk ${Math.round(bar * 100)} of 100`}>
        <span style={{ width: `${Math.max(3, bar * 100)}%` }} />
      </span>
      <span className="row-action">{e.action === "approve" ? "Approved" : e.action === "review" ? "Review" : "Declined"}</span>
      {(e.reasons.length > 0 || e.fraud) && (
        <span className="row-why">
          {e.reasons.map((r) => r.label).join(" · ")}
          {e.fraud && <span className={`truth ${missed ? "truth-missed" : ""}`}>{missed ? "Fraud, missed" : "Fraud"}</span>}
        </span>
      )}
    </li>
  );
}

function Results({ report, bench, browserMs, count }) {
  const m = report.model;
  const r = report.rules;
  const s = report.skewed_training;
  return (
    <section className="results" aria-labelledby="results-title">
      <h2 id="results-title">How well it works</h2>
      <p className="results-intro">
        Trained on days 1 to 20 of {int(report.transactions)} synthetic payments, tuned on days 21 to 24, and tested on days 25 to 30, which it never saw.
        Analysts can review 0.5% of traffic, so the question is how much fraud lands inside that budget.
      </p>

      <div className="compare">
        <Bar label="Countersign" value={m.recall} detail={`${m.fraud_caught} of ${m.fraud_total} frauds · ${money(m.dollars_caught)} of ${money(m.dollars_total)}`} strong />
        <Bar label="Hand-written rules" value={r.recall} detail={`${r.fraud_caught} of ${r.fraud_total} frauds · ${money(r.dollars_caught)}`} />
      </div>

      <div className="grid">
        <article>
          <h3>What it catches</h3>
          <ul className="patterns">
            {Object.entries(report.by_pattern).map(([k, v]) => (
              <li key={k}>
                <span className="pattern-name">{PATTERN_TEXT[k][0]}</span>
                <span className="pattern-score">
                  {v.caught}/{v.total}
                </span>
                <span className="pattern-note">{PATTERN_TEXT[k][1]}</span>
              </li>
            ))}
          </ul>
        </article>

        <article>
          <h3>Training sees what serving sees</h3>
          <p>
            Rebuilding features later from a tidy, time-sorted table changes {int(report.skewed_vectors)} of them, because late payments were not known
            when the decision was made. Countersign trains on the same arrival-ordered replay it serves, and a test checks the two match bit for bit.
          </p>
          <p className="muted">
            On this data the tidy rebuild happened to score about the same (PR-AUC {s.pr_auc.toFixed(3)} against {m.pr_auc.toFixed(3)}). The guarantee is
            the point; the gap shows up when late data is less random than it is here.
          </p>
        </article>

        <article>
          <h3>Fast enough for the till</h3>
          {bench ? (
            <p>
              {int(bench.events_per_second)} payments a second on one core. Half finish in {bench.p50_us} µs, 99% in {bench.p99_us} µs, including
              features, score, decision and reasons.
            </p>
          ) : null}
          <p className="muted">
            This page scored {int(count)} payments in {Math.round(browserMs)} ms with a JavaScript port of the engine, which agrees with the Python
            version on every score.
          </p>
        </article>
      </div>
    </section>
  );
}

function Bar({ label, value, detail, strong }) {
  return (
    <div className={`bar-row ${strong ? "bar-strong" : ""}`}>
      <div className="bar-label">
        <span>{label}</span>
        <span className="bar-value">{pct(value)}</span>
      </div>
      <div className="bar-track">
        <span style={{ width: `${value * 100}%` }} />
      </div>
      <p className="bar-detail">{detail}</p>
    </div>
  );
}
