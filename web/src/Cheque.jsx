import { useEffect, useMemo, useState } from "react";

// The homepage: each payment is written onto a cheque, and the model countersigns it.
// Approved: a signature draws itself. Review: an amber hold stamp. Declined: a red stamp,
// with the reasons written in the margin.

const ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
  "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

function words(n) {
  if (n === 0) return "zero";
  const under1000 = (x) => {
    const h = Math.floor(x / 100);
    const r = x % 100;
    const parts = [];
    if (h) parts.push(`${ONES[h]} hundred`);
    if (r >= 20) parts.push(TENS[Math.floor(r / 10)] + (r % 10 ? `-${ONES[r % 10]}` : ""));
    else if (r) parts.push(ONES[r]);
    return parts.join(" ");
  };
  const th = Math.floor(n / 1000);
  const rest = n % 1000;
  return [th ? `${under1000(th)} thousand` : "", rest ? under1000(rest) : ""].filter(Boolean).join(" ");
}

function amountInWords(amount) {
  const dollars = Math.floor(amount);
  const cents = Math.round((amount - dollars) * 100);
  const w = words(dollars);
  return `${w.charAt(0).toUpperCase()}${w.slice(1)} and ${String(cents).padStart(2, "0")}/100`;
}

const CATEGORY = {
  grocery: "Groceries", fuel: "Fuel", restaurant: "Dining", pharmacy: "Pharmacy", retail: "Retail", online: "Online shop",
  electronics: "Electronics", travel: "Travel", digital: "Digital goods", giftcard: "Gift cards",
};

function clock(t) {
  const s = t % 86400;
  return {
    day: Math.floor(t / 86400) + 1,
    time: `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`,
  };
}

function truthLine(e) {
  if (e.action === "approve") return e.fraud ? ["missed", "Missed: this one was fraud."] : ["right", "Right call: a genuine customer."];
  return e.fraud ? ["right", "Right call: this was fraud."] : ["wrong", "False alarm: a genuine customer, stopped by mistake."];
}

function Signature() {
  // An invented flourish, drawn as one stroke.
  return (
    <svg className="signature" viewBox="0 0 260 70" aria-hidden="true">
      <path
        pathLength="1"
        d="M6 50 C 18 18, 34 10, 40 26 C 46 44, 26 58, 22 46 C 18 32, 52 20, 66 34 C 74 42, 70 54, 62 50 C 54 46, 70 28, 86 30
           C 98 32, 92 48, 102 46 C 112 44, 112 26, 124 28 C 134 30, 128 48, 140 46 C 150 44, 152 30, 162 32 C 172 34, 166 50, 178 48
           C 192 46, 200 22, 214 20 C 226 18, 222 40, 210 46 C 200 50, 230 52, 254 38"
      />
    </svg>
  );
}

export default function Cheque({ stream, report, bench }) {
  // A reel that mixes routine approvals with every alert, so visitors see all three outcomes.
  const reel = useMemo(() => {
    if (!stream) return [];
    const out = [];
    let since = 0;
    for (const e of stream) {
      if (e.duplicate || e.arrivalTime < 4 * 86400) continue;
      if (e.action !== "approve" || (e.fraud && e.action === "approve")) {
        out.push(e);
        since = 0;
      } else if (since < 2 && e.amount > 8) {
        out.push(e);
        since++;
      }
    }
    return out;
  }, [stream]);

  const reduced = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(!reduced);
  const e = reel[i % Math.max(1, reel.length)];

  useEffect(() => {
    if (!playing || !e) return undefined;
    const t = setTimeout(() => setI((n) => n + 1), e.action === "approve" ? 3200 : 5200);
    return () => clearTimeout(t);
  }, [playing, i, e]);

  const m = report?.model;
  return (
    <section className="cheque-stage" aria-label="A payment being countersigned">
      {e ? <Paper key={`${e.id}-${i}`} e={e} /> : <div className="cheque cheque-blank" />}
      <div className="cheque-under">
        {e && (
          <p className={`truth-line truth-${truthLine(e)[0]}`}>
            <span>{truthLine(e)[1]}</span>
          </p>
        )}
        <div className="cheque-controls">
          <button type="button" className="btn btn-solid" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="btn" onClick={() => setI((n) => n + 1)}>
            Next payment
          </button>
        </div>
      </div>
      <p className="pitch">
        Every card payment gets a second signature. Countersign checks it in about {bench ? Math.round(bench.p50_us) : 8} µs, explains every no, and
        stopped {m ? Math.round(m.recall * 100) : 88}% of fraud on days it had never seen.
      </p>
    </section>
  );
}

function Paper({ e }) {
  const when = clock(e.eventTime);
  const amount = e.amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const place = e.city || "Online";
  return (
    <article className={`cheque cheque-${e.action}`}>
      <div className="cheque-top">
        <span className="cheque-bank">Countersign</span>
        <span className="cheque-no">No. {String(e.id).padStart(7, "0")}</span>
        <span className="cheque-date">
          Day {when.day} · {when.time}
        </span>
      </div>

      <div className="cheque-pay">
        <span className="cheque-label">Pay to the order of</span>
        <span className="payee">{e.merchantName}</span>
        <span className="amount-box">
          <span className="currency">$</span>
          {amount}
        </span>
      </div>

      <div className="cheque-words">
        <span>{amountInWords(e.amount)}</span>
        <span className="cheque-label">dollars</span>
      </div>

      <div className="cheque-bottom">
        <div className="memo">
          <span className="cheque-label">Memo</span>
          <span>
            {CATEGORY[e.category] || e.category} · {place}
            {e.late ? " · arrived late" : ""}
          </span>
        </div>
        <div className="sign">
          {e.action === "approve" && <Signature />}
          <span className="sign-line" />
          <span className="cheque-label">Countersigned</span>
        </div>
      </div>

      <p className="micr" aria-hidden="true">
        ⑆{String(e.card).padStart(4, "0")}⑆ {String(e.id).padStart(9, "0")} ⑈ {Math.round(e.score * 1000)
          .toString()
          .padStart(4, "0")}
      </p>

      {e.action !== "approve" && (
        <div className={`stamp stamp-${e.action}`}>
          <span className="stamp-word">{e.action === "decline" ? "Declined" : "Hold for review"}</span>
        </div>
      )}
      {e.reasons.length > 0 && (
        <ul className="notes" aria-label="Reasons">
          {e.reasons.map((r) => (
            <li key={r.feature}>{r.label}</li>
          ))}
        </ul>
      )}
    </article>
  );
}
