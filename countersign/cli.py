"""countersign train | bench | stream | export"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import __version__
from .evaluate import run
from .features import replay
from .gen import DAY, generate
from .model import Model
from .stream import benchmark, run_stream

ART = Path("artifacts")


def _model() -> Model:
    path = ART / "model.json"
    if not path.exists():
        sys.exit("No model yet. Run `countersign train` first.")
    return Model.from_json(path.read_text())


def cmd_train(args) -> None:
    model, report = run(args.seed)
    ART.mkdir(exist_ok=True)
    (ART / "model.json").write_text(model.to_json())
    (ART / "report.json").write_text(json.dumps(report, indent=2))
    m, r = report["model"], report["rules"]
    print(f"{report['transactions']:,} transactions, {report['fraud']:,} fraud, tested on days 24-29 ({report['split']['test_fraud']} fraud)")
    print(f"reviewing {m['reviewed']} transactions (0.5% of traffic):")
    print(f"  model  catches {m['fraud_caught']}/{m['fraud_total']} ({m['recall']:.0%}), ${m['dollars_caught']:,.0f} of ${m['dollars_total']:,.0f}, PR-AUC {m['pr_auc']:.3f}")
    print(f"  rules  catch   {r['fraud_caught']}/{r['fraud_total']} ({r['recall']:.0%}), ${r['dollars_caught']:,.0f}, PR-AUC {r['pr_auc']:.3f}")
    print(f"wrote {ART / 'model.json'} and {ART / 'report.json'}")


def cmd_bench(args) -> None:
    data = generate(args.seed)
    result = benchmark(data.events, _model(), data.homes)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


def cmd_stream(args) -> None:
    data = generate(args.seed)
    events = [e for e in data.events if e.arrival_time >= args.from_day * DAY][: args.limit]
    print(f"streaming {len(events):,} events at {args.rate or 'max'} per second; non-approvals below\n")
    stats = asyncio.run(run_stream(events, _model(), data.homes, rate=args.rate, log=sys.stdout))
    print(
        f"\nscored {stats.scored:,} · duplicates dropped {stats.duplicates} · late {stats.late} · "
        f"declined {stats.actions['decline']} · reviewed {stats.actions['review']} · "
        f"p50 {stats.percentile(0.5):.0f}µs p99 {stats.percentile(0.99):.0f}µs (including queue time) · max queue {stats.max_queue}"
    )


def cmd_export(args) -> None:
    """A small, separate world for the browser demo, plus the Python answers for the parity test."""
    model = _model()
    data = generate(seed=args.seed, cards=args.cards, days=args.days, merchants=300)
    merchants = data.merchants
    devices: dict[str, int] = {}
    pattern_code = {"": 0, "testing": 1, "takeover": 2, "skimmed": 3, "quiet": 4}
    rows = []
    for e in data.events:
        dev = -1
        if e.device is not None:
            dev = devices.setdefault(e.device, len(devices))
        rows.append([int(e.txn_id[1:]), e.card, e.event_time, e.arrival_time, e.amount, e.merchant, dev, int(e.fraud), pattern_code[e.pattern]])

    expected = []
    for e, x in replay(data.events, data.homes, "arrival"):
        p = model.score(x)
        expected.append([int(e.txn_id[1:]), round(p, 12), model.decide(p)])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "world.json").write_text(
        json.dumps(
            {
                "days": data.days,
                "homes": {str(k): v for k, v in data.homes.items()},
                "merchants": [[m["name"], m["category"], m["city"], m["lat"], m["lon"]] for m in merchants],
                "events": rows,
            },
            separators=(",", ":"),
        )
    )
    (out / "model.json").write_text(model.to_json())
    (out / "report.json").write_text((ART / "report.json").read_text())
    if (ART / "bench.json").exists():
        (out / "bench.json").write_text((ART / "bench.json").read_text())
    Path(args.expected).parent.mkdir(parents=True, exist_ok=True)
    Path(args.expected).write_text(json.dumps(expected, separators=(",", ":")))
    print(f"exported {len(rows):,} events for {args.cards} cards to {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="countersign", description="Real-time card fraud scoring.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="generate data, train, evaluate on held-out days")
    t.add_argument("--seed", type=int, default=7)
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("bench", help="measure scoring latency and throughput")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out")
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("stream", help="run the live pipeline and print every non-approval")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--rate", type=float, default=2000, help="events per second; 0 for as fast as possible")
    s.add_argument("--from-day", type=int, default=24)
    s.add_argument("--limit", type=int, default=20000)
    s.set_defaults(func=cmd_stream)

    x = sub.add_parser("export", help="write the browser demo's data")
    x.add_argument("--seed", type=int, default=11)
    x.add_argument("--cards", type=int, default=500)
    x.add_argument("--days", type=int, default=8)
    x.add_argument("--out", default="web/public/data")
    x.add_argument("--expected", default="web/test/expected.json")
    x.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    if getattr(args, "rate", None) == 0:
        args.rate = None
    args.func(args)


if __name__ == "__main__":
    main()
