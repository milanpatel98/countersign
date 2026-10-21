# Countersign

**Real-time card fraud scoring that explains every decline, with features that cannot drift between training and serving.**

Countersign scores a card payment in microseconds: it computes behavioural features for the card, runs a model, decides approve, review or decline, and gives the top reasons in plain words. The project is built around three problems production fraud systems actually have:

1. **Training/serving skew.** Features rebuilt later from a tidy warehouse differ from what the live scorer knew at decision time. Countersign trains and serves from one engine fed the same arrival-ordered stream, and a test proves the vectors match bit for bit.
2. **Messy delivery.** Events arrive late, out of order and twice. Late ones are still scored immediately (the customer is at the till) and marked late; duplicates are dropped.
3. **Explainability.** Issuers must say why a payment was declined, so every non-approval comes with its reasons.

**[Live demo](https://milanpatel98.github.io/countersign/)**: the model running in your browser on a stream of synthetic payments.

---

## Results

276,650 synthetic payments over 30 days, 925 of them fraudulent. Trained on days 1 to 20, thresholds tuned on days 21 to 24, tested on days 25 to 30. Analysts can review 0.5% of traffic.

| Reviewing 0.5% of payments | Fraud caught | Fraud dollars caught | PR-AUC |
| --- | --- | --- | --- |
| **Countersign** | **154 of 176 (88%)** | **$36,868 of $42,677** | **0.878** |
| Hand-written rules | 77 of 176 (44%) | $9,902 | 0.458 |

| Fraud pattern | Caught | What it looks like |
| --- | --- | --- |
| Card testing | 98 / 102 | Bursts of tiny online charges to check a stolen number |
| Skimmed card | 36 / 36 | The physical card used in another city soon after a real swipe |
| Account takeover | 18 / 27 | A new device buying gift cards and digital goods |
| Quiet misuse | 2 / 11 | One slightly-too-big purchase at an odd hour; hard on purpose |

Decisions on the test days: 157 declined (136 fraud, 87% precision), 81 sent to review, 55,457 approved (23 frauds missed).

**Speed** (one core, Python 3.13, 2-vCPU cloud VM): 84,000 payments a second; per payment p50 8.0 µs, p99 35.2 µs, p99.9 105 µs, covering features, score, decision and reasons.

**Skew, measured honestly.** Rebuilding features in event-time order changes 4,401 of 276,650 vectors. On this data, a model trained on those tidy features scored about the same (PR-AUC 0.879 against 0.878), because late arrivals here are random rather than tied to fraud. The guarantee still matters: in production, late data is rarely random.

## Checked on real payments

The public [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (Worldline and the Machine Learning Group of ULB): 284,807 card payments from European cardholders over two days in September 2013, 492 of them fraud. Its columns are anonymised with PCA (V1 to V28, plus time and amount), so there are no card ids or merchants and the per-card feature engine cannot run. What it does test is the rest: the numpy model, the time-ordered split, threshold selection and the alert-budget evaluation, on real traffic.

Trained on the first 60% by time, thresholds tuned on the next 15%, tested on the last 71,202 payments (94 fraud):

| Reviewing 0.5% of payments | Fraud caught | PR-AUC |
| --- | --- | --- |
| **Countersign model** | **81 of 94 (86%)** | **0.765** |
| Flag the largest payments | 1 of 94 (1%) | 0.002 |

Declines: 52 payments, 50 of them fraud (96%). Scoring: about 8 µs a payment.

```bash
# download creditcard.csv from Kaggle (free account), then:
countersign realdata path/to/creditcard.csv
CREDITCARD_CSV=path/to/creditcard.csv pytest     # adds the real-data test
```

The CSV is not in this repository: it is 150 MB and its licence asks users to obtain it from the source.

## How it works

```
 generator ──► arrival-ordered stream ──► bounded queue ──► scorer ──► decision log
 (seeded)       late, out of order,       (backpressure)     │
                duplicated                                    ├─ dedupe by transaction id
                                                              ├─ features as of arrival  ◄── same engine
                                                              ├─ logistic model              replays the
                                                              ├─ approve / review / decline  stream for
                                                              └─ top reasons                 training
```

**Features** (17, per card): spend versus the card's own history (Welford mean and variance), counts and spend in 10-minute and 1-hour windows, bursts of tiny charges, first time at a merchant, new device, travel speed from the last located purchase, distance from the billing address, time of night, channel and category, and history length. Windows are anchored at the payment's own time but only use what had arrived when it was scored.

**Model**: logistic regression trained with numpy (no ML framework), class-weighted, with the intercept corrected so scores read as probabilities. A linear model gives exact reasons: each feature's contribution is weight times standardised value.

**Thresholds** come from the validation days: review at the 0.5% alert budget, decline only where validation precision was at least 90%.

**Evaluation** splits by time, never at random; a random split lets the model peek at the future.

**The browser demo** runs a line-by-line JavaScript port of the engine and model. `npm test` replays the demo stream and checks every score against Python: the largest difference is 5e-13 and no decision differs.

## Run it

```bash
git clone https://github.com/milanpatel98/countersign.git
cd countersign
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

countersign train            # generate, train, evaluate; writes artifacts/
countersign bench            # latency and throughput
countersign stream           # live pipeline; prints every review and decline with reasons
pytest

countersign export           # data for the browser demo
cd web && npm install && npm test && npm run dev    # http://localhost:5180
```

## Project layout

```
countersign/
  gen.py        synthetic cardholders, merchants, four fraud patterns, messy delivery
  features.py   the one feature engine for training and serving
  model.py      logistic regression, decisions, reasons
  evaluate.py   time split, alert-budget metrics, rules and skew baselines
  stream.py     async pipeline: backpressure, dedupe, late handling, latency
  cli.py        train, bench, stream, export
web/src/engine.js   JavaScript port; web/test/parity.mjs checks it against Python
```

## Limitations

- The data is synthetic. Patterns are modelled on how these frauds work, but real traffic is messier and labels arrive weeks late.
- One process holds all card state in memory. A production version would shard state by card and checkpoint it.
- Logistic regression is chosen for exact reasons. A gradient-boosted model would likely catch more of the quiet cases at the cost of simpler explanations.

## License

MIT
