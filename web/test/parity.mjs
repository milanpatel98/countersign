// Replays the demo world through the JavaScript engine and checks every score
// and decision against what Python computed. Run: npm test (from web/).
import { readFileSync } from "node:fs";
import { scoreStream } from "../src/engine.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const world = read("../public/data/world.json");
const model = read("../public/data/model.json");
const expected = read("./expected.json");

const js = scoreStream(world, model, {}).filter((r) => !r.duplicate);
if (js.length !== expected.length) {
  console.error(`count mismatch: js ${js.length}, python ${expected.length}`);
  process.exit(1);
}

let worst = 0;
let actionMismatches = 0;
for (let i = 0; i < js.length; i++) {
  const [id, p, action] = expected[i];
  if (js[i].id !== id) {
    console.error(`order mismatch at ${i}: js ${js[i].id}, python ${id}`);
    process.exit(1);
  }
  worst = Math.max(worst, Math.abs(js[i].score - p));
  if (js[i].action !== action) actionMismatches++;
}

console.log(`${js.length} transactions: largest score difference ${worst.toExponential(2)}, decision mismatches ${actionMismatches}`);
if (worst > 1e-9 || actionMismatches > 0) process.exit(1);
