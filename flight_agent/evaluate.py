"""Run every pattern on every scenario with the real LLM and summarise the results.

Every (pattern, scenario) pair is run 3 times (reps). Each run is appended to results/runs.jsonl as soon as it ends, so the evaluation can be
resumed on another day: runs that already have a result (other than infra_error) are skipped.

Run:  uv run python -m flight_agent.evaluate
"""
import json
from dataclasses import asdict
from pathlib import Path

from flight_agent.core import SCENARIOS, RuleApprover
from flight_agent.run import PATTERNS, run

RESULTS = Path("results")


def load(path: Path) -> dict:
    """Latest record per (pattern, scenario, rep)."""
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            records[(r["pattern"], r["scenario"], r["rep"])] = r
    return records


def evaluate(make_model, path: Path = RESULTS / "runs.jsonl", reps=1,
             patterns=tuple(PATTERNS), scenarios=tuple(SCENARIOS)) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load(path)
    for rep in range(reps):
        for scenario in scenarios:
            for pattern in patterns:
                key = (pattern, scenario, rep)
                if key in records and records[key]["outcome"] != "infra_error":
                    continue
                r = {**asdict(run(pattern, SCENARIOS[scenario], make_model(), RuleApprover())), "rep": rep}
                records[key] = r
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"{pattern:13} {scenario:16} -> {r['outcome']:11} (expected {r['expected']}) "
                      f"stop={r['stop_reason']} calls={r['model_calls']} {r['error'][:80]}")
                if "per-day" in r["error"]:
                    print("Daily free quota used up: stop now, resume tomorrow.")
                    return records
    return records


def summary(records: dict) -> str:
    rows = [r for r in records.values() if r["outcome"] != "infra_error"]
    patterns = list(dict.fromkeys(r["pattern"] for r in records.values()))
    scenarios = list(dict.fromkeys(r["scenario"] for r in records.values()))
    lines = ["## Correct runs per scenario (✓ = every run matched the expected outcome; "
             "in brackets: stop reasons of the wrong runs)", "",
             "| Scenario | Expected | " + " | ".join(patterns) + " |",
             "|---|---|" + "---|" * len(patterns)]
    for s in scenarios:
        cells = []
        for p in patterns:
            rs = [r for r in rows if r["pattern"] == p and r["scenario"] == s]
            wrong = [r for r in rs if r["outcome"] != r["expected"]]
            reasons = ", ".join(sorted({r["stop_reason"] or r["outcome"] for r in wrong}))
            cells.append(f"{'✗' if wrong else '✓'} {len(rs) - len(wrong)}/{len(rs)}"
                         + (f" ({reasons})" if reasons else "") if rs else "-")
        lines.append(f"| {s} | {SCENARIOS[s].expected} | " + " | ".join(cells) + " |")

    lines += ["", "## Per pattern", "",
              "| Pattern | Runs | Correct | Done when expected | Handoff when expected | "
              "Blocked calls | Model calls (avg) | Tokens (avg) | Seconds (avg) | Infra errors |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for p in patterns:
        rs = [r for r in rows if r["pattern"] == p]
        n = len(rs) or 1
        exp_done = [r for r in rs if r["expected"] == "done"]
        exp_hand = [r for r in rs if r["expected"] == "handoff"]
        rate = lambda xs: f"{sum(r['outcome'] == r['expected'] for r in xs)}/{len(xs)}"
        blocked = sum(t["result"]["status"] == "denied" for r in rs for t in r["trace"])
        infra = sum(r["pattern"] == p and r["outcome"] == "infra_error" for r in records.values())
        lines.append(f"| {p} | {len(rs)} | {rate(rs)} | {rate(exp_done)} | {rate(exp_hand)} | {blocked} | "
                     f"{sum(r['model_calls'] for r in rs) / n:.1f} | {sum(r['tokens'] for r in rs) / n:.0f} | "
                     f"{sum(r['seconds'] for r in rs) / n:.1f} | {infra} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from flight_agent.model import make_model
    records = evaluate(make_model, reps=3)
    (RESULTS / "summary.md").write_text(summary(records), encoding="utf-8")
    print(f"\nSummary written to {RESULTS / 'summary.md'}")
