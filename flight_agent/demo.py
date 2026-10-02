"""Interactive demo with the real LLM and a human approver.

Run:  uv run python -m flight_agent.demo
"""
import json

from flight_agent.core import SCENARIOS, HumanApprover
from flight_agent.model import make_model
from flight_agent.run import PATTERNS, run


def choose(title, options):
    print(title)
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    choice = ""
    while not (choice.isdigit() and 1 <= int(choice) <= len(options)):
        choice = input(f"Your choice (1-{len(options)}): ").strip()
    return options[int(choice) - 1]


if __name__ == "__main__":
    pattern = choose("Pattern:", list(PATTERNS))
    name = choose("Scenario:", [f"{s.name}: {s.description}" for s in SCENARIOS.values()]).split(":")[0]
    scenario = SCENARIOS[name]

    print(f"\n=== {pattern} / {name} ===")
    print("Constraints (data):", scenario.constraints)
    print("User's request:", scenario.constraints.to_prompt())
    r = run(pattern, scenario, make_model(), HumanApprover())

    print("\n--- trace ---")
    for i, t in enumerate(r.trace, 1):
        print(f"[{i}] {t['tool']}({t['args']}) -> {t['result']['status']} {t['result'].get('reason', '')}")
    print("\n--- result (decided by the harness, not the model) ---")
    print(json.dumps({"outcome": r.outcome, "stop_reason": r.stop_reason, "handoff": r.handoff,
                      "model_said": r.final_answer, "model_calls": r.model_calls, "tokens": r.tokens,
                      "error": r.error}, indent=2, ensure_ascii=False))
