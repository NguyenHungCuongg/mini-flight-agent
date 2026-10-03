# Flight Booking Agent: a harness and three agent design patterns

Mini agent project from SE373 (Agentic AI Systems Engineering, UIT): a flight booking agent built with LangChain.

`Agent = Model + Harness`. The model only proposes the next step. A harness written in Python decides whether a tool call may run, when to stop, and whether the task is really done. The same harness is shared by three patterns, **ReAct**, **Plan-then-Execute** and **Hybrid**, which are compared on six scenarios.

The full report (in Vietnamese) is in [REPORT.md](REPORT.md).

## Results at a glance

18 runs (3 patterns × 6 scenarios) with `qwen/qwen3.8-27b:free`:

| Pattern           | Correct | Model calls (avg) | Tokens (avg) |
| ----------------- | ------- | ----------------- | ------------ |
| ReAct             | 6/6     | 5.3               | 5,766        |
| Plan-then-Execute | 4/6     | 1.0               | 640          |
| Hybrid            | 6/6     | 1.3               | 908          |

Details: [`results/summary.md`](results/summary.md). Per-run traces: [`results/runs.jsonl`](results/runs.jsonl).

## What the harness does

| Layer                | Implementation                                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Constraints as data  | `Constraints` builds the prompt and checks every flight with **one** rule, `is_ok()`                                                  |
| Done checked by code | `Harness.is_done()` reads the bookings back; `done` only when a booking is paid and meets the constraints                             |
| Permission check     | `Harness.check_permission()` runs before the tool: blocks bookings that break the constraints; payments above the limit need approval |
| Handoff              | `Harness.handoff()`: what was done, what was tried, and one concrete question for a human                                             |
| Extra                | A budget of 10 model calls, loop detection, and transient API errors (`infra_error`) kept apart from agent failures                   |

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python ≥ 3.12 if it is missing.

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with an OpenAI-compatible endpoint:

```dotenv
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=qwen/qwen3.8-27b:free
```

To use the university server (reachable only from the UIT network), set `OPENAI_BASE_URL=https://llm.uit.edu.vn/qwen/v1` and `OPENAI_MODEL=qwen3.8-27b`. No code change is needed.

## Run

```bash
uv run pytest -q                          # 23 tests, fake model: no network, no quota
uv run pytest -q -k hybrid                # one group of tests
uv run python -m flight_agent.demo        # interactive demo: pick a pattern + scenario, you are the approver
uv run python -m flight_agent.evaluate    # evaluate 3 patterns x 6 scenarios -> results/
```

> ⚠️ `demo` and `evaluate` call the real LLM. OpenRouter's free tier allows **50 requests/day**. One demo run uses 1–7 requests; one full `evaluate` pass uses about 50. `evaluate` saves each run as soon as it ends: rerunning it skips runs that already have a result and retries runs that hit an API error (`infra_error`).

## Scenarios

Every scenario uses the same request: **SGN → DAD, 2026-10-07, departing before 12:00, price ≤ 2,000,000 VND**. The flight data is mock data, defined in `SCENARIOS` in `flight_agent/core.py`.

| Scenario          | Situation                                            | Expected  |
| ----------------- | ---------------------------------------------------- | --------- |
| `valid`           | A valid flight exists                                | `done`    |
| `no_valid`        | No flight meets all constraints                      | `handoff` |
| `trap`            | The cheapest flight leaves in the afternoon          | `done`    |
| `transient_error` | The payment gateway fails once, then works           | `done`    |
| `sold_out`        | The best flight sells out between search and booking | `done`    |
| `needs_approval`  | The only valid flight is above the auto-pay limit    | `handoff` |

## Project layout

```
flight_agent/
  core.py          Constraints, scenarios, mock tools (World), approvers, Harness
  run.py           run(pattern, scenario, model, approver) -> RunResult: the single entry point
  react.py         ReAct: create_agent + middleware that plugs in the harness
  plan_execute.py  Plan-then-Execute: one model call writes the plan, code runs the steps
  hybrid.py        Hybrid: same as above, replans when a step fails
  model.py         ChatOpenAI configured from .env
  demo.py          interactive demo
  evaluate.py      runs the evaluation, builds the result tables
tests/test_run.py  23 tests through run() with a fake model
results/           evaluation results (runs.jsonl, summary.md)
REPORT.md          lab report (Vietnamese)
CONTEXT.md         glossary (Vietnamese)
```

The demo, the evaluation and the tests all call `run()` and nothing else. Patterns never check constraints or decide "done" themselves; the harness always decides the outcome.

## Adding a scenario

Add a `Scenario(...)` to `SCENARIOS` in `flight_agent/core.py`:

```python
Scenario("my_case", "Short description.",
         [_f("VN122", "08:10", 1_850_000), _f("QH118", "15:40", 1_640_000)],
         expected="done",            # or "handoff"
         sold_out=(), fail_first=(), approval_limit=None)
```

The demo and `evaluate` pick up the new scenario automatically.
