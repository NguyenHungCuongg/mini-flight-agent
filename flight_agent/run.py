"""The single entry point: run(pattern, scenario, model, approver) -> RunResult."""
import time
from dataclasses import dataclass, field

import openai

from flight_agent.core import Harness, StopRun
from flight_agent.hybrid import run_hybrid
from flight_agent.plan_execute import run_plan_execute
from flight_agent.react import run_react

# Transient API errors are not the agent's fault. A bad key or model name (401/404) still crashes.
INFRA_ERRORS = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                openai.InternalServerError)

PATTERNS = {"react": run_react, "plan_execute": run_plan_execute, "hybrid": run_hybrid}


@dataclass
class RunResult:
    pattern: str
    scenario: str
    outcome: str                # done | handoff | infra_error  (decided by the harness)
    expected: str
    stop_reason: str | None
    handoff: dict | None
    trace: list = field(default_factory=list)
    model_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    final_answer: str = ""
    error: str = ""


def run(pattern, scenario, model, approver) -> RunResult:
    harness = Harness(scenario, approver)
    started = time.perf_counter()
    answer, error = "", ""
    try:
        answer = PATTERNS[pattern](harness, model, scenario.constraints) or ""
    except StopRun:
        pass
    except INFRA_ERRORS as e:                 # 429 / timeout / 5xx after the client's retries
        error = f"{type(e).__name__}: {e}"

    if error:
        outcome = "infra_error"
    elif harness.is_done():
        outcome = "done"
    else:
        outcome = "handoff"
        harness.stop("gave_up")
    return RunResult(
        pattern=pattern, scenario=scenario.name, outcome=outcome, expected=scenario.expected,
        stop_reason=harness.stop_reason, handoff=harness.handoff() if outcome == "handoff" else None,
        trace=harness.trace, model_calls=harness.model_calls, tokens=harness.tokens,
        seconds=round(time.perf_counter() - started, 2), final_answer=str(answer), error=error)
