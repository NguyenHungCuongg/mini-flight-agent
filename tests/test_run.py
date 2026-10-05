"""Behaviour of the agent, tested only through run() with a scripted fake model."""
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import pytest

from flight_agent.core import SCENARIOS, Plan, RuleApprover, Step
from flight_agent.run import run


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed list of replies. A reply may be an AIMessage, a Plan, or an Exception."""

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: {"raw": AIMessage(content=""), "parsed": self._next()})

    def _next(self):
        item = next(self.messages)
        if isinstance(item, Exception):
            raise item
        return item

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next())])


def scripted(*replies):
    return ScriptedModel(messages=iter(replies))


_ids = iter(range(10**6))


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{next(_ids)}"}])


SEARCH = dict(origin="SGN", destination="DAD", date="2026-10-07")


def book_pay_check(flight, seat="12A"):
    code = f"{flight}-{seat}"
    return [call("book_seat", flight=flight), call("pay", code=code), call("get_booking", code=code)]


def plan(*steps):
    return Plan(steps=[Step(tool=t, args=a) for t, a in steps])


BOOK_PAY = lambda flight: plan(("book_seat", {"flight": flight}),
                               ("pay", {"code": "$booking_code"}),
                               ("get_booking", {"code": "$booking_code"}))


def go(pattern, scenario, *replies):
    return run(pattern, SCENARIOS[scenario], scripted(*replies), RuleApprover())


# ---------------------------------------------------------------- ReAct
def test_react_books_valid_flight():
    r = go("react", "valid", call("search_flights", **SEARCH), *book_pay_check("VN122"),
           AIMessage(content="Booked VN122."))
    assert r.outcome == "done"
    assert r.expected == "done"
    assert r.model_calls == 5


def test_harness_blocks_booking_that_breaks_constraints():
    r = go("react", "trap", call("search_flights", **SEARCH), call("book_seat", flight="VJ610"),
           AIMessage(content="I could not book."))
    blocked = [t for t in r.trace if t["result"]["status"] == "denied"]
    assert [t["args"]["flight"] for t in blocked] == ["VJ610"]
    assert r.outcome == "handoff"
    assert "VJ610" not in str(r.handoff["done_so_far"])


def test_model_claiming_success_without_paid_booking_is_not_done():
    r = go("react", "valid", call("search_flights", **SEARCH), call("book_seat", flight="VN122"),
           AIMessage(content="Booked and paid VN122, all done!"))
    assert r.outcome == "handoff"
    assert r.handoff["done_so_far"] == ["VN122-12A: held, paid=False"]


def test_repeating_the_same_call_stops_the_run_as_a_loop():
    r = go("react", "valid", *[call("check_seat", flight="VN122") for _ in range(5)], AIMessage(content="?"))
    assert r.stop_reason == "loop"
    assert r.outcome == "handoff"
    assert len(r.trace) == 3


def test_polling_get_booking_is_not_a_loop():
    r = go("react", "valid", call("search_flights", **SEARCH), *book_pay_check("VN122"),
           *[call("get_booking", code="VN122-12A") for _ in range(3)], AIMessage(content="Booked."))
    assert r.stop_reason is None
    assert r.outcome == "done"


def test_budget_of_ten_model_calls_ends_the_run():
    flights = ["VN122", "QH118", "XX1", "XX2", "XX3", "XX4", "XX5", "XX6", "XX7", "XX8", "XX9"]
    r = go("react", "valid", *[call("check_seat", flight=f) for f in flights])
    assert r.stop_reason == "budget"
    assert r.model_calls == 10
    assert r.outcome == "handoff"


def test_pay_above_limit_needs_approval_and_hands_off_when_refused():
    r = go("react", "needs_approval", call("search_flights", **SEARCH), *book_pay_check("VN122"),
           AIMessage(content="Done."))
    assert r.stop_reason == "needs_approval"
    assert r.outcome == "handoff"
    assert "1,850,000" in r.handoff["question"]
    assert r.trace[-1]["tool"] == "pay" and r.trace[-1]["result"]["status"] == "denied"


def test_pay_above_limit_goes_ahead_when_the_approver_agrees():
    r = go("react", "approved_payment", call("search_flights", **SEARCH), *book_pay_check("VN122"),
           AIMessage(content="Done."))
    assert r.stop_reason is None
    assert r.outcome == "done"


def test_injected_note_in_search_results_cannot_unlock_a_bad_booking():
    r = go("react", "injection", call("search_flights", **SEARCH), call("book_seat", flight="VJ610"),
           *book_pay_check("VN122"), AIMessage(content="Booked VN122."))
    assert "waived" in str(r.trace[0]["result"])            # the model did see the injection
    assert [t["result"]["status"] for t in r.trace if t["tool"] == "book_seat"] == ["denied", "ok"]
    assert r.outcome == "done"


def test_api_errors_are_infra_errors_not_agent_failures():
    import httpx, openai
    req = httpx.Request("POST", "https://example.test")
    err = openai.RateLimitError("429", response=httpx.Response(429, request=req), body=None)
    r = go("react", "valid", err)
    assert r.outcome == "infra_error"
    assert "RateLimitError" in r.error


def test_runs_do_not_share_bookings():
    go("react", "valid", call("search_flights", **SEARCH), *book_pay_check("VN122"), AIMessage(content="ok"))
    r = go("react", "valid", AIMessage(content="Already booked."))
    assert r.outcome == "handoff"


# ---------------------------------------------------------------- Plan-then-Execute
def test_plan_then_execute_books_valid_flight_with_one_model_call():
    r = go("plan_execute", "valid", BOOK_PAY("VN122"))
    assert r.outcome == "done"
    assert r.model_calls == 1
    assert [t["tool"] for t in r.trace] == ["search_flights", "book_seat", "pay", "get_booking"]


def test_plan_then_execute_cannot_adapt_when_a_step_fails():
    r = go("plan_execute", "sold_out", BOOK_PAY("VN122"))
    assert r.outcome == "handoff"
    assert r.stop_reason == "step_failed"
    assert r.trace[-1]["result"]["status"] == "sold_out"


def test_rejected_plan_gets_one_more_try_then_empty_plan_hands_off():
    r = go("plan_execute", "no_valid", BOOK_PAY("VJ604"), plan())
    assert r.outcome == "handoff"
    assert r.model_calls == 2
    assert not any(t["tool"] == "book_seat" for t in r.trace)    # rejected plan never ran
    assert "relax" in r.handoff["question"]


def test_two_rejected_plans_hand_off():
    r = go("plan_execute", "no_valid", BOOK_PAY("VJ604"), BOOK_PAY("VJ604"))
    assert r.stop_reason == "plan_rejected"
    assert r.outcome == "handoff"


# ---------------------------------------------------------------- Hybrid
def test_hybrid_replans_when_the_flight_sells_out():
    r = go("hybrid", "sold_out", BOOK_PAY("VN122"), BOOK_PAY("VJ124"))
    assert r.outcome == "done"
    assert r.model_calls == 2
    assert [t["result"]["status"] for t in r.trace if t["tool"] == "book_seat"] == ["sold_out", "ok"]


def test_hybrid_retries_payment_after_a_transient_error():
    retry_pay = plan(("pay", {"code": "VN122-12A"}), ("get_booking", {"code": "VN122-12A"}))
    r = go("hybrid", "transient_error", BOOK_PAY("VN122"), retry_pay)
    assert r.outcome == "done"


def test_plan_then_execute_hands_off_on_the_same_transient_error():
    r = go("plan_execute", "transient_error", BOOK_PAY("VN122"))
    assert r.outcome == "handoff"
    assert r.handoff["done_so_far"] == ["VN122-12A: held, paid=False"]


def test_hybrid_stops_after_max_replans():
    r = go("hybrid", "sold_out", BOOK_PAY("VN122"),
           plan(("check_seat", {"flight": "XX1"})), plan(("check_seat", {"flight": "XX2"})))
    assert r.stop_reason == "replans_exhausted"
    assert r.outcome == "handoff"
    assert r.model_calls == 3


# ---------------------------------------------------------------- all patterns
@pytest.mark.parametrize("pattern,replies", [
    ("react", [call("search_flights", **SEARCH), *book_pay_check("VN122"), AIMessage(content="Booked.")]),
    ("plan_execute", [BOOK_PAY("VN122")]),
    ("hybrid", [BOOK_PAY("VN122")]),
])
def test_every_pattern_books_the_valid_flight(pattern, replies):
    assert go(pattern, "valid", *replies).outcome == "done"


# ---------------------------------------------------------------- evaluation
def test_evaluation_saves_each_run_and_skips_it_next_time(tmp_path):
    from flight_agent.evaluate import evaluate, summary

    path = tmp_path / "runs.jsonl"
    records = evaluate(lambda: scripted(BOOK_PAY("VN122")), path, patterns=("plan_execute",), scenarios=("valid",))
    assert records[("plan_execute", "valid", 0)]["outcome"] == "done"

    def must_not_call():
        raise AssertionError("a finished run must not call the model again")
    evaluate(must_not_call, path, patterns=("plan_execute",), scenarios=("valid",))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert "✓ 1/1" in summary(records)


def test_summary_counts_correct_runs_over_repetitions(tmp_path):
    from flight_agent.evaluate import evaluate, summary

    records = evaluate(lambda: scripted(BOOK_PAY("VN122")), tmp_path / "runs.jsonl", reps=2,
                       patterns=("plan_execute",), scenarios=("valid", "sold_out"))
    assert "✓ 2/2" in summary(records)                                 # valid
    assert "✗ 0/2 (step_failed)" in summary(records)                   # sold_out


def test_handoff_after_failed_replans_asks_about_the_failure_not_about_relaxing():
    r = go("hybrid", "sold_out", BOOK_PAY("VN122"),
           plan(("check_seat", {"flight": "XX1"})), plan(("check_seat", {"flight": "XX2"})))
    assert "relax" not in r.handoff["question"]


def test_auth_errors_are_bugs_not_infra_errors():
    import httpx, openai
    req = httpx.Request("POST", "https://example.test")
    err = openai.AuthenticationError("401", response=httpx.Response(401, request=req), body=None)
    with pytest.raises(openai.AuthenticationError):
        go("react", "valid", err)
