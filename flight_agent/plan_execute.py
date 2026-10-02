"""Plan-then-execute: the model writes the WHOLE plan in one call, the approver reviews it,
then plain code runs the steps in order WITHOUT asking the model again.

    + The plan is visible before anything runs.
    - If a step fails, the plan cannot adapt: stop and hand off.
"""
import json

from langchain_core.prompts import ChatPromptTemplate

from flight_agent.core import MAX_PLANS, Plan, StopRun

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You plan a flight booking. Write the WHOLE plan at once, as a list of steps. "
     "Tools: check_seat(flight), book_seat(flight), pay(code), get_booking(code). "
     "The booking code is not known yet: write \"$booking_code\" and it will be filled in. "
     "End the plan with get_booking to read the booking back. "
     "If no flight meets ALL constraints, return an empty plan."),
    ("human", "Goal: {goal}\nAvailable flights: {flights}\n{feedback}"),
])


def search(harness, constraints) -> list:
    """Read-only search, arguments taken from the constraint DATA (not from the model)."""
    c = constraints
    found = harness.execute("search_flights", {"origin": c.origin, "destination": c.destination, "date": c.date})
    return found.get("flights", [])


def make_plan(harness, model, constraints, flights, feedback="") -> Plan:
    harness.before_model()
    planner = PLANNER_PROMPT | model.with_structured_output(Plan, method="function_calling", include_raw=True)
    out = planner.invoke({"goal": constraints.to_prompt(), "flights": json.dumps(flights), "feedback": feedback})
    harness.after_model(out["raw"])
    if out["parsed"] is None:
        harness.stop("invalid_plan")
        raise StopRun
    return out["parsed"]


def approved_plan(harness, model, constraints, flights, feedback="") -> Plan:
    """Ask for a plan until the approver accepts one (at most MAX_PLANS)."""
    for _ in range(MAX_PLANS):
        plan = make_plan(harness, model, constraints, flights, feedback)
        if not plan.steps:                      # the model says no flight fits
            harness.stop("gave_up")
            raise StopRun
        if harness.approver.approve_plan(plan, harness.scenario):
            return plan
        feedback += (f" The reviewer REJECTED this plan: {plan.model_dump_json()}. "
                     "Every booked flight must meet ALL constraints.")
    harness.stop("plan_rejected")
    raise StopRun


def fill_placeholders(harness, args: dict) -> dict:
    """Replace "$booking_code" with the code returned by the last successful book_seat."""
    code = next((t["result"]["code"] for t in reversed(harness.trace)
                 if t["tool"] == "book_seat" and t["result"]["status"] == "ok"), "")
    return {k: (code if v == "$booking_code" else v) for k, v in args.items()}


def execute_plan(harness, plan) -> dict | None:
    """Run the steps in order. Return the result of the first failed step, or None."""
    for step in plan.steps:
        result = harness.execute(step.tool, fill_placeholders(harness, step.args))
        if harness.stop_reason:                 # loop / needs approval: the harness ends the run
            raise StopRun
        if result["status"] != "ok":
            return result
    return None


def run_plan_execute(harness, model, constraints):
    flights = search(harness, constraints)
    plan = approved_plan(harness, model, constraints, flights)
    if execute_plan(harness, plan):
        harness.stop("step_failed")
