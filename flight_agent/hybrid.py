"""Hybrid (ReAct + Plan): plan, run the steps, and when a step fails or is blocked,
the HARNESS asks the model for a new plan with what it just observed (at most MAX_REPLANS).
See docs/adr/0001: the harness, not the model, decides when to replan.
"""
import json

from flight_agent.core import MAX_REPLANS
from flight_agent.plan_execute import approved_plan, execute_plan, search


def run_hybrid(harness, model, constraints):
    flights = search(harness, constraints)
    plan = approved_plan(harness, model, constraints, flights)
    for replans in range(MAX_REPLANS + 1):
        if not execute_plan(harness, plan):
            return
        if replans == MAX_REPLANS:
            break
        feedback = ("Your previous plan failed. Steps run so far and their results: "
                    f"{json.dumps(harness.trace[1:])}. Write a NEW plan from the current state.")
        plan = approved_plan(harness, model, constraints, flights, feedback)
    harness.stop("replans_exhausted")
