"""ReAct: the model decides ONE step at a time, after each observation.

LangChain's create_agent runs the loop; two middlewares plug in the harness:
every model call is charged to the budget, every tool call goes through Harness.execute().
"""
import json

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage

SYSTEM_PROMPT = ("You are a flight booking agent. Use the tools: search_flights, check_seat, "
                 "book_seat, pay, get_booking. Only book a flight that meets ALL constraints. "
                 "After paying, read the booking back with get_booking. "
                 "If no flight meets them, or an action is denied, stop and say so.")


def run_react(harness, model, constraints):
    @wrap_model_call
    def budget(request, handler):
        harness.before_model()
        response = handler(request)
        for m in response.result:
            harness.after_model(m)
        return response

    @wrap_tool_call
    def through_harness(request, handler):
        call = request.tool_call
        result = harness.execute(call["name"], call["args"])
        return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])

    agent = create_agent(model=model, tools=harness.world.tools, system_prompt=SYSTEM_PROMPT,
                         middleware=[budget, through_harness])
    out = agent.invoke({"messages": [{"role": "user", "content": constraints.to_prompt()}]})
    return out["messages"][-1].content
