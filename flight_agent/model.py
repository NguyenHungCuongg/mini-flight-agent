"""The real LLM, configured only from .env (OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY)."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def make_model():
    load_dotenv()
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_retries=3,          # 429 / timeout / 5xx: retried with backoff by the client
        timeout=120,
        # Qwen thinking off: OpenRouter reads "reasoning", the UIT vLLM server reads "chat_template_kwargs".
        extra_body={"reasoning": {"enabled": False}, "chat_template_kwargs": {"enable_thinking": False}},
    )
