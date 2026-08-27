"""
NIM Client - wrapper around NVIDIA Build (build.nvidia.com) OpenAI-compatible API.
Swap base_url to http://localhost:11434/v1 and model to a local one -> same code runs on Ollama.
"""
import os
import json
import time

from dotenv import load_dotenv
load_dotenv()  # reads .env from project root; real env vars still take precedence

from openai import OpenAI

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_MODEL = os.getenv("NIM_MODEL", "deepseek-ai/deepseek-v4-pro-0813")


class NimClient:
    def __init__(self, model: str = NIM_MODEL, temperature: float = 0.1):
        if not NIM_API_KEY and "nvidia" in NIM_BASE_URL:
            raise RuntimeError("Set NVIDIA_API_KEY env var (get one free at build.nvidia.com)")
        self.client = OpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY or "ollama")
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str, max_tokens: int = 2048, retries: int = 3) -> str:
        """Plain chat completion with retry/backoff for the 40 RPM free-tier limit."""
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return resp.choices[0].message.content
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    wait = 20 * (attempt + 1)
                    print(f"  [NIM] rate-limited, backing off {wait}s ...")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("NIM API: exhausted retries (rate limit)")

    def chat_json(self, system: str, user: str, max_tokens: int = 2048) -> dict:
        """Chat completion that must return JSON. Strips markdown fences defensively."""
        raw = self.chat(
            system + "\nRespond ONLY with valid JSON. No markdown, no preamble.",
            user,
            max_tokens=max_tokens,
        )
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        # last-resort: find first { ... last }
        if not clean.strip().startswith(("{", "[")):
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1:
                clean = clean[start : end + 1]
        return json.loads(clean)
