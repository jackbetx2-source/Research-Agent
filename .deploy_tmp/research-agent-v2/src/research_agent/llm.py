from __future__ import annotations

import os
import asyncio

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError


class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Set it in .env, .env.example, or the server environment."
            )

        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.default_model = os.getenv("OPENAI_MODEL")
        if not self.default_model:
            raise RuntimeError("Missing OPENAI_MODEL. Set it in .env or the environment.")

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        selected_model = model or self.default_model
        extra_body = None
        if selected_model.startswith("deepseek-v4"):
            extra_body = {
                "thinking": {
                    "type": os.getenv("DEEPSEEK_THINKING", "disabled"),
                }
            }

        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=90,
                    extra_body=extra_body,
                )
                break
            except (APIConnectionError, APITimeoutError, RateLimitError) as error:
                last_error = error
                await asyncio.sleep(2**attempt)
            except APIStatusError as error:
                last_error = error
                if error.status_code < 500:
                    raise
                await asyncio.sleep(2**attempt)
        else:
            raise RuntimeError("LLM request failed after 3 attempts.") from last_error

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return content.strip()
