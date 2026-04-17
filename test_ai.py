#!/usr/bin/env python3
"""Test script to verify AI connection works"""

import asyncio
import json
import aiohttp

NVIDIA_BASE_URL = "https://nvidia-key-rotation-proxy-ts.vercel.app/v1"
NVIDIA_MODEL = "moonshotai/kimi-k2.5"


async def test_ai():
    print("Testing NVIDIA AI connection...")
    print(f"Base URL: {NVIDIA_BASE_URL}")
    print(f"Model: {NVIDIA_MODEL}")
    print()

    async with aiohttp.ClientSession() as session:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": NVIDIA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": "Say hello and confirm you can analyze market data. Respond as JSON: {\"message\": \"your message\"}"
                },
            ],
            "temperature": 0.3,
            "max_tokens": 100,
        }

        try:
            async with session.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    print(f"\nAI Response:\n{content}")
                    print("\n✅ AI connection successful!")
                else:
                    error = await response.text()
                    print(f"❌ Error: {error}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_ai())
