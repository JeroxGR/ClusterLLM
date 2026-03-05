#!/usr/bin/env python3
"""Quick test to verify your LLM provider is configured correctly."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from llm_client import get_client, get_model_name, get_provider, delayed_completion

def main():
    provider = get_provider()
    model = get_model_name()
    print(f"Provider: {provider}")
    print(f"Model:    {model}")
    print(f"Testing connection...")

    completion, error = delayed_completion(
        delay_in_seconds=0,
        max_trials=1,
        model=model,
        messages=[{"role": "user", "content": "Reply with just the word 'hello'."}],
        max_tokens=10,
        temperature=0,
    )

    if completion is None:
        print(f"FAILED: {error}")
        sys.exit(1)
    else:
        content = completion.choices[0].message.content.strip()
        print(f"Response: {content}")
        print("SUCCESS - your LLM provider is working!")

if __name__ == "__main__":
    main()
