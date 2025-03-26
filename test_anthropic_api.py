# test_anthropic_api.py
import os
from anthropic import Anthropic, AnthropicError

# Load API key from environment
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("Error: ANTHROPIC_API_KEY not set in environment.")
    exit(1)

# Initialize Anthropic client
claude = Anthropic(api_key=api_key)

# List of model names to test
model_names = [
    "claude-3.5-sonnet-20240620",  # Known version from June 2024
    # "claude-3-opus-20240229",      # Older but possibly still active
    "claude-3-haiku-20240307",     # Lightweight model
    # "claude-3.5-haiku-20241022",   # Newer Haiku version
    "claude-3.5-sonnet-latest",    # Hypothetical alias for latest Sonnet
]

# Test each model
for model in model_names:
    print(f"\nTrying model: {model}")
    try:
        response = claude.messages.create(
            model=model,
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Hello, can you confirm you’re working?"}
            ]
        )
        print("Success! API response:")
        print(response.content[0].text)
        print(f"Working model: {model}")
        break  # Stop once we find a working model
    except AnthropicError as e:
        print(f"API Error: {e}")
        print(f"Status Code: {e.status_code if hasattr(e, 'status_code') else 'Unknown'}")
        print(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
    except Exception as e:
        print(f"Unexpected Error: {e}")