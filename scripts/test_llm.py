"""Test script to verify DeepSeek LLM connection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm_service import LLMService


def main():
    print("Testing DeepSeek LLM connection...")
    print("=" * 50)
    
    try:
        # Initialize LLM service
        llm = LLMService()
        print(f"✓ LLM Service initialized")
        print(f"  Model: {llm.model_name}")
        
        # Test simple generation
        print("\nSending test prompt...")
        response = llm.generate(
            prompt="What is 2+2? Answer in one word.",
            system_prompt="You are a helpful assistant."
        )
        
        print(f"\n✓ Response received:")
        print(f"  Content: {response.content}")
        print(f"  Prompt tokens: {response.prompt_tokens}")
        print(f"  Completion tokens: {response.completion_tokens}")
        print(f"  Total tokens: {response.total_tokens}")
        
        print("\n" + "=" * 50)
        print("✓ LLM connection successful!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
