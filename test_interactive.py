#!/usr/bin/env python3
"""Interactive test for user interaction tools - run this in your terminal!"""

import asyncio
from src.orchestrator.tools.user_interaction_tools import (
    UserPromptTool,
    ApprovalGateTool,
    FeedbackCollectionTool
)


async def main():
    """Run interactive tests."""
    print("\n🚀 ORCHESTRATOR USER INTERACTION TOOLS - INTERACTIVE TEST")
    print("="*60)
    print("This will test the user interaction tools with real input.")
    print("Please follow the prompts!\n")
    
    # Test 1: UserPromptTool
    print("TEST 1: UserPromptTool")
    print("-"*30)
    
    prompt_tool = UserPromptTool()
    
    # Simple text
    result = await prompt_tool.execute(
        prompt="What's your name?",
        input_type="text",
        context="cli"
    )
    name = result['value']
    print(f"✓ Hello, {name}!\n")
    
    # Number with validation
    result = await prompt_tool.execute(
        prompt="Pick a number between 1 and 10",
        input_type="number",
        context="cli"
    )
    print(f"✓ You picked: {result['value']}\n")
    
    # Choice
    result = await prompt_tool.execute(
        prompt="What's your favorite season?",
        input_type="choice",
        choices=["Spring", "Summer", "Fall", "Winter"],
        context="cli"
    )
    print(f"✓ Nice choice: {result['value']}\n")
    
    # Test 2: ApprovalGateTool
    print("\nTEST 2: ApprovalGateTool")
    print("-"*30)
    
    approval_tool = ApprovalGateTool()
    
    result = await approval_tool.execute(
        title="Cookie Policy",
        content=f"Hi {name}! This test app wants to save a cookie.\nDo you approve?",
        options=["approve", "reject"],
        context="cli"
    )
    
    if result['approved']:
        print("✓ Cookie approved! 🍪\n")
    else:
        print("✓ Cookie rejected! 🚫\n")
    
    # Test 3: FeedbackCollectionTool
    print("TEST 3: FeedbackCollectionTool")
    print("-"*30)
    
    feedback_tool = FeedbackCollectionTool()
    
    result = await feedback_tool.execute(
        title="Quick Feedback",
        questions=[
            {
                "id": "experience",
                "text": "Rate this test experience (1-5)",
                "type": "rating",
                "scale": 5
            },
            {
                "id": "useful",
                "text": "Was this test useful?",
                "type": "boolean"
            },
            {
                "id": "comments",
                "text": "Any comments? (optional - press Enter to skip)",
                "type": "text",
                "required": False
            }
        ],
        context="cli"
    )
    
    print("\n✓ Feedback collected!")
    print(f"  Rating: {result['responses'].get('experience', 'N/A')}/5")
    print(f"  Useful: {result['responses'].get('useful', 'N/A')}")
    if 'comments' in result['responses']:
        print(f"  Comments: {result['responses']['comments']}")
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETE!")
    print("The user interaction tools are working correctly.")
    print("="*60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure you're running this in an interactive terminal.")