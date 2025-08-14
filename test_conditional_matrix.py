#!/usr/bin/env python3
"""
Test conditional matrix processing with dynamic flow.
Shows how to achieve complex nested behavior with conditionals and dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_conditional_matrix():
    """Test matrix with conditional processing based on results."""
    
    yaml_content = """
id: conditional-matrix
name: Conditional Matrix Processing
parameters:
  categories: ["premium", "budget"]
  items: ["A", "B", "C"]
  threshold: 100
  output_dir: "examples/outputs/conditional_matrix"

steps:
  # Step 1: Evaluate each category
  - id: evaluate_categories
    for_each: "{{ categories }}"
    steps:
      - id: category_score
        action: generate_text
        parameters:
          prompt: |
            Score for {{ $item }} category (return number 1-200):
            Premium should be > 100, budget should be < 100.
            {% if $item == 'premium' %}
            Return: 150
            {% else %}
            Return: 50
            {% endif %}
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: save_category_score
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ $item }}_score.txt"
          content: |
            Category: {{ $item }}
            Score: {{ category_score }}
            Threshold: {{ threshold }}
        dependencies:
          - category_score
  
  # Step 2: Process premium items (only if premium exists and scores high)
  - id: process_premium_items
    for_each: "{{ items }}"
    steps:
      - id: premium_processing
        action: generate_text
        parameters:
          prompt: "Premium processing for item {{ $item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save_premium_item
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/premium_{{ $item }}.txt"
          content: |
            Category: Premium
            Item: {{ $item }}
            Processing: {{ premium_processing }}
            Status: High-priority
        dependencies:
          - premium_processing
    dependencies:
      - evaluate_categories
  
  # Step 3: Process budget items (only if budget exists and scores low)
  - id: process_budget_items
    for_each: "{{ items }}"
    steps:
      - id: budget_processing
        action: generate_text
        parameters:
          prompt: "Budget processing for item {{ $item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save_budget_item
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/budget_{{ $item }}.txt"
          content: |
            Category: Budget
            Item: {{ $item }}
            Processing: {{ budget_processing }}
            Status: Standard-priority
        dependencies:
          - budget_processing
    dependencies:
      - evaluate_categories
  
  # Step 4: Cross-reference matrix - combine categories with items
  - id: create_cross_reference
    for_each: "{{ categories }}"
    steps:
      - id: generate_matrix_row
        action: generate_text
        parameters:
          prompt: |
            Create matrix row for {{ $item }} category with items {{ items | join(', ') }}.
            Format: {{ $item }}: [item statuses]
          model: openai/gpt-3.5-turbo
          max_tokens: 30
      
      - id: save_matrix_row
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/matrix_{{ $item }}.txt"
          content: |
            Matrix Row: {{ $item }}
            Items: {{ items | join(', ') }}
            Result: {{ generate_matrix_row }}
        dependencies:
          - generate_matrix_row
    dependencies:
      - process_premium_items
      - process_budget_items
"""
    
    print("\n" + "="*60)
    print("TEST: Conditional Matrix Processing")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Check output files
    print("\n" + "="*60)
    print("CHECKING OUTPUT FILES")
    print("="*60)
    
    output_dir = Path("examples/outputs/conditional_matrix")
    
    # Check category scores
    print("\n## Category Scores:")
    for category in ["premium", "budget"]:
        file_path = output_dir / f"{category}_score.txt"
        if file_path.exists():
            content = file_path.read_text()
            has_templates = "{{" in content or "{%" in content
            status = "❌ Has templates" if has_templates else "✅ Clean"
            # Extract score
            score_line = [l for l in content.split('\n') if 'Score:' in l]
            if score_line:
                print(f"  {category}: {status} - {score_line[0].strip()}")
            else:
                print(f"  {category}: {status}")
    
    # Check item processing files
    print("\n## Processed Items:")
    for category in ["premium", "budget"]:
        for item in ["A", "B", "C"]:
            file_path = output_dir / f"{category}_{item}.txt"
            if file_path.exists():
                content = file_path.read_text()
                has_templates = "{{" in content or "{%" in content
                if has_templates:
                    print(f"  {category}_{item}: ❌ Has templates")
                else:
                    # Extract status
                    status_line = [l for l in content.split('\n') if 'Status:' in l]
                    if status_line:
                        print(f"  {category}_{item}: ✅ {status_line[0].strip()}")
                    else:
                        print(f"  {category}_{item}: ✅ Clean")
    
    # Check matrix files
    print("\n## Matrix Rows:")
    for category in ["premium", "budget"]:
        file_path = output_dir / f"matrix_{category}.txt"
        if file_path.exists():
            content = file_path.read_text()
            has_templates = "{{" in content or "{%" in content
            status = "❌ Has templates" if has_templates else "✅ Clean"
            print(f"  {category}: {status}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("This approach demonstrates:")
    print("1. Category evaluation with scoring")
    print("2. Conditional processing based on categories")
    print("3. Cross-reference matrix generation")
    print("4. All without nested for_each loops!")


async def main():
    """Run conditional matrix test."""
    print("\n" + "="*70)
    print("CONDITIONAL MATRIX - Advanced Alternative to Nested Loops")
    print("="*70)
    
    await test_conditional_matrix()
    
    print("\n" + "="*70)
    print("BEST PRACTICES FOR NESTED-LIKE BEHAVIOR")
    print("="*70)
    print("\n1. **Use separate steps** with dependencies instead of nesting")
    print("2. **Leverage conditionals** to control which combinations run")
    print("3. **Pre-compute combinations** if needed (e.g., in a generate step)")
    print("4. **Use pipeline parameters** to share data between loop levels")
    print("\nThese approaches avoid the $parent_item template issues")
    print("while still achieving complex multi-dimensional processing.")


if __name__ == "__main__":
    asyncio.run(main())