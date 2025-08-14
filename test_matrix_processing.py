#!/usr/bin/env python3
"""
Test matrix processing - a real-world nested loop scenario.
Process a matrix of items using separate steps instead of nested for_each.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_matrix_processing():
    """Test processing a matrix of items (regions x products)."""
    
    yaml_content = """
id: matrix-processing
name: Region-Product Matrix Processing
parameters:
  regions: ["US", "EU", "ASIA"]
  products: ["laptop", "phone"]
  output_dir: "examples/outputs/matrix"

steps:
  # Step 1: Analyze each region
  - id: analyze_regions
    for_each: "{{ regions }}"
    steps:
      - id: region_analysis
        action: generate_text
        parameters:
          prompt: |
            Analyze market for region {{ $item }}.
            Provide 3-word summary.
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save_region
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/region_{{ $item }}.txt"
          content: |
            Region: {{ $item }}
            Market Analysis: {{ region_analysis }}
            Products to analyze: {{ products | join(', ') }}
        dependencies:
          - region_analysis
  
  # Step 2: US product analysis
  - id: us_products
    for_each: "{{ products }}"
    steps:
      - id: analyze_us_product
        action: generate_text
        parameters:
          prompt: "Price for {{ $item }} in US market (just number):"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: save_us_product
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/US_{{ $item }}.txt"
          content: |
            Region: US
            Product: {{ $item }}
            Price: {{ analyze_us_product }}
        dependencies:
          - analyze_us_product
    dependencies:
      - analyze_regions
  
  # Step 3: EU product analysis  
  - id: eu_products
    for_each: "{{ products }}"
    steps:
      - id: analyze_eu_product
        action: generate_text
        parameters:
          prompt: "Price for {{ $item }} in EU market (just number with €):"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: save_eu_product
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/EU_{{ $item }}.txt"
          content: |
            Region: EU
            Product: {{ $item }}
            Price: {{ analyze_eu_product }}
        dependencies:
          - analyze_eu_product
    dependencies:
      - analyze_regions
  
  # Step 4: ASIA product analysis
  - id: asia_products
    for_each: "{{ products }}"
    steps:
      - id: analyze_asia_product
        action: generate_text
        parameters:
          prompt: "Price for {{ $item }} in ASIA market (just number with ¥):"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: save_asia_product
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/ASIA_{{ $item }}.txt"
          content: |
            Region: ASIA
            Product: {{ $item }}
            Price: {{ analyze_asia_product }}
        dependencies:
          - analyze_asia_product
    dependencies:
      - analyze_regions
  
  # Step 5: Create summary report
  - id: create_summary
    action: generate_text
    parameters:
      prompt: |
        Create a summary report for:
        Regions: {{ regions | join(', ') }}
        Products: {{ products | join(', ') }}
        Say "Matrix processing complete"
      model: openai/gpt-3.5-turbo
      max_tokens: 50
    dependencies:
      - us_products
      - eu_products
      - asia_products
  
  - id: save_summary
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_dir }}/summary.txt"
      content: |
        Matrix Processing Summary
        =========================
        Regions analyzed: {{ regions | join(', ') }}
        Products analyzed: {{ products | join(', ') }}
        
        Status: {{ create_summary }}
        
        Total combinations: {{ regions | length * products | length }}
    dependencies:
      - create_summary
"""
    
    print("\n" + "="*60)
    print("TEST: Matrix Processing (Region x Product)")
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
    
    output_dir = Path("examples/outputs/matrix")
    
    # Check region files
    print("\n## Region Files:")
    for region in ["US", "EU", "ASIA"]:
        file_path = output_dir / f"region_{region}.txt"
        if file_path.exists():
            content = file_path.read_text()
            has_templates = "{{" in content or "{%" in content
            status = "❌ Has templates" if has_templates else "✅ Clean"
            print(f"  {region}: {status}")
        else:
            print(f"  {region}: ❌ Missing")
    
    # Check product files (the matrix)
    print("\n## Product Matrix Files:")
    matrix_ok = True
    for region in ["US", "EU", "ASIA"]:
        for product in ["laptop", "phone"]:
            file_path = output_dir / f"{region}_{product}.txt"
            if file_path.exists():
                content = file_path.read_text()
                has_templates = "{{" in content or "{%" in content
                if has_templates:
                    print(f"  {region}_{product}: ❌ Has templates")
                    matrix_ok = False
                else:
                    # Show a sample
                    lines = content.strip().split('\n')
                    price_line = [l for l in lines if 'Price:' in l]
                    if price_line:
                        print(f"  {region}_{product}: ✅ {price_line[0].strip()}")
                    else:
                        print(f"  {region}_{product}: ✅ Clean")
            else:
                print(f"  {region}_{product}: ❌ Missing")
                matrix_ok = False
    
    # Check summary
    print("\n## Summary File:")
    summary_path = output_dir / "summary.txt"
    if summary_path.exists():
        content = summary_path.read_text()
        has_templates = "{{" in content or "{%" in content
        if has_templates:
            print(f"  ❌ Summary has unrendered templates")
        else:
            print(f"  ✅ Summary clean")
            print("\n--- Summary Content ---")
            print(content)
    else:
        print(f"  ❌ Summary missing")
    
    return matrix_ok


async def main():
    """Run matrix processing test."""
    print("\n" + "="*70)
    print("MATRIX PROCESSING - Alternative to Nested Loops")
    print("="*70)
    
    print("\nThis demonstrates processing a matrix (regions x products)")
    print("using separate steps for each 'row' of the matrix.")
    print("This avoids nested for_each loops entirely.")
    
    success = await test_matrix_processing()
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    
    if success:
        print("✅ Matrix processing successful!")
        print("\nThis approach works well for:")
        print("- Known dimensions (regions, products)")
        print("- When you need different processing per 'row'")
        print("- Avoiding nested loop template issues")
    else:
        print("❌ Some issues with matrix processing")
    
    print("\nRecommendation: Use separate steps with dependencies")
    print("instead of nested for_each loops for complex scenarios.")


if __name__ == "__main__":
    asyncio.run(main())