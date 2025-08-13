#!/usr/bin/env python3
"""
Debug test for control_flow_advanced translation templates.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_translation_templates():
    """Test a simplified version of the translation pipeline."""
    
    yaml_content = """
id: debug-translation
name: Debug Translation Templates
parameters:
  text: "Hello world"
  languages: ["es"]
  output_dir: "examples/outputs/debug_translation"
steps:
  - id: translate_text
    for_each: "{{ languages }}"
    steps:
      - id: translate
        action: generate_text
        parameters:
          prompt: "Translate '{{ text }}' to {{ $item }}. Reply with just the translation."
          model: openai/gpt-3.5-turbo
          max_tokens: 50
      
      - id: validate
        action: generate_text
        parameters:
          prompt: "Rate this {{ $item }} translation quality (1-10): {{ translate }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
        dependencies:
          - translate
      
      - id: save_translation
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ $item }}.txt"
          content: |
            Language: {{ $item }}
            Original: {{ text }}
            Translation: {{ translate }}
            Validation: {{ validate }}
        dependencies:
          - validate
"""
    
    print("\n" + "="*60)
    print("DEBUG TEST: Translation Pipeline Templates")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Enable debug logging
    logging.getLogger("orchestrator.tools.system_tools").setLevel(logging.DEBUG)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Check output file
    print("\n" + "="*60)
    print("CHECKING OUTPUT FILE")
    print("="*60)
    
    file_path = Path("examples/outputs/debug_translation/es.txt")
    
    if file_path.exists():
        content = file_path.read_text()
        print(f"\n--- es.txt ---")
        print(content)
        print("-" * 40)
        
        # Check for templates
        if "{{" in content:
            print(f"❌ FAIL: File contains unrendered templates!")
            import re
            templates = re.findall(r'{{.*?}}', content)
            print(f"   Unrendered templates: {templates}")
        else:
            print(f"✅ PASS: No template placeholders")
            
        # Check content
        checks = [
            ("Language: es", "language header"),
            ("Original: Hello world", "original text"),
            ("Translation:", "translation section"),
            ("Validation:", "validation section")
        ]
        
        for check_text, check_name in checks:
            if check_text in content:
                print(f"✅ PASS: {check_name} present")
            else:
                print(f"❌ FAIL: {check_name} missing")
                
        # Check that templates were actually replaced
        if "{{ translate }}" in content:
            print(f"❌ FAIL: translate template not rendered")
        elif "Translation: " in content and len(content.split("Translation: ")[1].split("\n")[0]) > 0:
            print(f"✅ PASS: translate template was rendered")
            
        if "{{ validate }}" in content:
            print(f"❌ FAIL: validate template not rendered")
        elif "Validation: " in content and len(content.split("Validation: ")[1].split("\n")[0]) > 0:
            print(f"✅ PASS: validate template was rendered")
    else:
        print(f"❌ FAIL: {file_path} does not exist!")
    
    print("\n" + "="*60)
    print("END OF DEBUG TEST")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_translation_templates())