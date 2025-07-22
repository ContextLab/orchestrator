#!/usr/bin/env python3
"""Simple test of tool catalog examples with minimal complexity."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator, init_models


async def test_filesystem_operations():
    """Test basic filesystem operations."""
    print("\n=== Testing FileSystem Operations ===")
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Simple write and read test
    test_file = "/tmp/test_doc_example.txt"
    test_content = "Hello from tool catalog test!"
    
    pipeline_yaml = f"""
name: filesystem-test
description: Test filesystem operations

steps:
  - id: write_file
    tool: filesystem
    action: write
    parameters:
      path: "{test_file}"
      content: "{test_content}"
      mode: "w"
      
  - id: read_file
    tool: filesystem
    action: read
    parameters:
      path: "{test_file}"
      
  - id: check_exists
    tool: filesystem
    action: exists
    parameters:
      path: "{test_file}"
"""
    
    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)
        
        # Verify
        assert result['check_exists']['exists'] is True
        assert test_content in result['read_file']['content']
        
        print("✅ FileSystem operations test passed")
        return True
        
    except Exception as e:
        print(f"❌ FileSystem test failed: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


async def test_terminal_command():
    """Test terminal command execution."""
    print("\n=== Testing Terminal Command ===")
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    pipeline_yaml = """
name: terminal-test
description: Test terminal command

steps:
  - id: run_echo
    tool: terminal
    action: execute
    parameters:
      command: "echo 'Tool catalog test successful'"
      timeout: 10
      
  - id: check_python
    tool: terminal
    action: execute
    parameters:
      command: "python --version"
      timeout: 10
"""
    
    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)
        
        # Verify
        assert 'Tool catalog test successful' in result['run_echo']['stdout']
        assert 'Python' in result['check_python']['stdout']
        
        print("✅ Terminal command test passed")
        return True
        
    except Exception as e:
        print(f"❌ Terminal test failed: {e}")
        return False


async def test_report_generation():
    """Test report generation."""
    print("\n=== Testing Report Generation ===")
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    pipeline_yaml = """
name: report-test
description: Test report generation

steps:
  - id: generate_report
    tool: report-generator
    action: generate
    parameters:
      title: "Tool Catalog Test Report"
      format: "markdown"
      template: |
        # Tool Catalog Test Report
        
        ## Summary
        This report validates the tool catalog documentation.
        
        ## Test Results
        - FileSystem tool: Working
        - Terminal tool: Working
        - Report tool: Working
        
        Generated at: {{ timestamp }}
      metadata:
        author: "Test Suite"
        timestamp: "2024-01-15"
"""
    
    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)
        
        # Verify
        content = result['generate_report']['content']
        assert 'Tool Catalog Test Report' in content
        assert 'FileSystem tool: Working' in content
        
        print("✅ Report generation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Report test failed: {e}")
        return False


async def test_llm_generation():
    """Test LLM text generation."""
    print("\n=== Testing LLM Generation ===")
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    pipeline_yaml = """
name: llm-test
description: Test LLM generation

steps:
  - id: generate_text
    tool: llm-generate
    action: generate
    parameters:
      prompt: "Write a one-sentence summary of what orchestrator tools do."
      temperature: 0.3
      max_tokens: 50
"""
    
    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)
        
        # Verify
        generated = result['generate_text']['result']
        assert len(generated) > 0
        assert len(generated) < 200  # Should be concise
        
        print("✅ LLM generation test passed")
        print(f"   Generated: {generated}")
        return True
        
    except Exception as e:
        print(f"❌ LLM generation test failed: {e}")
        return False


async def test_validation():
    """Test data validation."""
    print("\n=== Testing Data Validation ===")
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Create test data file
    test_data = {"email": "test@example.com", "age": 25}
    data_file = "/tmp/test_validation_data.json"
    with open(data_file, "w") as f:
        json.dump(test_data, f)
    
    pipeline_yaml = f"""
name: validation-test
description: Test data validation

steps:
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "{data_file}"
      
  - id: validate_data
    tool: validation
    action: validate
    parameters:
      data: "{{{{ read_data.content | json }}}}"
      schema:
        type: object
        properties:
          email:
            type: string
            format: email
          age:
            type: number
            minimum: 18
        required: ["email", "age"]
      mode: "STRICT"
"""
    
    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)
        
        # Verify
        validation = result['validate_data']
        assert validation['is_valid'] is True
        
        print("✅ Data validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False
    finally:
        if os.path.exists(data_file):
            os.unlink(data_file)


async def test_web_search():
    """Test web search functionality."""
    print("\n=== Testing Web Search ===")
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    pipeline_yaml = """
name: search-test
description: Test web search

steps:
  - id: search_web
    tool: web-search
    action: search
    parameters:
      query: "Python programming language"
      max_results: 3
"""
    
    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)
        
        # Verify
        search_results = result['search_web']['results']
        assert len(search_results) > 0
        assert all('title' in r and 'url' in r for r in search_results)
        
        print(f"✅ Web search test passed - found {len(search_results)} results")
        return True
        
    except Exception as e:
        print(f"❌ Web search test failed: {e}")
        return False


async def main():
    """Run simple tool catalog tests."""
    print("🧪 TOOL CATALOG SIMPLE TESTS")
    print("=" * 50)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  WARNING: No API keys found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    # Run tests
    tests = [
        ("FileSystem Operations", test_filesystem_operations),
        ("Terminal Commands", test_terminal_command),
        ("Report Generation", test_report_generation),
        ("LLM Generation", test_llm_generation),
        ("Data Validation", test_validation),
        ("Web Search", test_web_search),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print("="*50)
        
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    asyncio.run(main())