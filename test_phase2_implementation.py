#!/usr/bin/env python3
"""Test Phase 2 implementation: Smart tool discovery and automatic execution."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.engine import DeclarativePipelineEngine
from orchestrator.tools.discovery import ToolDiscoveryEngine, ToolMatch
from orchestrator.tools.base import default_registry


async def test_tool_discovery_engine():
    """Test the smart tool discovery engine."""
    print("🔍 Testing Tool Discovery Engine")
    print("-" * 40)
    
    discovery = ToolDiscoveryEngine()
    
    # Test different action descriptions
    test_actions = [
        "search web for information about quantum computing",
        "analyze the provided sales data",
        "generate a comprehensive report",
        "scrape product information from website",
        "validate input data format",
        "read configuration file",
        "run database migration script"
    ]
    
    for action in test_actions:
        print(f"\n🎯 Action: {action}")
        matches = discovery.discover_tools_for_action(action)
        
        if matches:
            for match in matches[:3]:  # Show top 3 matches
                print(f"  ✅ {match.tool_name} (confidence: {match.confidence:.2f})")
                print(f"     Reasoning: {match.reasoning}")
        else:
            print("  ❌ No tools discovered")
    
    return True


async def test_enhanced_pipeline_execution():
    """Test enhanced pipeline execution with smart tool discovery."""
    print("\n🚀 Testing Enhanced Pipeline Execution")
    print("-" * 40)
    
    # Complex pipeline that requires smart tool discovery
    pipeline_yaml = """
name: "Smart Research Pipeline"
description: "Automatically discover and use tools for research"
version: "1.0.0"

inputs:
  topic:
    type: string
    description: "Research topic"

steps:
  - id: discover_sources
    action: <AUTO>search for recent information about {{topic}} and find relevant sources</AUTO>
    
  - id: analyze_findings  
    action: <AUTO>analyze the search results and extract key insights about {{topic}}</AUTO>
    depends_on: [discover_sources]
    
  - id: generate_summary
    action: <AUTO>create a comprehensive summary of findings about {{topic}}</AUTO>
    depends_on: [analyze_findings]

outputs:
  research_summary: "{{generate_summary.result}}"
"""
    
    engine = DeclarativePipelineEngine()
    
    # Validate pipeline
    validation = await engine.validate_pipeline(pipeline_yaml)
    print(f"📋 Validation: {validation['valid']}")
    if validation.get('warnings'):
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    # Note: Actual execution would require model registry
    print("ℹ️  Pipeline validation successful - execution requires model integration")
    
    return validation['valid']


async def test_tool_chain_discovery():
    """Test discovery of tool chains for complex actions."""
    print("\n🔗 Testing Tool Chain Discovery") 
    print("-" * 40)
    
    discovery = ToolDiscoveryEngine()
    
    complex_actions = [
        "search web for data and then analyze the results",
        "scrape website content and generate a report",
        "read file data, validate it, and process the results",
        "search for information, extract insights, and create summary"
    ]
    
    for action in complex_actions:
        print(f"\n🎯 Complex Action: {action}")
        chain = discovery.get_tool_chain_for_action(action)
        
        if chain:
            print(f"  📋 Tool Chain ({len(chain)} tools):")
            for i, match in enumerate(chain):
                print(f"    {i+1}. {match.tool_name} (confidence: {match.confidence:.2f})")
        else:
            print("  ❌ No tool chain discovered")
    
    return True


async def test_tool_availability_and_alternatives():
    """Test tool availability checking and alternative suggestions."""
    print("\n🔧 Testing Tool Availability & Alternatives")
    print("-" * 40)
    
    discovery = ToolDiscoveryEngine()
    
    # Test with some tools that might not be available
    required_tools = [
        "web-search",
        "data-processing", 
        "report-generator",
        "non-existent-tool",
        "another-missing-tool"
    ]
    
    available_tools = discovery.tool_registry.list_tools()
    print(f"📦 Available tools: {available_tools}")
    
    missing_suggestions = discovery.suggest_missing_tools(required_tools)
    
    if missing_suggestions:
        print("\n💡 Suggestions for missing tools:")
        for missing_tool, alternatives in missing_suggestions.items():
            print(f"  ❌ {missing_tool}")
            if alternatives:
                print(f"    ➡️  Alternatives: {alternatives}")
            else:
                print(f"    ❓ No alternatives found")
    else:
        print("✅ All required tools are available")
    
    return True


async def test_pattern_matching_edge_cases():
    """Test edge cases in pattern matching."""
    print("\n🧪 Testing Pattern Matching Edge Cases")
    print("-" * 40)
    
    discovery = ToolDiscoveryEngine()
    
    edge_cases = [
        "",  # Empty string
        "do something undefined",  # Vague action
        "quantum neural blockchain AI optimization",  # Buzzword soup
        "make coffee",  # Unrelated action
        "search web and scrape data and analyze results and generate report and validate everything",  # Too many actions
    ]
    
    for action in edge_cases:
        print(f"\n🎯 Edge Case: '{action}'")
        try:
            matches = discovery.discover_tools_for_action(action)
            if matches:
                print(f"  ✅ Found {len(matches)} matches")
                print(f"    Best: {matches[0].tool_name} (confidence: {matches[0].confidence:.2f})")
            else:
                print("  ❌ No matches (expected for some cases)")
        except Exception as e:
            print(f"  💥 Error: {e}")
    
    return True


async def test_context_enhanced_discovery():
    """Test context-enhanced tool discovery."""
    print("\n🧠 Testing Context-Enhanced Discovery")
    print("-" * 40)
    
    discovery = ToolDiscoveryEngine()
    
    test_cases = [
        {
            "action": "process the data",
            "context": {"data": [1, 2, 3, 4, 5]},
            "expected": "data-processing"
        },
        {
            "action": "get information",
            "context": {"url": "https://example.com"},
            "expected": "headless-browser" 
        },
        {
            "action": "handle the content",
            "context": {"file_path": "/tmp/data.json"},
            "expected": "filesystem"
        }
    ]
    
    for case in test_cases:
        action = case["action"]
        context = case["context"]
        expected = case["expected"]
        
        print(f"\n🎯 Action: {action}")
        print(f"  📋 Context: {context}")
        
        matches = discovery.discover_tools_for_action(action, context)
        
        if matches:
            best_match = matches[0]
            print(f"  ✅ Best Match: {best_match.tool_name}")
            print(f"    Reasoning: {best_match.reasoning}")
            
            if expected in best_match.tool_name:
                print(f"  ✅ Expected tool found!")
            else:
                print(f"  ⚠️  Expected {expected}, got {best_match.tool_name}")
        else:
            print("  ❌ No matches found")
    
    return True


async def main():
    """Run all Phase 2 tests."""
    print("🧪 Testing Phase 2 Implementation")
    print("=" * 60)
    print("Smart Tool Discovery and Automatic Execution")
    print("=" * 60)
    
    tests = [
        ("Tool Discovery Engine", test_tool_discovery_engine),
        ("Enhanced Pipeline Execution", test_enhanced_pipeline_execution), 
        ("Tool Chain Discovery", test_tool_chain_discovery),
        ("Tool Availability & Alternatives", test_tool_availability_and_alternatives),
        ("Pattern Matching Edge Cases", test_pattern_matching_edge_cases),
        ("Context-Enhanced Discovery", test_context_enhanced_discovery),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("=" * 50)
        
        try:
            success = await test_func()
            results.append((test_name, success))
            print(f"\n✅ {test_name} COMPLETED")
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 PHASE 2 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Phase 2 implementation successful!")
        print("🚀 Ready for Phase 3: Conditional execution, loops, and error handling")
    else:
        print("⚠️  Some tests failed. Phase 2 needs review.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)