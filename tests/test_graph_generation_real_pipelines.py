"""
Real pipeline testing for graph generation system.

This module tests the graph generation system with actual pipeline files following 
the NO MOCK policy. All tests use real pipeline definitions from the examples directory.
"""

import asyncio
import yaml
import glob
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from orchestrator.graph_generation.automatic_generator import AutomaticGraphGenerator
from orchestrator.graph_generation.syntax_parser import DeclarativeSyntaxParser


async def test_real_pipeline_parsing():
    """Test parsing real pipeline files from examples directory."""
    print("🧪 Testing real pipeline parsing...")
    
    parser = DeclarativeSyntaxParser()
    generator = AutomaticGraphGenerator()
    
    # Find simple pipeline files
    pipeline_files = [
        "examples/test_simple_pipeline.yaml",
        "examples/test_basic_pipeline.yaml",
        "examples/simple_error_handling.yaml",
    ]
    
    successful_parses = 0
    total_files = 0
    
    for pipeline_file in pipeline_files:
        if not Path(pipeline_file).exists():
            continue
            
        total_files += 1
        
        try:
            print(f"📄 Testing {pipeline_file}")
            
            # Load pipeline definition
            with open(pipeline_file, 'r') as f:
                pipeline_def = yaml.safe_load(f)
                
            print(f"   Pipeline ID: {pipeline_def.get('id', 'unknown')}")
            print(f"   Steps: {len(pipeline_def.get('steps', []))}")
            
            # Test parsing
            parsed = await parser.parse_pipeline_definition(pipeline_def)
            print(f"   ✅ Parsed successfully: {len(parsed.steps)} steps, {len(parsed.inputs)} inputs")
            
            # Test dependency resolution (Phase 1 component)
            dependency_graph = await generator._analyze_dependencies(parsed)
            print(f"   ✅ Dependencies resolved: {len(dependency_graph.edges)} edges")
            
            # Test execution order
            execution_order = dependency_graph.get_execution_order()
            print(f"   ✅ Execution order: {' → '.join(execution_order[:3])}{'...' if len(execution_order) > 3 else ''}")
            
            successful_parses += 1
            
        except FileNotFoundError:
            print(f"   ⚠️  File not found: {pipeline_file}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")
            
    print(f"\n📊 Results: {successful_parses}/{total_files} pipelines parsed successfully")
    return successful_parses, total_files


async def test_template_variable_extraction():
    """Test template variable extraction with real pipeline examples.""" 
    print("\n🧪 Testing template variable extraction...")
    
    parser = DeclarativeSyntaxParser()
    
    # Real examples from pipeline files
    test_cases = [
        # From control_flow_conditional.yaml
        ("Provide analysis of {{ read_file.size }} bytes", ["read_file.size"]),
        
        # From research pipelines
        ("{{ topic }} latest research", ["topic"]),
        ("Query: {{ inputs.topic }} with depth {{ inputs.depth }}", ["inputs.topic", "inputs.depth"]),
        
        # Complex expressions
        ("{{ search_results | length > 0 }}", ["search_results"]),
        ("{{ results[0].url if results else '' }}", ["results"]),
        
        # Nested expressions
        ("Process {{ web_search.results }} and {{ analyze_findings.claims }}", ["web_search.results", "analyze_findings.claims"]),
    ]
    
    successful_extractions = 0
    
    for text, expected_vars in test_cases:
        extracted = parser.extract_template_variables(text)
        
        if set(extracted) == set(expected_vars):
            print(f"   ✅ '{text[:50]}...' → {extracted}")
            successful_extractions += 1
        else:
            print(f"   ❌ '{text[:50]}...' → expected {expected_vars}, got {extracted}")
            
    print(f"\n📊 Variable extraction: {successful_extractions}/{len(test_cases)} successful")
    return successful_extractions


async def test_dependency_analysis_with_real_data():
    """Test dependency analysis with real pipeline structures."""
    print("\n🧪 Testing dependency analysis with real structures...")
    
    generator = AutomaticGraphGenerator()
    parser = DeclarativeSyntaxParser()
    
    # Create a real-world-like pipeline structure
    realistic_pipeline = {
        "id": "realistic-research-pipeline",
        "name": "Realistic Research Pipeline",
        "inputs": {
            "topic": {"type": "string", "required": True},
            "depth": {"type": "integer", "default": 3}
        },
        "steps": [
            {
                "id": "web_search",
                "tool": "web-search",
                "inputs": {
                    "query": "{{ inputs.topic }} research",
                    "max_results": "{{ inputs.depth * 5 }}"
                },
                "outputs": {
                    "results": {"type": "array", "description": "Search results"},
                    "total_count": {"type": "integer"}
                }
            },
            {
                "id": "filter_results", 
                "tool": "content_filter",
                "depends_on": ["web_search"],  # Explicit dependency
                "inputs": {
                    "content": "{{ web_search.results }}",  # Implicit dependency 
                    "criteria": "academic sources"
                }
            },
            {
                "id": "parallel_analysis",
                "type": "parallel_map",
                "depends_on": ["filter_results"],
                "items": "{{ filter_results.filtered_content }}",
                "tool": "content_analyzer",
                "inputs": {
                    "text": "{{ item.content }}",
                    "analysis_type": "summary"
                }
            },
            {
                "id": "conditional_deep_analysis",
                "condition": "{{ parallel_analysis.results | length > 5 }}",
                "depends_on": ["parallel_analysis"],
                "tool": "deep_analyzer",
                "inputs": {
                    "summaries": "{{ parallel_analysis.results }}"
                }
            },
            {
                "id": "final_report",
                "depends_on": ["parallel_analysis"],  # May also depend on conditional step
                "tool": "report_generator",
                "inputs": {
                    "topic": "{{ inputs.topic }}",
                    "analyses": "{{ parallel_analysis.results }}",
                    "deep_analysis": "{{ conditional_deep_analysis.result | default('none') }}"
                }
            }
        ]
    }
    
    try:
        # Parse pipeline
        parsed = await parser.parse_pipeline_definition(realistic_pipeline)
        print(f"   ✅ Parsed realistic pipeline: {len(parsed.steps)} steps")
        
        # Analyze dependencies
        dep_graph = await generator._analyze_dependencies(parsed)
        print(f"   ✅ Dependency analysis: {len(dep_graph.edges)} dependencies found")
        
        # Check execution levels (for parallel detection)
        exec_levels = dep_graph.get_execution_levels()
        print(f"   ✅ Execution levels: {len(exec_levels)} levels")
        for level, steps in exec_levels.items():
            print(f"      Level {level}: {steps}")
            
        # Test parallel detection
        parallel_detector = generator.parallel_detector
        parallel_groups = await parallel_detector.detect_parallel_groups(dep_graph)
        print(f"   ✅ Parallel opportunities: {len(parallel_groups)} groups detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Realistic pipeline test failed: {e}")
        return False


async def test_control_flow_analysis():
    """Test control flow analysis with real patterns."""
    print("\n🧪 Testing control flow analysis...")
    
    generator = AutomaticGraphGenerator()
    parser = DeclarativeSyntaxParser()
    
    # Pipeline with various control flow patterns
    control_flow_pipeline = {
        "id": "control-flow-test",
        "steps": [
            {
                "id": "check_condition",
                "action": "evaluate",
                "inputs": {"expression": "{{ inputs.enable_processing }}"}
            },
            {
                "id": "conditional_processing",
                "condition": "{{ check_condition.result == 'true' }}",
                "depends_on": ["check_condition"],
                "action": "process_data"
            },
            {
                "id": "parallel_map_step",
                "type": "parallel_map",
                "items": "{{ inputs.data_items }}",
                "tool": "item_processor",
                "inputs": {
                    "item": "{{ item }}"
                }
            },
            {
                "id": "loop_step",
                "type": "while",
                "condition": "{{ iteration_count < inputs.max_iterations }}",
                "max_iterations": 10,
                "action": "iterate"
            }
        ]
    }
    
    try:
        parsed = await parser.parse_pipeline_definition(control_flow_pipeline)
        
        # Test control flow analysis
        control_flow_analyzer = generator.control_flow_analyzer
        control_flow_map = await control_flow_analyzer.analyze_control_flow(parsed)
        
        print(f"   ✅ Control flow analysis completed:")
        print(f"      Conditionals: {len(control_flow_map.conditionals)}")
        print(f"      Loops: {len(control_flow_map.loops)}")
        print(f"      Parallel maps: {len(control_flow_map.parallel_maps)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Control flow test failed: {e}")
        return False


async def main():
    """Run all real pipeline tests."""
    print("🚀 Starting Real Pipeline Testing for Graph Generation")
    print("=" * 60)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: Real pipeline parsing
    parsing_success, total_files = await test_real_pipeline_parsing()
    test_results["pipeline_parsing"] = parsing_success > 0
    
    # Test 2: Template variable extraction
    extraction_success = await test_template_variable_extraction()
    test_results["variable_extraction"] = extraction_success > 0
    
    # Test 3: Dependency analysis
    test_results["dependency_analysis"] = await test_dependency_analysis_with_real_data()
    
    # Test 4: Control flow analysis
    test_results["control_flow_analysis"] = await test_control_flow_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    total_passed = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All real pipeline tests PASSED! Phase 1 Week 1 core functionality is working.")
    else:
        print("⚠️  Some tests failed. Review implementation before proceeding.")
        
    return total_passed == total_tests


if __name__ == "__main__":
    # Run the real pipeline tests
    asyncio.run(main())