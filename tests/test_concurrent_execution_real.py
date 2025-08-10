"""
Real Concurrent Execution Tests - Issue #153 Phase 3

Tests concurrent pipeline execution with dependency fixes to ensure template
rendering works correctly under load and concurrent access patterns.
"""

import pytest
import os
import sys
import asyncio
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
from functools import wraps

from orchestrator import Orchestrator, init_models


def load_controlled_test(max_concurrent=5, timeout=300):
    """Decorator for load-controlled concurrent tests."""
    def decorator(test_func):
        @wraps(test_func)
        async def wrapper(*args, **kwargs):
            try:
                # Add semaphore for concurrent control
                semaphore = asyncio.Semaphore(max_concurrent)
                async def controlled_execution():
                    async with semaphore:
                        return await test_func(*args, **kwargs)
                
                result = await asyncio.wait_for(
                    controlled_execution(),
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                pytest.fail(f"Concurrent test timed out after {timeout} seconds")
        return wrapper
    return decorator


@pytest.fixture(scope="session")
def real_orchestrator():
    """Create orchestrator with real models for concurrent testing."""
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
    available_keys = [key for key in required_keys if os.getenv(key)]
    
    if not available_keys:
        pytest.skip(f"Skipping concurrent tests - no API keys found. Set one of: {required_keys}")
    
    print(f"Using real APIs for concurrent testing: {available_keys}")
    
    # Use same initialization as users
    registry = init_models()
    return Orchestrator(model_registry=registry)


class TestConcurrentPipelineExecution:
    """Test concurrent pipeline execution with real APIs."""
    
    @load_controlled_test(max_concurrent=3, timeout=360)
    async def test_multiple_pipelines_concurrent_template_rendering(self, real_orchestrator):
        """Test multiple pipelines executing concurrently with proper template rendering."""
        
        async def execute_research_pipeline(topic: str, output_id: str) -> Dict[str, Any]:
            """Execute a research pipeline with specific topic."""
            pipeline_yaml = f"""
            name: Concurrent Research Pipeline {output_id}
            parameters:
              research_topic: "{topic}"
              output_id: "{output_id}"
            steps:
              - id: research_{output_id}
                action: generate_text
                parameters:
                  prompt: "Research {{{{ research_topic }}}} in 2-3 sentences"
                  max_tokens: 100
                  
              - id: analyze_{output_id}
                action: analyze_text
                dependencies:
                  - research_{output_id}  # Proper dependency
                parameters:
                  text: "{{{{ research_{output_id}.result }}}}"
                  analysis_type: "summary"
                  prompt: "Summarize the key points about {{{{ research_topic }}}}"
                  
              - id: save_{output_id}
                tool: filesystem
                action: write
                dependencies:
                  - research_{output_id}  # Proper dependencies
                  - analyze_{output_id}
                parameters:
                  path: "/tmp/concurrent_{{{{ output_id }}}}.md"
                  content: |
                    # Concurrent Research: {{{{ research_topic }}}}
                    
                    **Generated**: {{{{ execution.timestamp }}}}
                    **Pipeline ID**: {{{{ output_id }}}}
                    
                    ## Research Results
                    {{{{ research_{output_id}.result }}}}
                    
                    ## Analysis  
                    {{{{ analyze_{output_id}.result }}}}
                    
                    ## Metadata
                    - Topic: {{{{ research_topic }}}}
                    - Pipeline: {output_id}
                    - Research Length: {{{{ research_{output_id}.result | length }}}} chars
                    - Analysis Length: {{{{ analyze_{output_id}.result | length }}}} chars
                    - Completed: {{{{ execution.timestamp }}}}
            """
            
            start_time = time.time()
            result = await real_orchestrator.execute_yaml(pipeline_yaml, {
                "research_topic": topic,
                "output_id": output_id
            })
            execution_time = time.time() - start_time
            
            return {
                'pipeline_id': output_id,
                'topic': topic,
                'result': result,
                'execution_time': execution_time,
                'output_file': f"/tmp/concurrent_{output_id}.md"
            }
        
        # Define concurrent test cases
        test_cases = [
            ("artificial intelligence", "ai"),
            ("renewable energy", "energy"),
            ("quantum computing", "quantum"),
            ("biotechnology", "biotech"),
            ("robotics automation", "robotics")
        ]
        
        # Execute all pipelines concurrently
        print(f"🔄 Starting {len(test_cases)} concurrent pipeline executions...")
        
        start_time = time.time()
        tasks = [execute_research_pipeline(topic, pipeline_id) for topic, pipeline_id in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        print(f"✅ Concurrent execution completed in {total_time:.2f}s")
        
        # Validate all executions succeeded
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(str(result))
            else:
                successful_results.append(result)
        
        assert len(failed_results) == 0, f"Failed executions: {failed_results}"
        assert len(successful_results) == len(test_cases), f"Expected {len(test_cases)} results, got {len(successful_results)}"
        
        # Validate template rendering in all outputs
        template_validation_results = []
        
        for result in successful_results:
            output_file = Path(result['output_file'])
            
            if not output_file.exists():
                template_validation_results.append(f"❌ {result['pipeline_id']}: Output file not created")
                continue
            
            content = output_file.read_text()
            
            # Check for unrendered templates
            if '{{' in content or '{%' in content:
                template_validation_results.append(f"❌ {result['pipeline_id']}: Unrendered templates found")
                continue
                
            # Validate topic inclusion
            if result['topic'].lower() not in content.lower():
                template_validation_results.append(f"❌ {result['pipeline_id']}: Topic not found in content")
                continue
                
            # Validate structured content
            required_sections = ["Research Results", "Analysis", "Metadata"]
            missing_sections = [section for section in required_sections if section not in content]
            if missing_sections:
                template_validation_results.append(f"❌ {result['pipeline_id']}: Missing sections: {missing_sections}")
                continue
                
            # Validate execution metadata
            if result['pipeline_id'] not in content:
                template_validation_results.append(f"❌ {result['pipeline_id']}: Pipeline ID not in content")
                continue
                
            template_validation_results.append(f"✅ {result['pipeline_id']}: All template validations passed")
        
        # Clean up output files
        for result in successful_results:
            output_file = Path(result['output_file'])
            if output_file.exists():
                output_file.unlink()
        
        # Report results
        print("\\n📊 Concurrent Template Rendering Results:")
        for validation in template_validation_results:
            print(f"   {validation}")
        
        # Calculate performance metrics
        avg_execution_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
        max_execution_time = max(r['execution_time'] for r in successful_results)
        
        print(f"\\n⚡ Performance Metrics:")
        print(f"   - Total concurrent time: {total_time:.2f}s")
        print(f"   - Average individual time: {avg_execution_time:.2f}s")
        print(f"   - Max individual time: {max_execution_time:.2f}s")
        print(f"   - Concurrency benefit: {(avg_execution_time * len(test_cases) / total_time):.2f}x")
        
        # Validate all passed
        failed_validations = [v for v in template_validation_results if '❌' in v]
        assert len(failed_validations) == 0, f"Template validation failures: {failed_validations}"

    @load_controlled_test(max_concurrent=4, timeout=240)
    async def test_template_context_isolation_concurrent(self, real_orchestrator):
        """Test that template contexts are properly isolated between concurrent executions."""
        
        async def execute_isolated_pipeline(unique_id: str, test_data: str) -> Dict[str, Any]:
            """Execute pipeline with unique test data."""
            pipeline_yaml = f"""
            name: Context Isolation Test {unique_id}
            parameters:
              unique_data: "{test_data}"
              pipeline_id: "{unique_id}"
            steps:
              - id: generate_unique_{unique_id}
                action: generate_text
                parameters:
                  prompt: "Generate content about {{{{ unique_data }}}} with identifier {{{{ pipeline_id }}}}"
                  max_tokens: 80
                  
              - id: save_unique_{unique_id}
                tool: filesystem
                action: write
                dependencies:
                  - generate_unique_{unique_id}
                parameters:
                  path: "/tmp/isolation_{{{{ pipeline_id }}}}.md"
                  content: |
                    # Context Isolation Test: {{{{ pipeline_id }}}}
                    
                    **Data**: {{{{ unique_data }}}}
                    **Generated Content**:
                    {{{{ generate_unique_{unique_id}.result }}}}
                    
                    **Validation**:
                    - Pipeline ID: {{{{ pipeline_id }}}}
                    - Data: {{{{ unique_data }}}}
                    - Timestamp: {{{{ execution.timestamp }}}}
            """
            
            result = await real_orchestrator.execute_yaml(pipeline_yaml, {
                "unique_data": test_data,
                "pipeline_id": unique_id
            })
            
            return {
                'unique_id': unique_id,
                'test_data': test_data,
                'result': result,
                'output_file': f"/tmp/isolation_{unique_id}.md"
            }
        
        # Create test cases with unique, easily identifiable data
        isolation_tests = [
            ("test_A", "context_data_ALPHA"),
            ("test_B", "context_data_BETA"), 
            ("test_C", "context_data_GAMMA"),
            ("test_D", "context_data_DELTA")
        ]
        
        # Execute concurrently
        print("🔄 Testing template context isolation...")
        
        tasks = [execute_isolated_pipeline(unique_id, test_data) for unique_id, test_data in isolation_tests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate isolation
        isolation_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                isolation_results.append(f"❌ {isolation_tests[i][0]}: Exception: {result}")
                continue
            
            unique_id = result['unique_id']
            expected_data = result['test_data'] 
            output_file = Path(result['output_file'])
            
            if not output_file.exists():
                isolation_results.append(f"❌ {unique_id}: Output file not created")
                continue
            
            content = output_file.read_text()
            
            # Validate this pipeline's data is present
            if expected_data not in content:
                isolation_results.append(f"❌ {unique_id}: Expected data '{expected_data}' not found")
                continue
                
            # Validate other pipelines' data is NOT present (context isolation)
            other_data = [test_data for test_unique_id, test_data in isolation_tests if test_unique_id != unique_id]
            contamination = [data for data in other_data if data in content]
            
            if contamination:
                isolation_results.append(f"❌ {unique_id}: Context contamination detected: {contamination}")
                continue
                
            # Validate proper template rendering
            if '{{' in content or '{%' in content:
                isolation_results.append(f"❌ {unique_id}: Unrendered templates found")
                continue
                
            isolation_results.append(f"✅ {unique_id}: Context properly isolated")
        
        # Clean up
        for result in results:
            if not isinstance(result, Exception):
                output_file = Path(result['output_file'])
                if output_file.exists():
                    output_file.unlink()
        
        # Report results
        print("\\n🔒 Context Isolation Results:")
        for isolation_result in isolation_results:
            print(f"   {isolation_result}")
        
        # Validate all passed
        failed_isolation = [r for r in isolation_results if '❌' in r]
        assert len(failed_isolation) == 0, f"Context isolation failures: {failed_isolation}"

    @load_controlled_test(max_concurrent=3, timeout=300)
    async def test_high_load_template_performance(self, real_orchestrator):
        """Test template rendering performance under high concurrent load."""
        
        async def execute_load_test_pipeline(batch_id: int, pipeline_num: int) -> Dict[str, Any]:
            """Execute a template-heavy pipeline for load testing."""
            # Use string formatting to avoid f-string conflicts with Jinja templates
            pipeline_yaml = """
            name: Load Test Pipeline B{batch_id}P{pipeline_num}
            parameters:
              batch_id: {batch_id}
              pipeline_num: {pipeline_num}
              test_topic: "load testing batch {batch_id} pipeline {pipeline_num}"
            steps:
              - id: generate_load_content_b{batch_id}p{pipeline_num}
                action: generate_text
                parameters:
                  prompt: "Generate test content for {{{{ test_topic }}}}"
                  max_tokens: 150
                  
              - id: process_templates_b{batch_id}p{pipeline_num}
                tool: filesystem
                action: write
                dependencies:
                  - generate_load_content_b{batch_id}p{pipeline_num}
                parameters:
                  path: "/tmp/load_test_b{{{{ batch_id }}}}_p{{{{ pipeline_num }}}}.md"
                  content: |
                    # Load Test Results
                    
                    **Batch**: {{{{ batch_id }}}}
                    **Pipeline**: {{{{ pipeline_num }}}}
                    **Topic**: {{{{ test_topic }}}}
                    **Timestamp**: {{{{ execution.timestamp }}}}
                    
                    ## Generated Content
                    {{{{ generate_load_content_b{batch_id}p{pipeline_num}.result }}}}
                    
                    ## Template Processing Tests
                    {{% set content_length = generate_load_content_b{batch_id}p{pipeline_num}.result | length %}}
                    - Content Length: {{{{ content_length }}}} characters
                    - Batch ID: {{{{ batch_id }}}}
                    - Pipeline Number: {{{{ pipeline_num }}}}
                    - Has Content: {{{{ "Yes" if content_length > 10 else "No" }}}}
                    
                    {{% for i in range(5) %}}
                    {{{{ loop.index }}}}. Load test item for batch {{{{ batch_id }}}} pipeline {{{{ pipeline_num }}}}
                    {{% endfor %}}
                    
                    ## Performance Metadata
                    - Execution Time: {{{{ execution.timestamp }}}}
                    - Pipeline Identifier: B{{{{ batch_id }}}}P{{{{ pipeline_num }}}}
                    - Template Complexity: High (loops, conditionals, filters)
            """.format(batch_id=batch_id, pipeline_num=pipeline_num)
            
            start_time = time.time()
            
            result = await real_orchestrator.execute_yaml(pipeline_yaml, {
                "batch_id": batch_id,
                "pipeline_num": pipeline_num,
                "test_topic": f"load testing batch {batch_id} pipeline {pipeline_num}"
            })
            
            execution_time = time.time() - start_time
            
            return {
                'batch_id': batch_id,
                'pipeline_num': pipeline_num,
                'execution_time': execution_time,
                'result': result,
                'output_file': f"/tmp/load_test_b{batch_id}_p{pipeline_num}.md"
            }
        
        # Create high-load scenario: 3 batches of 4 pipelines each = 12 concurrent executions
        print("🔄 Starting high-load concurrent template test...")
        
        tasks = []
        for batch_id in range(1, 4):  # 3 batches
            for pipeline_num in range(1, 5):  # 4 pipelines per batch
                tasks.append(execute_load_test_pipeline(batch_id, pipeline_num))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_load_time = time.time() - start_time
        
        print(f"✅ High-load test completed in {total_load_time:.2f}s")
        
        # Analyze results
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append(str(result))
            else:
                successful_results.append(result)
        
        print(f"📊 Load Test Results:")
        print(f"   - Total executions: {len(tasks)}")
        print(f"   - Successful: {len(successful_results)}")
        print(f"   - Failed: {len(failed_results)}")
        
        # Validate performance
        if successful_results:
            execution_times = [r['execution_time'] for r in successful_results]
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            print(f"   - Average execution time: {avg_time:.2f}s")
            print(f"   - Max execution time: {max_time:.2f}s")
            print(f"   - Min execution time: {min_time:.2f}s")
            
            # Performance assertions
            assert avg_time < 60.0, f"Average execution time too high: {avg_time:.2f}s"
            assert max_time < 120.0, f"Max execution time too high: {max_time:.2f}s"
        
        # Validate template rendering quality
        template_quality_results = []
        
        for result in successful_results:
            output_file = Path(result['output_file'])
            
            if not output_file.exists():
                template_quality_results.append(f"❌ B{result['batch_id']}P{result['pipeline_num']}: No output file")
                continue
            
            content = output_file.read_text()
            
            # Check for unrendered templates
            if '{{' in content or '{%' in content:
                template_quality_results.append(f"❌ B{result['batch_id']}P{result['pipeline_num']}: Unrendered templates")
                continue
            
            # Check for expected content
            expected_identifier = f"B{result['batch_id']}P{result['pipeline_num']}"
            if expected_identifier not in content:
                template_quality_results.append(f"❌ {expected_identifier}: Missing identifier")
                continue
                
            # Check for loop processing (should have 5 numbered items)
            numbered_items = [line for line in content.split('\\n') if line.strip().startswith(('1.', '2.', '3.', '4.', '5.'))]
            if len(numbered_items) < 5:
                template_quality_results.append(f"❌ {expected_identifier}: Loop processing incomplete")
                continue
            
            template_quality_results.append(f"✅ {expected_identifier}: All validations passed")
        
        # Clean up output files
        for result in successful_results:
            output_file = Path(result['output_file'])
            if output_file.exists():
                output_file.unlink()
        
        # Report template quality
        print("\\n🎨 Template Quality Results:")
        for quality_result in template_quality_results:
            print(f"   {quality_result}")
        
        # Final validations
        assert len(failed_results) <= 1, f"Too many failed executions: {len(failed_results)}"  # Allow 1 failure
        
        quality_failures = [r for r in template_quality_results if '❌' in r]
        assert len(quality_failures) == 0, f"Template quality failures: {quality_failures}"
        
        print(f"✅ High-load template rendering test completed successfully")
        print(f"   - Success rate: {len(successful_results)}/{len(tasks)} ({len(successful_results)/len(tasks)*100:.1f}%)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])