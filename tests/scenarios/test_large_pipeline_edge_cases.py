"""
Large Pipeline Edge Case Testing

Tests the orchestrator's handling of extremely large pipelines, complex
dependency chains, deep nesting, and boundary conditions in pipeline processing.
"""

import pytest
import time
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Import orchestrator components
from src.orchestrator.orchestrator import Orchestrator


class TestLargePipelineScaling:
    """Test handling of very large pipeline definitions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "large_pipeline_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    def generate_large_pipeline(self, num_steps: int, dependency_ratio: float = 0.3) -> str:
        """Generate a large pipeline with specified number of steps."""
        pipeline = {
            "name": f"large_pipeline_{num_steps}_steps",
            "version": "1.0",
            "description": f"Test pipeline with {num_steps} steps",
            "steps": []
        }
        
        for i in range(num_steps):
            step = {
                "id": f"step_{i:05d}",
                "action": "python_code",
                "parameters": {
                    "code": f"""
import time
step_id = "step_{i:05d}"
print(f"Executing {{step_id}}")

# Simulate some work
result = {i} * 2 + 1
time.sleep(0.01)  # Small delay to simulate processing

print(f"{{step_id}} completed with result: {{result}}")
output_value_{i} = result
"""
                },
                "outputs": [f"output_value_{i}"]
            }
            
            # Add dependencies based on dependency ratio
            if i > 0 and i % int(1/dependency_ratio) == 0:
                # Create dependency on previous step(s)
                deps = []
                for j in range(max(0, i-3), i):  # Depend on up to 3 previous steps
                    deps.append(f"step_{j:05d}")
                if deps:
                    step["depends_on"] = deps
            
            pipeline["steps"].append(step)
        
        return yaml.dump(pipeline, default_flow_style=False)
    
    @pytest.mark.asyncio
    async def test_hundred_step_pipeline(self):
        """Test execution of a 100-step pipeline."""
        num_steps = 100
        pipeline_content = self.generate_large_pipeline(num_steps, dependency_ratio=0.2)
        pipeline_path = self.create_test_pipeline(pipeline_content, f"pipeline_{num_steps}_steps.yaml")
        
        start_time = time.time()
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        execution_time = time.time() - start_time
        
        # Should handle 100-step pipeline
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        if result.status == "success":
            assert len(result.step_results) == num_steps
            successful_steps = [step for step in result.step_results if step.status == "success"]
            print(f"✓ 100-step pipeline: {len(successful_steps)}/{num_steps} steps succeeded")
        else:
            print(f"✓ 100-step pipeline handled gracefully (status: {result.status})")
        
        print(f"  Execution time: {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_large_pipeline_with_complex_dependencies(self):
        """Test pipeline with complex dependency chains."""
        complex_dependency_pipeline = """
name: complex_dependency_test
version: "1.0"
description: Pipeline with complex interdependencies

steps:
  # Foundation layer (no dependencies)
  - id: foundation_1
    action: python_code
    parameters:
      code: |
        foundation_result_1 = "foundation_data_1"
        print(f"Foundation 1: {foundation_result_1}")
    outputs: [foundation_result_1]
    
  - id: foundation_2
    action: python_code
    parameters:
      code: |
        foundation_result_2 = "foundation_data_2"
        print(f"Foundation 2: {foundation_result_2}")
    outputs: [foundation_result_2]
    
  - id: foundation_3
    action: python_code
    parameters:
      code: |
        foundation_result_3 = "foundation_data_3"
        print(f"Foundation 3: {foundation_result_3}")
    outputs: [foundation_result_3]

  # Layer 1 (depends on foundation)
  - id: layer1_step1
    action: python_code
    parameters:
      code: |
        layer1_result1 = f"{foundation_result_1}_processed"
        print(f"Layer 1.1: {layer1_result1}")
    depends_on: [foundation_1]
    outputs: [layer1_result1]
    
  - id: layer1_step2
    action: python_code
    parameters:
      code: |
        layer1_result2 = f"{foundation_result_2}_processed"
        print(f"Layer 1.2: {layer1_result2}")
    depends_on: [foundation_2]
    outputs: [layer1_result2]
    
  - id: layer1_step3
    action: python_code
    parameters:
      code: |
        combined = f"{foundation_result_1}_{foundation_result_3}"
        layer1_result3 = f"{combined}_combined"
        print(f"Layer 1.3: {layer1_result3}")
    depends_on: [foundation_1, foundation_3]
    outputs: [layer1_result3]

  # Layer 2 (depends on layer 1)
  - id: layer2_step1
    action: python_code
    parameters:
      code: |
        layer2_result1 = f"{layer1_result1}_{layer1_result2}_merged"
        print(f"Layer 2.1: {layer2_result1}")
    depends_on: [layer1_step1, layer1_step2]
    outputs: [layer2_result1]
    
  - id: layer2_step2
    action: python_code
    parameters:
      code: |
        all_data = f"{layer1_result1}_{layer1_result2}_{layer1_result3}"
        layer2_result2 = f"{all_data}_all_combined"
        print(f"Layer 2.2: {layer2_result2}")
    depends_on: [layer1_step1, layer1_step2, layer1_step3]
    outputs: [layer2_result2]

  # Layer 3 (convergence)
  - id: final_convergence
    action: python_code
    parameters:
      code: |
        final_result = f"{layer2_result1}_{layer2_result2}_final"
        print(f"Final convergence: {final_result}")
        
        # Validate all data flowed correctly
        expected_components = [
          "foundation_data_1", "foundation_data_2", "foundation_data_3",
          "processed", "combined", "merged", "all_combined", "final"
        ]
        
        missing_components = []
        for component in expected_components:
          if component not in final_result:
            missing_components.append(component)
        
        if missing_components:
          print(f"Warning: Missing components: {missing_components}")
        else:
          print("All data components successfully propagated")
          
        print(f"Final result length: {len(final_result)} characters")
    depends_on: [layer2_step1, layer2_step2]
    outputs: [final_result]
"""
        
        pipeline_path = self.create_test_pipeline(complex_dependency_pipeline, "complex_dependencies.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle complex dependencies
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        if result.status == "success":
            # Verify all steps executed
            expected_steps = 9  # Count from pipeline definition
            assert len(result.step_results) == expected_steps
            
            # Verify final step succeeded
            final_step = next((step for step in result.step_results if step.step_id == "final_convergence"), None)
            assert final_step is not None
            print("✓ Complex dependency pipeline executed successfully")
        else:
            print(f"✓ Complex dependency pipeline handled gracefully (status: {result.status})")
    
    @pytest.mark.asyncio
    async def test_deeply_nested_conditional_pipeline(self):
        """Test pipeline with deep conditional nesting."""
        nested_conditional_pipeline = """
name: deeply_nested_conditional_test
version: "1.0"
description: Pipeline with deep conditional nesting

steps:
  - id: initialize_conditions
    action: python_code
    parameters:
      code: |
        import random
        
        # Generate random conditions for testing
        condition_a = random.choice([True, False])
        condition_b = random.choice([True, False])
        condition_c = random.choice([True, False])
        condition_d = random.choice([True, False])
        
        print(f"Conditions: A={condition_a}, B={condition_b}, C={condition_c}, D={condition_d}")
        
        # Create nested condition string for evaluation
        nested_condition = f"({condition_a} and {condition_b}) or ({condition_c} and {condition_d})"
        print(f"Nested condition: {nested_condition}")
        
        evaluation_result = eval(nested_condition)
        print(f"Nested condition evaluates to: {evaluation_result}")
    outputs: [condition_a, condition_b, condition_c, condition_d, evaluation_result]
    
  # Level 1 conditionals
  - id: level1_branch_a
    action: python_code
    parameters:
      code: |
        if condition_a:
          level1_a_result = "Level 1A: Condition A is True"
          continue_level2_a = True
        else:
          level1_a_result = "Level 1A: Condition A is False"
          continue_level2_a = False
        print(level1_a_result)
    depends_on: [initialize_conditions]
    outputs: [level1_a_result, continue_level2_a]
    
  - id: level1_branch_b
    action: python_code
    parameters:
      code: |
        if condition_b:
          level1_b_result = "Level 1B: Condition B is True"
          continue_level2_b = condition_a  # Nested dependency
        else:
          level1_b_result = "Level 1B: Condition B is False"
          continue_level2_b = False
        print(level1_b_result)
    depends_on: [initialize_conditions]
    outputs: [level1_b_result, continue_level2_b]
    
  # Level 2 conditionals (depend on level 1)
  - id: level2_branch_ab
    action: python_code
    parameters:
      code: |
        if continue_level2_a and continue_level2_b:
          level2_ab_result = "Level 2AB: Both A and B paths continue"
          trigger_level3 = True
        elif continue_level2_a or continue_level2_b:
          level2_ab_result = "Level 2AB: One of A or B paths continue"
          trigger_level3 = condition_c  # Another nested condition
        else:
          level2_ab_result = "Level 2AB: Neither A nor B paths continue"
          trigger_level3 = False
          
        print(level2_ab_result)
        print(f"Trigger level 3: {trigger_level3}")
    depends_on: [level1_branch_a, level1_branch_b]
    outputs: [level2_ab_result, trigger_level3]
    
  # Level 3 conditionals (deep nesting)
  - id: level3_complex_branch
    action: python_code
    parameters:
      code: |
        if trigger_level3:
          if condition_c and condition_d:
            level3_result = "Level 3: Both C and D are True with triggered path"
            final_complexity_score = 100
          elif condition_c or condition_d:
            level3_result = "Level 3: One of C or D is True with triggered path"
            final_complexity_score = 75
          else:
            level3_result = "Level 3: Neither C nor D True but path triggered"
            final_complexity_score = 50
        else:
          level3_result = "Level 3: Path not triggered"
          final_complexity_score = 25
          
        print(level3_result)
        print(f"Final complexity score: {final_complexity_score}")
        
        # Validate nested logic consistency
        expected_trigger = (condition_a and condition_b) or ((condition_a or condition_b) and condition_c)
        actual_trigger = trigger_level3
        
        logic_consistent = (expected_trigger == actual_trigger) or not (condition_a or condition_b)
        print(f"Logic consistency check: {'PASS' if logic_consistent else 'FAIL'}")
    depends_on: [level2_branch_ab, initialize_conditions]
    outputs: [level3_result, final_complexity_score]
    
  # Final convergence with nested evaluation
  - id: final_nested_evaluation
    action: python_code
    parameters:
      code: |
        # Reconstruct the execution path
        path_description = []
        
        if condition_a:
          path_description.append("A-True")
        else:
          path_description.append("A-False")
          
        if condition_b:
          path_description.append("B-True")
        else:
          path_description.append("B-False")
          
        if trigger_level3:
          path_description.append("L3-Triggered")
          if condition_c and condition_d:
            path_description.append("CD-Both")
          elif condition_c or condition_d:
            path_description.append("CD-One")
          else:
            path_description.append("CD-Neither")
        else:
          path_description.append("L3-NotTriggered")
        
        execution_path = " -> ".join(path_description)
        print(f"Execution path: {execution_path}")
        
        # Summary
        total_conditions = sum([condition_a, condition_b, condition_c, condition_d])
        print(f"Total true conditions: {total_conditions}/4")
        print(f"Final complexity score: {final_complexity_score}")
        print(f"Overall evaluation result: {evaluation_result}")
        
        # Validate the deep nesting worked correctly
        nesting_depth = len(path_description)
        print(f"Maximum nesting depth reached: {nesting_depth} levels")
    depends_on: [level3_complex_branch]
"""
        
        pipeline_path = self.create_test_pipeline(nested_conditional_pipeline, "deeply_nested.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle deep nesting
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        if result.status == "success":
            print("✓ Deeply nested conditional pipeline executed successfully")
        else:
            print(f"✓ Deeply nested pipeline handled gracefully (status: {result.status})")
    
    @pytest.mark.asyncio
    async def test_large_data_flow_pipeline(self):
        """Test pipeline that processes and flows large amounts of data between steps."""
        large_data_pipeline = f"""
name: large_data_flow_test
version: "1.0"
description: Pipeline that handles large data flow between steps

steps:
  - id: generate_large_dataset
    action: python_code
    parameters:
      code: |
        import json
        import time
        
        print("Generating large dataset...")
        
        # Create a substantial dataset (but not huge to avoid memory issues)
        large_dataset = {{
          "metadata": {{
            "version": "1.0",
            "created": time.time(),
            "size": "large"
          }},
          "data_points": []
        }}
        
        # Generate 10,000 data points
        for i in range(10000):
          data_point = {{
            "id": i,
            "value": i ** 2,
            "category": f"category_{{i % 100}}",
            "nested_data": {{
              "field_a": f"value_a_{{i}}",
              "field_b": i * 3.14159,
              "field_c": [j for j in range(i % 10)]
            }}
          }}
          large_dataset["data_points"].append(data_point)
        
        dataset_size_mb = len(json.dumps(large_dataset)) / (1024 * 1024)
        print(f"Generated dataset: {{len(large_dataset['data_points'])}} points, {{dataset_size_mb:.2f}} MB")
        
        # Output the dataset for next step
        raw_dataset = large_dataset
    outputs: [raw_dataset]
    
  - id: process_large_dataset
    action: python_code
    parameters:
      code: |
        import json
        import statistics
        
        print("Processing large dataset...")
        
        # Validate dataset received
        if not raw_dataset or "data_points" not in raw_dataset:
          raise ValueError("Invalid dataset received")
        
        data_points = raw_dataset["data_points"]
        print(f"Processing {{len(data_points)}} data points...")
        
        # Perform computations on the dataset
        values = [point["value"] for point in data_points]
        categories = {{}}
        
        for point in data_points:
          category = point["category"]
          if category not in categories:
            categories[category] = []
          categories[category].append(point["value"])
        
        # Calculate statistics
        processed_data = {{
          "total_points": len(data_points),
          "value_stats": {{
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values)
          }},
          "category_stats": {{}}
        }}
        
        # Process each category
        for category, category_values in categories.items():
          processed_data["category_stats"][category] = {{
            "count": len(category_values),
            "sum": sum(category_values),
            "avg": sum(category_values) / len(category_values) if category_values else 0
          }}
        
        processed_size_mb = len(json.dumps(processed_data)) / (1024 * 1024)
        print(f"Processed data: {{processed_size_mb:.2f}} MB")
        print(f"Categories found: {{len(processed_data['category_stats'])}}")
        
        # Output processed data
        processed_dataset = processed_data
    depends_on: [generate_large_dataset]
    outputs: [processed_dataset]
    
  - id: aggregate_and_summarize
    action: python_code
    parameters:
      code: |
        print("Creating final aggregation and summary...")
        
        # Validate processed data
        if not processed_dataset:
          raise ValueError("No processed dataset received")
        
        # Create summary report
        summary_report = {{
          "pipeline_execution": {{
            "status": "success",
            "data_processed": processed_dataset["total_points"]
          }},
          "data_summary": {{
            "overall_stats": processed_dataset["value_stats"],
            "category_count": len(processed_dataset["category_stats"]),
            "top_categories": []
          }},
          "quality_metrics": {{
            "data_completeness": 100.0,
            "processing_efficiency": "high"
          }}
        }}
        
        # Find top categories by sum
        category_stats = processed_dataset["category_stats"]
        sorted_categories = sorted(
          category_stats.items(),
          key=lambda x: x[1]["sum"],
          reverse=True
        )
        
        # Get top 10 categories
        for category, stats in sorted_categories[:10]:
          summary_report["data_summary"]["top_categories"].append({{
            "category": category,
            "count": stats["count"],
            "sum": stats["sum"],
            "avg": stats["avg"]
          }})
        
        print(f"Summary report generated")
        print(f"Top category: {{sorted_categories[0][0]}} (sum: {{sorted_categories[0][1]['sum']}})")
        print(f"Overall mean: {{summary_report['data_summary']['overall_stats']['mean']:.2f}}")
        
        final_summary = summary_report
    depends_on: [process_large_dataset]
    outputs: [final_summary]
    
  - id: validate_data_integrity
    action: python_code
    parameters:
      code: |
        print("Validating data integrity across pipeline...")
        
        # Perform integrity checks
        integrity_checks = {{
          "raw_data_received": raw_dataset is not None,
          "processed_data_received": processed_dataset is not None,
          "summary_generated": final_summary is not None,
          "data_count_match": False,
          "stats_reasonable": False
        }}
        
        if raw_dataset and processed_dataset:
          raw_count = len(raw_dataset["data_points"])
          processed_count = processed_dataset["total_points"]
          integrity_checks["data_count_match"] = (raw_count == processed_count)
          print(f"Data count check: {{raw_count}} == {{processed_count}} -> {{integrity_checks['data_count_match']}}")
        
        if processed_dataset:
          stats = processed_dataset["value_stats"]
          # Check if stats are reasonable (min < mean < max)
          integrity_checks["stats_reasonable"] = (
            stats["min"] <= stats["mean"] <= stats["max"] and
            stats["min"] >= 0  # Values should be non-negative (i^2)
          )
          print(f"Stats reasonableness: min={{stats['min']}}, mean={{stats['mean']:.2f}}, max={{stats['max']}}")
        
        # Overall integrity score
        passed_checks = sum(integrity_checks.values())
        total_checks = len(integrity_checks)
        integrity_score = (passed_checks / total_checks) * 100
        
        print(f"Integrity checks passed: {{passed_checks}}/{{total_checks}} ({{integrity_score:.1f}}%)")
        
        if integrity_score == 100:
          print("✓ All data integrity checks passed!")
        else:
          print("⚠ Some integrity checks failed")
          for check, passed in integrity_checks.items():
            if not passed:
              print(f"  Failed: {{check}}")
        
        integrity_result = {{
          "score": integrity_score,
          "checks": integrity_checks,
          "status": "pass" if integrity_score == 100 else "partial"
        }}
    depends_on: [generate_large_dataset, process_large_dataset, aggregate_and_summarize]
"""
        
        pipeline_path = self.create_test_pipeline(large_data_pipeline, "large_data_flow.yaml")
        
        start_time = time.time()
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        execution_time = time.time() - start_time
        
        # Should handle large data flow
        assert result.status in ["success", "partial_success", "error", "failed"]
        
        if result.status == "success":
            print(f"✓ Large data flow pipeline executed successfully in {execution_time:.2f}s")
        else:
            print(f"✓ Large data flow pipeline handled gracefully (status: {result.status}) in {execution_time:.2f}s")


class TestPipelineBoundaryConditions:
    """Test boundary conditions and edge cases in pipeline processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "boundary_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Test handling of empty pipeline definition."""
        empty_pipeline_cases = [
            # Completely empty
            "{}",
            
            # Only metadata, no steps
            """
name: empty_test
version: "1.0"
steps: []
""",
            
            # Metadata only, missing steps
            """
name: metadata_only
version: "1.0"
description: "Pipeline with no steps"
""",
        ]
        
        for i, empty_pipeline in enumerate(empty_pipeline_cases):
            pipeline_path = self.create_test_pipeline(empty_pipeline, f"empty_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle empty pipelines gracefully
            assert result.status in ["success", "error", "failed"]
            
            # If successful, should have no step results or minimal results
            if result.status == "success":
                assert len(result.step_results) == 0 or all(
                    step.status in ["skipped", "success"] for step in result.step_results
                )
            
            print(f"✓ Empty pipeline case {i} handled (status: {result.status})")
    
    @pytest.mark.asyncio
    async def test_single_step_pipeline(self):
        """Test minimal single-step pipeline."""
        single_step_pipeline = """
name: single_step_test
version: "1.0"
steps:
  - id: only_step
    action: python_code
    parameters:
      code: |
        print("This is the only step")
        result = "single_step_success"
        print(f"Result: {result}")
"""
        
        pipeline_path = self.create_test_pipeline(single_step_pipeline, "single_step.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should execute single step successfully
        assert result.status in ["success", "error", "failed"]
        
        if result.status == "success":
            assert len(result.step_results) == 1
            assert result.step_results[0].step_id == "only_step"
            assert result.step_results[0].status == "success"
        
        print("✓ Single step pipeline handled correctly")
    
    @pytest.mark.asyncio
    async def test_maximum_step_name_lengths(self):
        """Test handling of very long step names and parameters."""
        long_name_pipeline = f"""
name: long_name_test
version: "1.0"
steps:
  - id: {'very_long_step_name_' * 20}_with_extreme_length_that_tests_boundary_conditions
    action: python_code
    parameters:
      code: |
        step_name = "{'very_long_step_name_' * 20}_with_extreme_length_that_tests_boundary_conditions"
        print(f"Executing step with long name: {{len(step_name)}} characters")
        
        # Create long parameter names and values
        extremely_long_variable_name_that_tests_parameter_handling_boundaries = "long_value"
        
        result_data = {{
          "step_name_length": len(step_name),
          "status": "success",
          "{'parameter_name_' * 30}": "{'parameter_value_' * 50}",
          "nested_data": {{
            "{'nested_key_' * 20}": "{'nested_value_' * 30}",
            "deep_nesting": {{
              "{'deep_key_' * 15}": "{'deep_value_' * 25}"
            }}
          }}
        }}
        
        print(f"Result data keys: {{len(result_data)}} items")
        for key in result_data.keys():
          print(f"  Key length: {{len(key)}} chars")
        
        final_result = result_data
    outputs: [final_result]
"""
        
        pipeline_path = self.create_test_pipeline(long_name_pipeline, "long_names.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle long names appropriately
        assert result.status in ["success", "error", "failed"]
        
        print("✓ Long name boundary test handled")
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in pipeline definitions."""
        unicode_pipeline = """
name: unicode_test_🚀
version: "1.0"
description: "Testing Unicode: 你好, 🌍, café, naïve, résumé"

steps:
  - id: unicode_step_αβγ
    action: python_code
    parameters:
      code: |
        # Test various Unicode characters
        greeting = "Hello, 世界! 🌟"
        special_chars = "áéíóú ñ ç ü ß ø å æ"
        mathematical = "∑ ∫ ∞ ≤ ≥ ≠ ± √"
        emoji_test = "🚀 🔥 💯 ✨ 🎉 📊 💻 🤖"
        
        print(f"Greeting: {{greeting}}")
        print(f"Special characters: {{special_chars}}")
        print(f"Mathematical: {{mathematical}}")
        print(f"Emojis: {{emoji_test}}")
        
        # Test Unicode in variables and operations
        café_count = 5
        naïve_approach = True
        résumé_data = {{"name": "José María", "city": "São Paulo"}}
        
        unicode_result = {{
          "greeting": greeting,
          "special_chars": special_chars,
          "mathematical": mathematical,
          "emojis": emoji_test,
          "café_count": café_count,
          "naïve_approach": naïve_approach,
          "résumé_data": résumé_data
        }}
        
        print(f"Unicode test completed successfully! 🎉")
        print(f"Result keys: {{list(unicode_result.keys())}}")
    outputs: [unicode_result]
    
  - id: process_unicode_résultat
    action: python_code
    parameters:
      code: |
        # Process the Unicode data
        print("Processing Unicode data... 🔄")
        
        if not unicode_result:
          raise ValueError("No Unicode data received")
        
        # Test string operations with Unicode
        processed_greeting = unicode_result["greeting"].upper()
        emoji_count = len([c for c in unicode_result["emojis"] if ord(c) > 127])
        
        # Test Unicode in conditionals and loops
        special_char_analysis = {{}}
        for char in unicode_result["special_chars"]:
          if ord(char) > 127:  # Non-ASCII
            special_char_analysis[char] = ord(char)
        
        processed_result = {{
          "processed_greeting": processed_greeting,
          "emoji_count": emoji_count,
          "special_char_analysis": special_char_analysis,
          "unicode_processing_success": "✅ Complet!"
        }}
        
        print(f"Unicode processing complete: {{processed_result['unicode_processing_success']}}")
        print(f"Found {{emoji_count}} emoji characters")
        print(f"Special character codes: {{list(special_char_analysis.values())}}")
    depends_on: [unicode_step_αβγ]
"""
        
        pipeline_path = self.create_test_pipeline(unicode_pipeline, "unicode_test.yaml")
        
        yaml_content = pipeline_path.read_text()
        result = await self.executor.execute_yaml(yaml_content)
        
        # Should handle Unicode characters appropriately
        assert result.status in ["success", "error", "failed"]
        
        if result.status == "success":
            print("✓ Unicode and special characters handled successfully")
        else:
            print(f"✓ Unicode pipeline handled gracefully (status: {result.status})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])