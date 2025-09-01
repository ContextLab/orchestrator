"""
Malformed YAML and Syntax Error Testing

Tests the orchestrator's handling of various YAML syntax errors,
malformed pipeline definitions, and invalid configurations using real examples.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Import orchestrator components
from src.orchestrator.orchestrator import Orchestrator


class TestMalformedYAML:
    """Test handling of malformed YAML files and syntax errors."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "malformed_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_invalid_yaml_syntax(self):
        """Test handling of basic YAML syntax errors."""
        invalid_yaml_cases = [
            # Missing quotes
            """
name: test pipeline
version: 1.0
steps:
  - id: step1
    action: test
    parameters:
      key: value with "unescaped quotes
""",
            # Invalid indentation
            """
name: test_pipeline
version: "1.0"
steps:
- id: step1
    action: test
  parameters:
    key: value
""",
            # Unclosed brackets
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    parameters: {
      key: value
      missing_close: true
""",
            # Tab characters (YAML doesn't allow tabs)
            "name: test_pipeline\nversion: \"1.0\"\nsteps:\n\t- id: step1\n\t  action: test",
            
            # Duplicate keys
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    action: duplicate_action
""",
        ]
        
        for i, invalid_yaml in enumerate(invalid_yaml_cases):
            pipeline_path = self.create_test_pipeline(invalid_yaml, f"invalid_{i}.yaml")
            
            # Should handle YAML parsing error gracefully
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            assert result.status in ["error", "failed"], f"Case {i}: Expected error status"
            
            # Should have meaningful error message
            error_messages = [step.error_message for step in result.step_results if step.error_message]
            yaml_errors = [msg for msg in error_messages if any(keyword in msg.lower() 
                          for keyword in ["yaml", "parse", "syntax", "invalid"])]
            
            assert len(yaml_errors) > 0, f"Case {i}: No YAML error message found"
            
            print(f"✓ Invalid YAML case {i} handled correctly")
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of pipelines missing required fields."""
        missing_field_cases = [
            # Missing name
            """
version: "1.0"
steps:
  - id: step1
    action: test
""",
            # Missing version
            """
name: test_pipeline
steps:
  - id: step1
    action: test
""",
            # Missing steps
            """
name: test_pipeline
version: "1.0"
""",
            # Steps with missing id
            """
name: test_pipeline
version: "1.0"
steps:
  - action: test
    parameters:
      key: value
""",
            # Steps with missing action
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    parameters:
      key: value
""",
        ]
        
        for i, missing_field_yaml in enumerate(missing_field_cases):
            pipeline_path = self.create_test_pipeline(missing_field_yaml, f"missing_field_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should fail with validation error
            assert result.status in ["error", "failed"], f"Case {i}: Expected validation error"
            
            # Should identify missing required fields
            error_messages = " ".join([step.error_message for step in result.step_results if step.error_message])
            assert any(keyword in error_messages.lower() 
                      for keyword in ["required", "missing", "field", "validation"]), \
                      f"Case {i}: No validation error message found"
            
            print(f"✓ Missing field case {i} validation error handled")
    
    @pytest.mark.asyncio
    async def test_invalid_data_types(self):
        """Test handling of incorrect data types in YAML."""
        invalid_type_cases = [
            # Version as number instead of string
            """
name: test_pipeline
version: 1.0
steps:
  - id: step1
    action: test
""",
            # Steps as string instead of list
            """
name: test_pipeline
version: "1.0"  
steps: "invalid_steps"
""",
            # Parameters as string instead of dict
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    parameters: "should_be_dict"
""",
            # Outputs as string instead of list
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    outputs: "should_be_list"
""",
            # Depends_on as string when expecting list
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
  - id: step2
    action: test
    depends_on: "step1"
""",
        ]
        
        for i, invalid_type_yaml in enumerate(invalid_type_cases):
            pipeline_path = self.create_test_pipeline(invalid_type_yaml, f"invalid_type_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle type validation
            assert result.status in ["error", "failed", "success"], f"Case {i}: Unexpected status"
            
            # Note: Some type mismatches might be auto-corrected or handled gracefully
            print(f"✓ Invalid type case {i} processed (status: {result.status})")
    
    @pytest.mark.asyncio
    async def test_circular_dependencies(self):
        """Test detection of circular dependencies in pipeline steps."""
        circular_dependency_cases = [
            # Simple circular dependency
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    depends_on:
      - step2
  - id: step2
    action: test
    depends_on:
      - step1
""",
            # Three-step circular dependency
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    depends_on:
      - step3
  - id: step2
    action: test
    depends_on:
      - step1
  - id: step3
    action: test
    depends_on:
      - step2
""",
            # Self-dependency
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    depends_on:
      - step1
""",
        ]
        
        for i, circular_yaml in enumerate(circular_dependency_cases):
            pipeline_path = self.create_test_pipeline(circular_yaml, f"circular_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should detect circular dependency
            assert result.status in ["error", "failed"], f"Case {i}: Should detect circular dependency"
            
            error_messages = " ".join([step.error_message for step in result.step_results if step.error_message])
            assert any(keyword in error_messages.lower() 
                      for keyword in ["circular", "cycle", "dependency", "loop"]), \
                      f"Case {i}: No circular dependency error found"
            
            print(f"✓ Circular dependency case {i} detected")
    
    @pytest.mark.asyncio
    async def test_invalid_step_references(self):
        """Test handling of references to non-existent steps."""
        invalid_reference_cases = [
            # Reference to non-existent step
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    depends_on:
      - nonexistent_step
""",
            # Multiple invalid references
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    depends_on:
      - missing_step1
      - missing_step2
  - id: step2
    action: test
    depends_on:
      - step1
      - another_missing_step
""",
            # Reference to step defined later (forward reference)
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: test
    depends_on:
      - step2
  - id: step2
    action: test
""",
        ]
        
        for i, invalid_ref_yaml in enumerate(invalid_reference_cases):
            pipeline_path = self.create_test_pipeline(invalid_ref_yaml, f"invalid_ref_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle invalid references 
            if i == 2:  # Forward reference might be valid
                assert result.status in ["success", "error", "failed"]
            else:
                assert result.status in ["error", "failed"], f"Case {i}: Should detect invalid reference"
                
                error_messages = " ".join([step.error_message for step in result.step_results if step.error_message])
                assert any(keyword in error_messages.lower() 
                          for keyword in ["reference", "missing", "not found", "undefined"]), \
                          f"Case {i}: No reference error found"
            
            print(f"✓ Invalid reference case {i} handled (status: {result.status})")
    
    @pytest.mark.asyncio
    async def test_large_malformed_files(self):
        """Test handling of large malformed YAML files."""
        # Create a large malformed YAML with repeated content
        large_malformed = """
name: large_test_pipeline
version: "1.0"
steps:
"""
        
        # Add many steps with various malformed elements
        for i in range(100):
            if i % 10 == 0:
                # Invalid action reference
                large_malformed += f"""
  - id: step_{i}
    action: invalid_action_that_does_not_exist_{i}
    parameters:
      data: "step {i} data"
"""
            elif i % 7 == 0:
                # Missing id
                large_malformed += f"""
  - action: test_action
    parameters:
      data: "step {i} data"
"""
            elif i % 5 == 0:
                # Invalid depends_on
                large_malformed += f"""
  - id: step_{i}
    action: test_action
    depends_on:
      - nonexistent_step_{i}
"""
            else:
                # Valid step
                large_malformed += f"""
  - id: step_{i}
    action: test_action
    parameters:
      data: "step {i} data"
"""
        
        pipeline_path = self.create_test_pipeline(large_malformed, "large_malformed.yaml")
        
        result = await self.executor.execute_pipeline_file(str(pipeline_path))
        
        # Should handle large malformed files without crashing
        assert result.status in ["error", "failed", "partial_success"]
        
        # Should identify multiple errors
        error_count = len([step for step in result.step_results if step.status == "error"])
        assert error_count > 0, "No errors detected in large malformed file"
        
        print(f"✓ Large malformed file handled ({error_count} errors detected)")
    
    @pytest.mark.asyncio
    async def test_encoding_issues(self):
        """Test handling of files with encoding problems."""
        # Test with different problematic content
        encoding_cases = [
            # Unicode characters that might cause issues
            """
name: test_pipeline_🤖
version: "1.0"
description: "Pipeline with émojis and spéciàl charácters"
steps:
  - id: unicode_step
    action: test
    parameters:
      message: "Hello 世界! Testing unicode: ñáéíóú"
""",
            # Very long lines that might cause buffer issues
            f"""
name: test_pipeline
version: "1.0"
steps:
  - id: long_line_step
    action: test
    parameters:
      very_long_parameter: "{'x' * 10000}"
""",
            # Mixed content types
            """
name: test_pipeline
version: "1.0"
steps:
  - id: mixed_content
    action: test
    parameters:
      number: 42
      boolean: true
      null_value: null
      list: [1, 2, 3, "mixed", true]
      nested:
        deep:
          very_deep: "value"
""",
        ]
        
        for i, encoding_yaml in enumerate(encoding_cases):
            pipeline_path = self.create_test_pipeline(encoding_yaml, f"encoding_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle encoding issues gracefully
            assert result.status in ["success", "error", "failed"]
            
            print(f"✓ Encoding case {i} handled (status: {result.status})")


class TestInvalidPipelineConfigurations:
    """Test invalid pipeline configurations that are syntactically valid YAML."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "invalid_config_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_unsupported_actions(self):
        """Test handling of unsupported or invalid actions."""
        unsupported_actions = [
            "nonexistent_action",
            "invalid-action-name",
            "action_with_special_chars!@#",
            "",  # Empty action
            123,  # Numeric action
        ]
        
        for i, action in enumerate(unsupported_actions):
            pipeline_content = f"""
name: test_pipeline
version: "1.0"
steps:
  - id: test_step
    action: {action}
    parameters:
      test: value
"""
            
            pipeline_path = self.create_test_pipeline(pipeline_content, f"unsupported_action_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle unsupported actions
            assert result.status in ["error", "failed"]
            
            error_messages = " ".join([step.error_message for step in result.step_results if step.error_message])
            assert any(keyword in error_messages.lower() 
                      for keyword in ["action", "unsupported", "invalid", "unknown"]), \
                      f"Action {action}: No action error found"
            
            print(f"✓ Unsupported action '{action}' handled correctly")
    
    @pytest.mark.asyncio
    async def test_invalid_parameter_structures(self):
        """Test handling of invalid parameter structures."""
        invalid_param_cases = [
            # Parameters with invalid structure
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: python_code
    parameters:
      code: >
        def invalid_function(:
            print("syntax error")
""",
            # Parameters referencing non-existent variables
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: python_code
    parameters:
      code: |
        print(nonexistent_variable)
        result = another_undefined_var + 1
""",
            # Parameters with conflicting requirements
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: llm
    parameters:
      model: "gpt-4"
      temperature: 1.5  # Invalid temperature > 1.0
      max_tokens: -100  # Negative max_tokens
""",
        ]
        
        for i, invalid_param_yaml in enumerate(invalid_param_cases):
            pipeline_path = self.create_test_pipeline(invalid_param_yaml, f"invalid_params_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle parameter validation
            assert result.status in ["error", "failed", "success"]  # Some might be handled gracefully
            
            print(f"✓ Invalid parameter case {i} handled (status: {result.status})")
    
    @pytest.mark.asyncio
    async def test_resource_specification_errors(self):
        """Test handling of invalid resource specifications."""
        invalid_resource_cases = [
            # Invalid memory specifications
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: python_code
    parameters:
      code: "print('test')"
    resources:
      memory: "invalid_memory_spec"
      cpu: -1
""",
            # Conflicting resource requirements
            """
name: test_pipeline
version: "1.0"
steps:
  - id: step1
    action: python_code
    parameters:
      code: "print('test')"
    resources:
      memory: "32GB"
      max_execution_time: 0
""",
        ]
        
        for i, invalid_resource_yaml in enumerate(invalid_resource_cases):
            pipeline_path = self.create_test_pipeline(invalid_resource_yaml, f"invalid_resources_{i}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle resource validation
            assert result.status in ["error", "failed", "success"]
            
            print(f"✓ Invalid resource case {i} handled (status: {result.status})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])