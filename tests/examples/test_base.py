"""Base test class for YAML examples."""
import os
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.tools.registry import ToolRegistry


class BaseExampleTest:
    """Base class for testing YAML examples."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return Orchestrator()
    
    @pytest.fixture
    def example_dir(self):
        """Get examples directory."""
        return Path(__file__).parent.parent.parent / "examples"
    
    def load_yaml_pipeline(self, pipeline_name: str) -> Dict[str, Any]:
        """Load YAML pipeline configuration."""
        example_dir = Path(__file__).parent.parent.parent / "examples"
        pipeline_path = example_dir / pipeline_name
        
        # Read the raw content
        with open(pipeline_path, 'r') as f:
            content = f.read()
        
        # Use the proper AUTO tag parser
        from orchestrator.compiler.auto_tag_yaml_parser import parse_yaml_with_auto_tags
        return parse_yaml_with_auto_tags(content)
    
    @pytest.fixture
    def mock_model_registry(self):
        """Create test model registry."""
        # Create a test registry
        class TestModelRegistry(ModelRegistry):
            def __init__(self):
                super().__init__()
                self.resolve_calls = []
                
            async def resolve_model(self, model_spec):
                """Test model resolution."""
                self.resolve_calls.append(model_spec)
                return {
                    "provider": "test",
                    "model": "test-model",
                    "temperature": 0.7
                }
        
        # Store original registry class
        import orchestrator.models.registry
        original_registry = orchestrator.models.registry.ModelRegistry
        
        # Replace with test registry
        test_registry = TestModelRegistry()
        orchestrator.models.registry.ModelRegistry = lambda: test_registry
        
        yield test_registry
        
        # Restore original registry
        orchestrator.models.registry.ModelRegistry = original_registry
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create test tool registry."""
        # Create a test registry
        class TestToolRegistry(ToolRegistry):
            def __init__(self):
                super().__init__()
                self.discover_calls = []
                
            async def discover_tool(self, action_desc):
                """Test tool discovery."""
                self.discover_calls.append(action_desc)
                
                # Simple tool mapping based on keywords
                action_lower = action_desc.lower()
                if "search" in action_lower or "web" in action_lower:
                    return {"tool": "web_search", "params": {}}
                elif "analyze" in action_lower:
                    return {"tool": "analyzer", "params": {}}
                elif "generate" in action_lower:
                    return {"tool": "generator", "params": {}}
                else:
                    return {"tool": "generic", "params": {}}
        
        # Store original registry class
        import orchestrator.tools.registry
        original_registry = orchestrator.tools.registry.ToolRegistry
        
        # Replace with test registry
        test_registry = TestToolRegistry()
        orchestrator.tools.registry.ToolRegistry = lambda: test_registry
        
        yield test_registry
        
        # Restore original registry
        orchestrator.tools.registry.ToolRegistry = original_registry
    
    async def run_pipeline_test(
        self,
        orchestrator: Orchestrator,
        pipeline_name: str,
        inputs: Dict[str, Any],
        expected_outputs: Optional[Dict[str, Any]] = None,
        mock_responses: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a pipeline test with mocked responses."""
        # Load pipeline
        pipeline_config = self.load_yaml_pipeline(pipeline_name)
        
        # Configure test responses if provided
        if mock_responses:
            # Store original execute_step method
            original_execute_step = orchestrator._execute_step if hasattr(orchestrator, '_execute_step') else None
            
            # Create a test execute_step that returns configured responses
            async def test_execute_step(step, context):
                step_id = step.id if hasattr(step, 'id') else step.get('id')
                if step_id in mock_responses:
                    return mock_responses[step_id]
                # Fall back to original if available
                if original_execute_step:
                    return await original_execute_step(step, context)
                return {}
            
            # Replace the method
            orchestrator._execute_step = test_execute_step
        
        try:
            # Run pipeline
            result = await orchestrator.execute_yaml(
                yaml.dump(pipeline_config),
                context=inputs
            )
            
            # Validate outputs if expected
            if expected_outputs:
                for key, expected_value in expected_outputs.items():
                    assert key in result['outputs']
                    if isinstance(expected_value, dict):
                        # For complex objects, check structure
                        assert isinstance(result['outputs'][key], dict)
                        for sub_key in expected_value:
                            assert sub_key in result['outputs'][key]
                    else:
                        assert result['outputs'][key] == expected_value
            
            return result
        finally:
            # Restore original method if we replaced it
            if mock_responses and original_execute_step:
                orchestrator._execute_step = original_execute_step
    
    def validate_pipeline_structure(self, pipeline_name: str):
        """Validate basic pipeline structure."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check required fields
        assert 'name' in config
        assert 'description' in config
        assert 'steps' in config
        assert isinstance(config['steps'], list)
        
        # Check each step
        for step in config['steps']:
            assert 'id' in step
            assert 'action' in step
            
            # Check for AUTO tags
            if isinstance(step['action'], str) and '<AUTO>' in step['action']:
                assert step['action'].count('<AUTO>') == step['action'].count('</AUTO>')
        
        # Check outputs if defined
        if 'outputs' in config:
            assert isinstance(config['outputs'], dict)
    
    def extract_auto_tags(self, pipeline_name: str) -> Dict[str, list]:
        """Extract all AUTO tags from a pipeline."""
        config = self.load_yaml_pipeline(pipeline_name)
        auto_tags = {}
        
        for step in config['steps']:
            if isinstance(step['action'], str) and '<AUTO>' in step['action']:
                # Extract content between AUTO tags
                import re
                pattern = r'<AUTO>(.*?)</AUTO>'
                matches = re.findall(pattern, step['action'], re.DOTALL)
                if matches:
                    auto_tags[step['id']] = matches
        
        return auto_tags