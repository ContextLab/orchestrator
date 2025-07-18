"""Base test class for YAML examples."""
import os
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from unittest.mock import Mock, patch, AsyncMock

from orchestrator import Orchestrator


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
        """Mock model registry for tests."""
        with patch('orchestrator.models.registry.ModelRegistry') as mock:
            registry = Mock()
            
            # Mock model resolution
            async def mock_resolve(model_spec):
                return {
                    "provider": "mock",
                    "model": "mock-model",
                    "temperature": 0.7
                }
            
            registry.resolve_model = AsyncMock(side_effect=mock_resolve)
            mock.return_value = registry
            yield registry
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Mock tool registry for tests."""
        with patch('orchestrator.tools.registry.ToolRegistry') as mock:
            registry = Mock()
            
            # Mock tool discovery
            async def mock_discover(action_desc):
                # Simple tool mapping based on keywords
                if "search" in action_desc.lower() or "web" in action_desc.lower():
                    return {"tool": "web_search", "params": {}}
                elif "analyze" in action_desc.lower():
                    return {"tool": "analyzer", "params": {}}
                elif "generate" in action_desc.lower():
                    return {"tool": "generator", "params": {}}
                else:
                    return {"tool": "generic", "params": {}}
            
            registry.discover_tool = AsyncMock(side_effect=mock_discover)
            mock.return_value = registry
            yield registry
    
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
        
        # Mock tool executions if provided
        if mock_responses:
            for step_id, response in mock_responses.items():
                # Mock the step execution
                with patch.object(
                    orchestrator,
                    '_execute_step',
                    new_callable=AsyncMock
                ) as mock_exec:
                    mock_exec.return_value = response
        
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