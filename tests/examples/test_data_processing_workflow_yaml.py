"""Tests for data_processing_workflow.yaml example."""
import pytest
from .test_base import BaseExampleTest


class TestDataProcessingWorkflowYAML(BaseExampleTest):
    """Test the data processing workflow YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "data_processing_workflow.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "data_sources": ["database://sales", "api://inventory", "file://customers.csv"],
            "output_format": "parquet",
            "quality_threshold": 0.95
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check inputs
        assert 'inputs' in config
        assert 'data_sources' in config['inputs']
        assert config['inputs']['data_sources']['type'] == 'list'
        
        # Check parallel processing steps
        parallel_steps = [s for s in config['steps'] if s.get('loop', {}).get('parallel')]
        assert len(parallel_steps) > 0, "Should have parallel processing steps"
        
        # Check key steps
        step_ids = [step['id'] for step in config['steps']]
        assert 'discover_sources' in step_ids
        assert 'profile_data' in step_ids
        assert 'enrich_data' in step_ids
        assert 'validate_output' in step_ids
    
    def test_parallel_loop_configuration(self, pipeline_name):
        """Test parallel loop configurations."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find steps with loops
        loop_steps = [s for s in config['steps'] if 'loop' in s]
        
        for step in loop_steps:
            loop_config = step['loop']
            
            # Check loop has required fields
            if 'foreach' in loop_config:
                assert isinstance(loop_config['foreach'], str)
                assert '{{' in loop_config['foreach']  # Template variable
            
            # Check parallel configuration
            if 'parallel' in loop_config:
                assert isinstance(loop_config['parallel'], bool)
                if 'max_workers' in loop_config:
                    assert isinstance(loop_config['max_workers'], int)
    
    @pytest.mark.asyncio
    async def test_data_source_discovery(self, orchestrator, pipeline_name):
        """Test data source discovery step."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'discover_sources':
                    return {
                        'result': {
                            'sources': [
                                {'type': 'database', 'name': 'sales', 'size': '1GB'},
                                {'type': 'api', 'name': 'inventory', 'records': 50000},
                                {'type': 'file', 'name': 'customers.csv', 'rows': 10000}
                            ],
                            'total_sources': 3
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify discovery was called
            discovery_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'discover_sources'
            ]
            assert len(discovery_calls) == 1
    
    @pytest.mark.asyncio
    async def test_quality_validation_pass(self, orchestrator, pipeline_name):
        """Test quality validation with passing threshold."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                if step_id == 'validate_output':
                    return {
                        'result': {
                            'quality_score': 0.95,
                            'passed': True,
                            'issues': []
                        }
                    }
                elif step_id == 'aggregate_results':
                    return {
                        'result': {
                            'total_records': 10000,
                            'processing_time': 45.2
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify quality passed
            quality_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'validate_output'
            ]
            assert len(quality_calls) > 0
    
    @pytest.mark.asyncio
    async def test_quality_validation_fail(self, orchestrator, pipeline_name):
        """Test quality validation with failing threshold."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                if step_id == 'validate_output':
                    return {
                        'result': {
                            'quality_score': 0.85,
                            'passed': False,
                            'issues': ['Missing values in column X', 'Duplicates found']
                        }
                    }
                elif step_id == 'handle_quality_issues':
                    return {
                        'result': {
                            'resolution': 'Applied data cleaning',
                            'new_quality_score': 0.92
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify quality handling was triggered
            handling_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'handle_quality_issues'
            ]
            assert len(handling_calls) > 0
    
    def test_output_format_validation(self, pipeline_name):
        """Test output format configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find load_data step
        load_step = next(s for s in config['steps'] if s['id'] == 'load_data')
        
        # Check it references output format
        assert '{{output_format}}' in str(load_step)
        
        # Check supported formats in input definition
        output_format_input = config['inputs']['output_format']
        assert 'default' in output_format_input
        assert output_format_input['default'] == 'parquet'