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
            "source": "s3://data-lake/raw/*.parquet",
            "output_path": "s3://data-lake/processed/",
            "output_format": "parquet",
            "quality_threshold": 0.95,
            "chunk_size": 10000,
            "parallel_workers": 4
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check inputs
        assert 'inputs' in config
        assert 'source' in config['inputs']
        assert config['inputs']['source']['type'] == 'string'
        assert config['inputs']['source']['required'] == True
        
        # Check for parallel processing configuration
        parallel_steps = [s for s in config['steps'] if s.get('parallel', False)]
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
            
            # Check loop configuration
            assert 'over' in loop_config
            assert loop_config['over'].startswith('{{')
            
            # Check if parallel
            if 'parallel' in loop_config:
                assert isinstance(loop_config['parallel'], bool)
                if 'max_workers' in loop_config:
                    assert isinstance(loop_config['max_workers'], int)
    
    @pytest.mark.asyncio
    async def test_data_source_discovery(self, orchestrator, pipeline_name, sample_inputs):
        """Test data source discovery step."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'total_records_processed': int,
                'processing_time': str  # It's a string like "5.2s"
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    @pytest.mark.asyncio
    async def test_quality_validation_pass(self, orchestrator, pipeline_name):
        """Test quality validation with passing threshold."""
        inputs = {
            "source": "database://sales",
            "output_path": "/tmp/output/",
            "output_format": "parquet",
            "quality_threshold": 0.90
        }
        
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            inputs,
            expected_outputs={
                'data_quality_score': float
            },
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_quality_validation_fail(self, orchestrator, pipeline_name):
        """Test quality validation with failing threshold."""
        inputs = {
            "source": "database://sales",
            "output_path": "/tmp/output/",
            "output_format": "parquet",
            "quality_threshold": 0.99  # Very high threshold
        }
        
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            inputs,
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    
    def test_data_transformation_steps(self, pipeline_name):
        """Test data transformation step configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find transformation steps
        transform_steps = [s for s in config['steps'] if 'transform' in s.get('id', '').lower()]
        
        for step in transform_steps:
            # Check basic properties
            assert 'action' in step
            assert 'depends_on' in step or step == config['steps'][0]
            
            # Check for appropriate tags
            if 'tags' in step:
                assert isinstance(step['tags'], list)
    
    def test_output_configuration(self, pipeline_name):
        """Test output configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check outputs section
        assert 'outputs' in config
        outputs = config['outputs']
        
        # Check key outputs
        assert 'total_records_processed' in outputs
        assert 'processing_time' in outputs
        assert 'data_quality_score' in outputs
        
        # Verify output references
        for output_name, output_value in outputs.items():
            if isinstance(output_value, str) and '{{' in output_value:
                # Should reference step results
                assert '.result' in output_value
    
    def test_error_handling_configuration(self, pipeline_name):
        """Test error handling configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find steps with error handling
        error_handling_steps = [s for s in config['steps'] if 'on_error' in s]
        
        assert len(error_handling_steps) > 0, "Should have error handling configured"
        
        for step in error_handling_steps:
            on_error = step['on_error']
            
            # Check error handling configuration
            assert 'action' in on_error
            
            # Check for retry configuration
            if 'retry_count' in on_error:
                assert isinstance(on_error['retry_count'], int)
                assert on_error['retry_count'] > 0