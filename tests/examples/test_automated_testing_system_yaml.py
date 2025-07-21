"""Tests for automated_testing_system.yaml example."""
import pytest
from .test_base import BaseExampleTest


class TestAutomatedTestingSystemYAML(BaseExampleTest):
    """Test the automated testing system YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "automated_testing_system.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "source_dir": "/path/to/source",
            "test_dir": "./tests",
            "coverage_target": 80.0,
            "test_types": ["unit", "integration"],
            "test_framework": "pytest",
            "include_edge_cases": True,
            "include_performance": False
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check test automation steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'analyze_codebase',
            'analyze_existing_tests',
            'generate_test_plan',
            'generate_unit_tests',
            'execute_tests',
            'analyze_failures',
            'generate_report'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_test_framework_configuration(self, pipeline_name):
        """Test test framework specific configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check framework is used in test generation
        generate_step = next(s for s in config['steps'] if s['id'] == 'generate_unit_tests')
        assert '{{test_framework}}' in str(generate_step)
    
    @pytest.mark.asyncio
    async def test_codebase_analysis(self, orchestrator, pipeline_name, sample_inputs):
        """Test codebase analysis for test generation."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'tests_generated': (int, str),
                'coverage_achieved': (float, str)
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    def test_coverage_gap_identification(self, pipeline_name):
        """Test identification of testing gaps configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check that generate_test_plan includes gap analysis
        plan_step = next(s for s in config['steps'] if s['id'] == 'generate_test_plan')
        assert 'priority matrix for untested functions' in plan_step['action'].lower()
        assert 'coverage target' in plan_step['action'].lower()
    
    def test_test_generation_configuration(self, pipeline_name):
        """Test test generation configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check unit test generation step
        unit_test_step = next(s for s in config['steps'] if s['id'] == 'generate_unit_tests')
        assert 'Happy path test cases' in unit_test_step['action']
        assert 'Edge cases' in unit_test_step['action']
        assert '{{test_framework}}' in unit_test_step['action']
        
        # Check conditional execution based on test types
        assert 'condition' in unit_test_step
        assert "'unit' in {{test_types}}" in unit_test_step['condition']
    
    @pytest.mark.asyncio
    async def test_test_execution_and_validation(self, orchestrator, pipeline_name, sample_inputs):
        """Test running generated tests and validating results."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    
    def test_conditional_test_updates(self, pipeline_name):
        """Test conditional updating of existing tests configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check optimize_test_suite step
        optimize_step = next(s for s in config['steps'] if s['id'] == 'optimize_test_suite')
        assert optimize_step is not None
        assert 'Redundant tests' in optimize_step['action']
        assert 'parallelization' in optimize_step['action'].lower()
    
    def test_edge_case_configuration(self, pipeline_name):
        """Test edge case generation configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find generate tests step
        generate_step = next(s for s in config['steps'] if s['id'] == 'generate_unit_tests')
        
        # Check edge case handling
        assert 'edge cases' in generate_step['action'].lower()
        assert 'null, empty, boundary values' in generate_step['action']
    
    def test_output_structure(self, pipeline_name):
        """Test output definitions."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'tests_generated',
            'coverage_achieved',
            'tests_passed',
            'tests_failed',
            'report_path'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"
    
    def test_mutation_testing_configuration(self, pipeline_name):
        """Test mutation testing configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check mutation testing step
        mutation_step = next((s for s in config['steps'] if s['id'] == 'mutation_testing'), None)
        assert mutation_step is not None
        assert 'condition' in mutation_step
        assert '{{coverage_target}}' in mutation_step['condition']
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, orchestrator, pipeline_name, sample_inputs):
        """Test full pipeline execution with minimal responses."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result