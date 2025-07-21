"""Tests for code_analysis_suite.yaml example.

This test file follows the NO MOCKS policy. Tests use real orchestration
when API keys are available, otherwise they skip gracefully.
"""
import pytest
from .test_base import BaseExampleTest


class TestCodeAnalysisSuiteYAML(BaseExampleTest):
    """Test the code analysis suite YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "code_analysis_suite.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "repository_path": "/path/to/repo",
            "language": "python",
            "include_security": True,
            "include_performance": True,
            "include_complexity": True,
            "fix_issues": False,
            "output_format": "json"
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check analysis steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'scan_codebase',
            'static_analysis',
            'security_scan',
            'performance_analysis',
            'complexity_analysis',
            'generate_report'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_language_specific_configuration(self, pipeline_name):
        """Test language-specific analysis configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check language is used in analysis
        static_step = next(s for s in config['steps'] if s['id'] == 'static_analysis')
        assert 'language' in str(static_step), "Language should be referenced in static analysis"
    
    def test_conditional_steps(self, pipeline_name):
        """Test conditional security analysis."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check security scan has condition
        security_step = next(s for s in config['steps'] if s['id'] == 'security_scan')
        assert 'condition' in security_step, "Security scan should be conditional"
        assert 'include_security' in security_step['condition'], \
            "Security scan should depend on include_security flag"
    
    def test_auto_tags_present(self, pipeline_name):
        """Test that AUTO tags are used for analysis configuration."""
        # Load raw YAML to check for AUTO tags
        from pathlib import Path
        example_dir = Path(__file__).parent.parent.parent / "examples"
        pipeline_path = example_dir / pipeline_name
        
        with open(pipeline_path, 'r') as f:
            content = f.read()
        
        # Check for AUTO tags
        assert '<AUTO>' in content, "Pipeline should use AUTO tags for analysis configuration"
        assert '</AUTO>' in content, "AUTO tags should be properly closed"
    
    @pytest.mark.asyncio
    async def test_basic_execution_structure(self, orchestrator, pipeline_name, sample_inputs):
        """Test basic pipeline execution structure without full execution."""
        # This test verifies the pipeline can be loaded and initialized
        # Full execution would require a real code repository
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Verify the pipeline can be parsed and validated
        assert config is not None
        assert 'steps' in config
        assert len(config['steps']) > 0
        
        # Verify all required inputs are defined
        if 'inputs' in config:
            required_inputs = config['inputs']
            for input_key in required_inputs:
                assert input_key in sample_inputs, f"Missing required input: {input_key}"
    
    def test_output_format_handling(self, pipeline_name):
        """Test that output format is properly handled."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check report generation uses output format
        report_step = next(s for s in config['steps'] if s['id'] == 'generate_report')
        assert 'output_format' in str(report_step), \
            "Report generation should reference output format"
    
    def test_fix_issues_conditional(self, pipeline_name):
        """Test that fix_issues flag controls remediation."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check if there's a conditional fix step
        fix_steps = [s for s in config['steps'] if 'fix' in s.get('id', '').lower()]
        if fix_steps:
            for step in fix_steps:
                assert 'condition' in step or 'fix_issues' in str(step), \
                    f"Fix step {step['id']} should be conditional on fix_issues flag"
    
    def test_analysis_dependencies(self, pipeline_name):
        """Test that analysis steps have proper dependencies."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check that report generation depends on analysis steps
        report_step = next(s for s in config['steps'] if s['id'] == 'generate_report')
        if 'depends_on' in report_step:
            deps = report_step['depends_on']
            # At least one analysis step should be a dependency
            analysis_steps = ['static_analysis', 'security_scan', 
                            'performance_analysis', 'complexity_analysis']
            assert any(step in deps for step in analysis_steps), \
                "Report should depend on at least one analysis step"
    
    @pytest.mark.asyncio
    async def test_error_handling_structure(self, pipeline_name):
        """Test that pipeline has proper error handling structure."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check for error handling configuration
        if 'error_handling' in config:
            error_config = config['error_handling']
            assert 'strategy' in error_config, "Error handling should define a strategy"
            assert error_config['strategy'] in ['continue', 'fail', 'retry'], \
                "Error strategy should be valid"
        
        # Check individual steps for error handling
        critical_steps = ['scan_codebase', 'generate_report']
        for step_id in critical_steps:
            step = next((s for s in config['steps'] if s['id'] == step_id), None)
            if step and 'error_handling' in step:
                assert 'on_error' in step['error_handling'], \
                    f"Step {step_id} should define on_error behavior"


# Note: Full integration tests that would execute the pipeline against
# real code repositories are not included here as they would require
# actual repository access and could be time-consuming. The tests above
# verify the pipeline structure and configuration without mocks.