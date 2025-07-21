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
            "repo_path": "/path/to/repo",
            "languages": ["python", "javascript"],
            "analysis_depth": "comprehensive",
            "security_scan": True,
            "performance_check": True,
            "doc_check": True,
            "severity_threshold": "medium"
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check analysis steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'discover_code',
            'static_analysis',
            'ai_code_review',
            'performance_analysis',
            'documentation_check',
            'generate_report'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_language_specific_configuration(self, pipeline_name):
        """Test language-specific analysis configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check languages is used in discovery
        discover_step = next(s for s in config['steps'] if s['id'] == 'discover_code')
        assert 'languages' in str(discover_step), "Languages should be referenced in code discovery"
    
    def test_conditional_steps(self, pipeline_name):
        """Test conditional security analysis."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check static analysis has security scan condition
        static_step = next(s for s in config['steps'] if s['id'] == 'static_analysis')
        assert 'condition' in static_step, "Static analysis should be conditional"
        assert 'security_scan' in static_step['condition'], \
            "Static analysis should depend on security_scan flag"
    
    @pytest.mark.asyncio
    async def test_basic_execution_structure(self, orchestrator, pipeline_name, sample_inputs):
        """Test basic pipeline execution structure without full execution."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'quality_score': (int, str),  # Can be int or str from template
                'total_issues': (int, str),
                'files_analyzed': (int, str)
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    def test_output_format_handling(self, pipeline_name):
        """Test that output format is properly handled."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check report generation exists
        report_step = next(s for s in config['steps'] if s['id'] == 'generate_report')
        assert 'markdown' in str(report_step).lower(), \
            "Report generation should create markdown report"
    
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
            analysis_steps = ['generate_insights', 'test_coverage', 'architecture_review']
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
        critical_steps = ['generate_report']
        for step_id in critical_steps:
            step = next((s for s in config['steps'] if s['id'] == step_id), None)
            if step:
                assert 'on_error' in step, \
                    f"Step {step_id} should define on_error behavior"


# Note: Full integration tests that would execute the pipeline against
# real code repositories are not included here as they would require
# actual repository access and could be time-consuming. The tests above
# verify the pipeline structure and configuration without mocks.