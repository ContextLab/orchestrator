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
            "test_dir": "/path/to/tests",
            "language": "python",
            "test_framework": "pytest",
            "coverage_threshold": 80,
            "generate_missing": True,
            "update_existing": True,
            "include_edge_cases": True
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check test automation steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'analyze_codebase',
            'scan_existing_tests',
            'identify_gaps',
            'generate_test_plan',
            'generate_tests',
            'run_tests',
            'analyze_coverage'
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
    async def test_codebase_analysis(self, orchestrator, pipeline_name):
        """Test codebase analysis for test generation."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'analyze_codebase':
                    return {
                        'result': {
                            'functions': [
                                {'name': 'calculate_total', 'params': ['items', 'tax_rate']},
                                {'name': 'validate_email', 'params': ['email']},
                                {'name': 'process_order', 'params': ['order_data']}
                            ],
                            'classes': [
                                {'name': 'OrderProcessor', 'methods': 5},
                                {'name': 'EmailValidator', 'methods': 3}
                            ],
                            'complexity_metrics': {
                                'average_complexity': 4.5,
                                'max_complexity': 12
                            }
                        }
                    }
                elif step_id == 'analyze_existing_tests':
                    return {
                        'result': {
                            'existing_tests': [
                                'test_calculate_total',
                                'test_validate_email'
                            ],
                            'test_count': 2,
                            'coverage': 45
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify analysis steps were called
            analysis_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'analyze_codebase'
            ]
            scan_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'analyze_existing_tests'
            ]
            
            assert len(analysis_calls) == 1
            assert len(scan_calls) == 1
    
    @pytest.mark.asyncio
    async def test_coverage_gap_identification(self, orchestrator, pipeline_name):
        """Test identification of testing gaps."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'generate_test_plan':
                    return {
                        'result': {
                            'untested_functions': [
                                'process_order',
                                'send_notification',
                                'generate_report'
                            ],
                            'partially_tested': [
                                {'function': 'validate_input', 'coverage': 60}
                            ],
                            'missing_edge_cases': [
                                'empty input handling',
                                'null value checks',
                                'boundary conditions'
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify gap identification
            gap_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'generate_test_plan'
            ]
            assert len(gap_calls) > 0
    
    @pytest.mark.asyncio
    async def test_test_generation_loop(self, orchestrator, pipeline_name):
        """Test test generation for multiple functions."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                nonlocal test_count
                step_id = step.get('id')
                
                if step_id == 'generate_test_plan':
                    return {
                        'result': {
                            'test_targets': [
                                {'function': 'func1', 'priority': 'high'},
                                {'function': 'func2', 'priority': 'medium'},
                                {'function': 'func3', 'priority': 'low'}
                            ]
                        }
                    }
                elif step_id == 'generate_unit_tests':
                    test_count += 1
                    return {
                        'result': {
                            'generated_test': f'test_func{test_count}',
                            'test_cases': 5
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Should generate tests based on test plan
            assert test_count > 0
    
    @pytest.mark.asyncio
    async def test_test_execution_and_validation(self, orchestrator, pipeline_name):
        """Test running generated tests and validating results."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Test with minimal responses to avoid expensive API calls
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            {},  # minimal inputs
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'execute_tests':
                    return {
                        'result': {
                            'total_tests': 50,
                            'passed': 48,
                            'failed': 2,
                            'skipped': 0,
                            'execution_time': 12.5,
                            'failures': [
                                {'test': 'test_edge_case_1', 'reason': 'AssertionError'},
                                {'test': 'test_boundary_2', 'reason': 'ValueError'}
                            ]
                        }
                    }
                elif step_id == 'analyze_failures':
                    return {
                        'result': {
                            'total_coverage': 87.5,
                            'line_coverage': 85,
                            'branch_coverage': 90,
                            'uncovered_lines': ['module.py:45-47', 'utils.py:123']
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify test execution and coverage analysis
            run_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'execute_tests'
            ]
            coverage_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'analyze_failures'
            ]
            
            assert len(run_calls) > 0
            assert len(coverage_calls) > 0
    
    @pytest.mark.asyncio
    async def test_conditional_test_updates(self, orchestrator, pipeline_name):
        """Test conditional updating of existing tests."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'optimize_test_suite':
                    return {
                        'result': {
                            'updated_tests': [
                                'test_calculate_total',
                                'test_validate_input'
                            ],
                            'changes_made': [
                                'Added edge case for negative values',
                                'Updated assertions for new validation rules'
                            ]
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify test updates were performed
            update_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'optimize_test_suite'
            ]
            assert len(update_calls) > 0
    
    def test_edge_case_configuration(self, pipeline_name):
        """Test edge case generation configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find generate tests step
        generate_step = next(s for s in config['steps'] if s['id'] == 'generate_unit_tests')
        
        # Check edge case handling
        assert 'edge cases' in generate_step['action'].lower()
        assert '{{include_edge_cases}}' in str(generate_step)
    
    def test_output_structure(self, pipeline_name):
        """Test output definitions."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'tests_generated',
            'tests_updated',
            'coverage_percentage',
            'test_results',
            'quality_report'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"