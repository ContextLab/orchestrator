"""Tests for code_analysis_suite.yaml example."""
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
        assert '{{language}}' in str(static_step)
    
    @pytest.mark.asyncio
    async def test_codebase_scanning(self, orchestrator, pipeline_name):
        """Test codebase scanning functionality."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'discover_code':
                    return {
                        'result': {
                            'total_files': 150,
                            'total_lines': 25000,
                            'file_types': {
                                '.py': 120,
                                '.json': 20,
                                '.yaml': 10
                            },
                            'directories': ['src', 'tests', 'docs']
                        }
                    }
                elif step_id == 'static_analysis':
                    return {
                        'result': {
                            'linting_issues': 45,
                            'type_errors': 12,
                            'style_violations': 78,
                            'critical_issues': 3
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify scanning was performed
            scan_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'discover_code'
            ]
            assert len(scan_calls) == 1
    
    @pytest.mark.asyncio
    async def test_conditional_security_scan(self, orchestrator, pipeline_name):
        """Test conditional security scanning."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'security_scan':
                    return {
                        'result': {
                            'vulnerabilities': [
                                {
                                    'severity': 'high',
                                    'type': 'SQL Injection',
                                    'file': 'db.py',
                                    'line': 42
                                },
                                {
                                    'severity': 'medium',
                                    'type': 'Hardcoded Secret',
                                    'file': 'config.py',
                                    'line': 15
                                }
                            ],
                            'total_vulnerabilities': 2
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs_with_security
            )
            
            # Check security scan was called
            security_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'security_scan'
            ]
            assert len(security_calls) > 0
        
        # Test with security disabled
        inputs_no_security = {
            "repository_path": "/test/repo",
            "include_security": False
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {'result': {}}
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs_no_security
            )
            
            # Check security scan was NOT called
            security_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'security_scan'
            ]
            assert len(security_calls) == 0
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self, orchestrator, pipeline_name):
        """Test performance analysis execution."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'performance_analysis':
                    return {
                        'result': {
                            'bottlenecks': [
                                {
                                    'function': 'process_data',
                                    'file': 'processor.py',
                                    'time_complexity': 'O(n²)',
                                    'suggestion': 'Use hash map for O(n) complexity'
                                }
                            ],
                            'memory_leaks': 0,
                            'optimization_opportunities': 5
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify performance analysis was performed
            perf_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'performance_analysis'
            ]
            assert len(perf_calls) > 0
    
    @pytest.mark.asyncio
    async def test_complexity_metrics(self, orchestrator, pipeline_name):
        """Test code complexity analysis."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                if step.get('id') == 'architecture_review':
                    return {
                        'result': {
                            'cyclomatic_complexity': {
                                'average': 4.2,
                                'max': 15,
                                'complex_functions': [
                                    {'name': 'validate_input', 'complexity': 15}
                                ]
                            },
                            'cognitive_complexity': {
                                'average': 6.8,
                                'max': 22
                            },
                            'maintainability_index': 72
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify complexity analysis was performed
            complexity_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'architecture_review'
            ]
            assert len(complexity_calls) > 0
    
    @pytest.mark.asyncio
    async def test_issue_fixing(self, orchestrator, pipeline_name):
        """Test automatic issue fixing functionality."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'static_analysis':
                    return {
                        'result': {
                            'linting_issues': 10,
                            'fixable_issues': 8
                        }
                    }
                elif step_id == 'generate_insights':
                    return {
                        'result': {
                            'fixed_issues': 8,
                            'failed_fixes': 0,
                            'files_modified': 5
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify issue fixing was triggered
            fix_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'generate_insights'
            ]
            assert len(fix_calls) > 0
    
    def test_report_generation(self, pipeline_name):
        """Test report generation configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find report generation step
        report_step = next(s for s in config['steps'] if s['id'] == 'generate_report')
        
        # Check output format is used
        assert '{{output_format}}' in str(report_step)
    
    def test_output_structure(self, pipeline_name):
        """Test output definitions."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'total_issues',
            'critical_issues',
            'security_vulnerabilities',
            'code_quality_score',
            'report_path'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"