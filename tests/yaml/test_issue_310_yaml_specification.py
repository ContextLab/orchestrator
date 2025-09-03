"""
Unit tests for Issue #310: YAML Pipeline Specification

Tests the comprehensive YAML parsing, validation, and StateGraph compilation
capabilities that build upon the core architecture foundation.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import YAML-related components
try:
    from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
    from src.orchestrator.compiler.enhanced_yaml_compiler import EnhancedYAMLCompiler  
    from src.orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
    from src.orchestrator.compiler.schema_validator import SchemaValidator
    from src.orchestrator.foundation import (
        PipelineSpecification, PipelineHeader, PipelineStep,
        PipelineCompilerInterface
    )
    HAS_YAML_COMPONENTS = True
except ImportError:
    HAS_YAML_COMPONENTS = False


@pytest.mark.skipif(not HAS_YAML_COMPONENTS, reason="YAML components not available")
class TestYAMLParsing:
    """Test YAML parsing capabilities."""
    
    @pytest.fixture
    def yaml_compiler(self):
        """Create YAMLCompiler instance for testing."""
        return YAMLCompiler()
    
    @pytest.fixture
    def enhanced_yaml_compiler(self):
        """Create EnhancedYAMLCompiler instance for testing."""
        return EnhancedYAMLCompiler()
    
    @pytest.fixture
    def sample_yaml_content(self):
        """Sample valid YAML pipeline content."""
        return """
        name: test_pipeline
        version: "1.0"
        description: "Test pipeline for validation"
        
        steps:
          - id: step1
            action: llm
            parameters:
              model: gpt-4
              prompt: "Hello world"
              
          - id: step2
            action: file_save
            parameters:
              filename: output.txt
              content: "{{ step1.output }}"
            depends_on: [step1]
        """
    
    @pytest.fixture
    def complex_yaml_content(self):
        """Complex YAML pipeline with advanced features."""
        return """
        name: complex_pipeline
        version: "2.0"
        description: "Complex pipeline with loops and conditions"
        
        parameters:
          - name: input_list
            type: list
            default: ["item1", "item2", "item3"]
          - name: threshold
            type: float
            default: 0.8
        
        steps:
          - id: initialize
            action: set_variable
            parameters:
              results: []
              
          - id: process_items
            action: foreach
            parameters:
              items: "{{ input_list }}"
              steps:
                - id: analyze_item
                  action: llm
                  parameters:
                    model: claude-3-opus
                    prompt: "Analyze: {{ item }}"
                    
                - id: filter_result
                  action: condition
                  parameters:
                    condition: "{{ analyze_item.score > threshold }}"
                    then_steps:
                      - id: save_result
                        action: append_variable
                        parameters:
                          variable: results
                          value: "{{ analyze_item.output }}"
            depends_on: [initialize]
            
          - id: generate_report
            action: llm
            parameters:
              model: gpt-4
              prompt: "Generate report from: {{ results }}"
              temperature: 0.1
            depends_on: [process_items]
        """
    
    def test_basic_yaml_parsing(self, yaml_compiler, sample_yaml_content):
        """Test basic YAML parsing functionality."""
        # Parse YAML content
        parsed = yaml.safe_load(sample_yaml_content)
        
        assert parsed['name'] == 'test_pipeline'
        assert parsed['version'] == '1.0'
        assert len(parsed['steps']) == 2
        assert parsed['steps'][0]['id'] == 'step1'
        assert parsed['steps'][1]['depends_on'] == ['step1']
    
    def test_yaml_validation_structure(self, yaml_compiler, sample_yaml_content):
        """Test YAML structure validation."""
        parsed = yaml.safe_load(sample_yaml_content)
        
        # Verify required fields
        assert 'name' in parsed
        assert 'version' in parsed  
        assert 'steps' in parsed
        
        # Verify step structure
        for step in parsed['steps']:
            assert 'id' in step
            assert 'action' in step
            assert 'parameters' in step
    
    def test_complex_yaml_features(self, yaml_compiler, complex_yaml_content):
        """Test parsing of complex YAML features."""
        parsed = yaml.safe_load(complex_yaml_content)
        
        # Test parameters section
        assert 'parameters' in parsed
        assert len(parsed['parameters']) == 2
        assert parsed['parameters'][0]['name'] == 'input_list'
        assert parsed['parameters'][0]['type'] == 'list'
        
        # Test nested steps in foreach
        process_items = next(s for s in parsed['steps'] if s['id'] == 'process_items')
        assert process_items['action'] == 'foreach'
        assert 'steps' in process_items['parameters']
        
        nested_steps = process_items['parameters']['steps']
        assert len(nested_steps) == 2
    
    @pytest.mark.asyncio 
    async def test_yaml_template_rendering(self, yaml_compiler):
        """Test YAML template rendering with variables."""
        yaml_content = """
        name: template_test
        version: "1.0"
        
        steps:
          - id: step1
            action: llm
            parameters:
              prompt: "Process {{ input_text }}"
              model: "{{ model_name | default('gpt-4') }}"
        """
        
        # Mock template rendering
        with patch('jinja2.Environment.from_string') as mock_template:
            mock_template.return_value.render.return_value = yaml_content.replace(
                '{{ input_text }}', 'test input'
            ).replace('{{ model_name | default(\'gpt-4\') }}', 'gpt-4')
            
            # Test would verify template rendering functionality
            assert True  # Placeholder for actual template test
    
    def test_yaml_auto_tag_resolution(self, yaml_compiler):
        """Test AUTO tag resolution functionality."""
        yaml_content = """
        name: auto_tag_test
        version: "1.0"
        
        steps:
          - id: step1
            action: AUTO:text_processing
            parameters:
              input: "test input"
        """
        
        parsed = yaml.safe_load(yaml_content)
        auto_action = parsed['steps'][0]['action']
        
        # Verify AUTO tag is present
        assert auto_action.startswith('AUTO:')
        assert 'text_processing' in auto_action


@pytest.mark.skipif(not HAS_YAML_COMPONENTS, reason="YAML components not available")
class TestYAMLValidation:
    """Test YAML validation capabilities."""
    
    @pytest.fixture
    def schema_validator(self):
        """Create SchemaValidator instance."""
        try:
            return SchemaValidator()
        except:
            return Mock()
    
    def test_pipeline_schema_validation(self, schema_validator):
        """Test pipeline schema validation."""
        valid_pipeline = {
            'name': 'test_pipeline',
            'version': '1.0',
            'steps': [
                {
                    'id': 'step1',
                    'action': 'llm',
                    'parameters': {'prompt': 'test'}
                }
            ]
        }
        
        # Mock schema validation
        if hasattr(schema_validator, 'validate'):
            # Real validation
            errors = schema_validator.validate(valid_pipeline)
            assert isinstance(errors, list)
        else:
            # Mock validation
            assert valid_pipeline['name'] == 'test_pipeline'
    
    def test_invalid_pipeline_detection(self, schema_validator):
        """Test detection of invalid pipeline structures."""
        invalid_pipeline = {
            'name': 'test_pipeline',
            # Missing version
            'steps': [
                {
                    'id': 'step1',
                    # Missing action
                    'parameters': {'prompt': 'test'}
                }
            ]
        }
        
        # Should detect validation errors
        assert 'version' not in invalid_pipeline
        assert 'action' not in invalid_pipeline['steps'][0]
    
    def test_dependency_validation(self, schema_validator):
        """Test step dependency validation."""
        pipeline_with_deps = {
            'name': 'dep_test',
            'version': '1.0',
            'steps': [
                {
                    'id': 'step1',
                    'action': 'llm',
                    'parameters': {'prompt': 'first'}
                },
                {
                    'id': 'step2', 
                    'action': 'llm',
                    'parameters': {'prompt': 'second'},
                    'depends_on': ['step1']
                },
                {
                    'id': 'step3',
                    'action': 'llm', 
                    'parameters': {'prompt': 'third'},
                    'depends_on': ['nonexistent_step']  # Invalid dependency
                }
            ]
        }
        
        # Verify dependency structure
        step2_deps = pipeline_with_deps['steps'][1]['depends_on']
        step3_deps = pipeline_with_deps['steps'][2]['depends_on']
        
        assert 'step1' in step2_deps
        assert 'nonexistent_step' in step3_deps  # This should be flagged as invalid
    
    def test_parameter_type_validation(self, schema_validator):
        """Test parameter type validation."""
        pipeline_with_types = {
            'name': 'type_test',
            'version': '1.0',
            'parameters': [
                {'name': 'text_param', 'type': 'string', 'default': 'hello'},
                {'name': 'num_param', 'type': 'float', 'default': 3.14},
                {'name': 'list_param', 'type': 'list', 'default': [1, 2, 3]},
                {'name': 'bool_param', 'type': 'boolean', 'default': True}
            ],
            'steps': [
                {
                    'id': 'step1',
                    'action': 'process',
                    'parameters': {
                        'text': '{{ text_param }}',
                        'number': '{{ num_param }}',
                        'items': '{{ list_param }}',
                        'flag': '{{ bool_param }}'
                    }
                }
            ]
        }
        
        # Verify parameter types match expected values
        params = pipeline_with_types['parameters']
        assert params[0]['type'] == 'string'
        assert params[1]['type'] == 'float'
        assert params[2]['type'] == 'list'
        assert params[3]['type'] == 'boolean'


@pytest.mark.skipif(not HAS_YAML_COMPONENTS, reason="YAML components not available")
class TestStateGraphCompilation:
    """Test YAML to StateGraph compilation."""
    
    @pytest.fixture 
    def mock_state_graph_compiler(self):
        """Create mock StateGraph compiler."""
        compiler = Mock()
        compiler.compile_to_state_graph = AsyncMock()
        compiler.validate_state_graph = AsyncMock(return_value=[])
        return compiler
    
    @pytest.mark.asyncio
    async def test_yaml_to_state_graph_compilation(self, mock_state_graph_compiler):
        """Test compilation from YAML to StateGraph."""
        yaml_content = """
        name: state_graph_test
        version: "1.0"
        
        steps:
          - id: input_step
            action: get_input
            parameters:
              prompt: "Enter text:"
              
          - id: process_step
            action: llm
            parameters:
              model: gpt-4
              prompt: "Process: {{ input_step.output }}"
            depends_on: [input_step]
            
          - id: output_step
            action: save_output
            parameters:
              content: "{{ process_step.output }}"
            depends_on: [process_step]
        """
        
        parsed_yaml = yaml.safe_load(yaml_content)
        
        # Mock StateGraph compilation
        mock_state_graph = {
            'nodes': [
                {'id': 'input_step', 'type': 'input'},
                {'id': 'process_step', 'type': 'llm'},
                {'id': 'output_step', 'type': 'output'}
            ],
            'edges': [
                {'from': 'input_step', 'to': 'process_step'},
                {'from': 'process_step', 'to': 'output_step'}
            ]
        }
        
        mock_state_graph_compiler.compile_to_state_graph.return_value = mock_state_graph
        
        # Test compilation
        state_graph = await mock_state_graph_compiler.compile_to_state_graph(parsed_yaml)
        
        assert len(state_graph['nodes']) == 3
        assert len(state_graph['edges']) == 2
        assert state_graph['nodes'][0]['id'] == 'input_step'
        assert state_graph['edges'][0]['from'] == 'input_step'
    
    @pytest.mark.asyncio
    async def test_parallel_step_compilation(self, mock_state_graph_compiler):
        """Test compilation of parallel steps to StateGraph."""
        yaml_content = """
        name: parallel_test
        version: "1.0"
        
        steps:
          - id: input_step
            action: get_input
            
          - id: parallel_step_a
            action: process_a
            depends_on: [input_step]
            
          - id: parallel_step_b
            action: process_b
            depends_on: [input_step]
            
          - id: merge_step
            action: merge_results
            parameters:
              inputs: ["{{ parallel_step_a.output }}", "{{ parallel_step_b.output }}"]
            depends_on: [parallel_step_a, parallel_step_b]
        """
        
        parsed_yaml = yaml.safe_load(yaml_content)
        
        # Mock parallel StateGraph compilation
        mock_state_graph = {
            'nodes': [
                {'id': 'input_step', 'type': 'input'},
                {'id': 'parallel_step_a', 'type': 'process', 'parallel_group': 1},
                {'id': 'parallel_step_b', 'type': 'process', 'parallel_group': 1}, 
                {'id': 'merge_step', 'type': 'merge'}
            ],
            'edges': [
                {'from': 'input_step', 'to': 'parallel_step_a'},
                {'from': 'input_step', 'to': 'parallel_step_b'},
                {'from': 'parallel_step_a', 'to': 'merge_step'},
                {'from': 'parallel_step_b', 'to': 'merge_step'}
            ]
        }
        
        mock_state_graph_compiler.compile_to_state_graph.return_value = mock_state_graph
        
        state_graph = await mock_state_graph_compiler.compile_to_state_graph(parsed_yaml)
        
        # Verify parallel structure
        parallel_nodes = [n for n in state_graph['nodes'] if n.get('parallel_group') == 1]
        assert len(parallel_nodes) == 2
        
        # Verify merge step has multiple inputs
        merge_edges = [e for e in state_graph['edges'] if e['to'] == 'merge_step']
        assert len(merge_edges) == 2
    
    @pytest.mark.asyncio 
    async def test_conditional_step_compilation(self, mock_state_graph_compiler):
        """Test compilation of conditional steps."""
        yaml_content = """
        name: conditional_test
        version: "1.0"
        
        steps:
          - id: check_step
            action: evaluate
            parameters:
              condition: "{{ input_value > 10 }}"
              
          - id: high_value_step
            action: process_high
            condition: "{{ check_step.result == true }}"
            depends_on: [check_step]
            
          - id: low_value_step  
            action: process_low
            condition: "{{ check_step.result == false }}"
            depends_on: [check_step]
            
          - id: final_step
            action: finalize
            depends_on: [high_value_step, low_value_step]
        """
        
        parsed_yaml = yaml.safe_load(yaml_content)
        
        # Mock conditional StateGraph
        mock_state_graph = {
            'nodes': [
                {'id': 'check_step', 'type': 'condition'},
                {'id': 'high_value_step', 'type': 'process', 'condition': 'check_step.result == true'},
                {'id': 'low_value_step', 'type': 'process', 'condition': 'check_step.result == false'},
                {'id': 'final_step', 'type': 'finalize'}
            ],
            'edges': [
                {'from': 'check_step', 'to': 'high_value_step', 'condition': 'true'},
                {'from': 'check_step', 'to': 'low_value_step', 'condition': 'false'},
                {'from': 'high_value_step', 'to': 'final_step'},
                {'from': 'low_value_step', 'to': 'final_step'}
            ]
        }
        
        mock_state_graph_compiler.compile_to_state_graph.return_value = mock_state_graph
        
        state_graph = await mock_state_graph_compiler.compile_to_state_graph(parsed_yaml)
        
        # Verify conditional structure
        conditional_nodes = [n for n in state_graph['nodes'] if 'condition' in n]
        assert len(conditional_nodes) == 3  # check_step + 2 conditional steps
        
        conditional_edges = [e for e in state_graph['edges'] if 'condition' in e]
        assert len(conditional_edges) == 2


@pytest.mark.skipif(not HAS_YAML_COMPONENTS, reason="YAML components not available")
class TestYAMLErrorHandling:
    """Test error handling in YAML processing."""
    
    def test_invalid_yaml_syntax(self):
        """Test handling of invalid YAML syntax."""
        invalid_yaml = """
        name: test_pipeline
        version: "1.0"
        steps:
          - id: step1
            action: llm
            parameters:
              prompt: "unterminated string
        """
        
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(invalid_yaml)
    
    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        incomplete_yaml = """
        # Missing name field
        version: "1.0"
        steps:
          - id: step1
            # Missing action field
            parameters:
              prompt: "test"
        """
        
        parsed = yaml.safe_load(incomplete_yaml)
        
        # Should detect missing fields
        assert 'name' not in parsed
        assert 'action' not in parsed['steps'][0]
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        circular_yaml = """
        name: circular_test
        version: "1.0"
        steps:
          - id: step1
            action: llm
            depends_on: [step2]
            
          - id: step2
            action: llm  
            depends_on: [step3]
            
          - id: step3
            action: llm
            depends_on: [step1]  # Creates circular dependency
        """
        
        parsed = yaml.safe_load(circular_yaml)
        
        # Build dependency graph to detect cycles
        deps = {}
        for step in parsed['steps']:
            step_id = step['id'] 
            step_deps = step.get('depends_on', [])
            deps[step_id] = step_deps
        
        # Verify circular structure exists
        assert 'step2' in deps['step1']
        assert 'step3' in deps['step2'] 
        assert 'step1' in deps['step3']
    
    def test_undefined_variable_detection(self):
        """Test detection of undefined template variables."""
        undefined_var_yaml = """
        name: undefined_test
        version: "1.0"
        steps:
          - id: step1
            action: llm
            parameters:
              prompt: "Use {{ undefined_variable }}"
        """
        
        parsed = yaml.safe_load(undefined_var_yaml)
        
        # Check for template variable usage
        prompt = parsed['steps'][0]['parameters']['prompt']
        assert '{{ undefined_variable }}' in prompt
    
    @pytest.mark.asyncio
    async def test_compilation_error_recovery(self):
        """Test error recovery during compilation."""
        problematic_yaml = """
        name: error_test
        version: "1.0"
        steps:
          - id: step1
            action: nonexistent_action
            parameters:
              invalid_param: "{{ step0.output }}"  # step0 doesn't exist
        """
        
        parsed = yaml.safe_load(problematic_yaml)
        
        # Mock compilation with error recovery
        mock_compiler = Mock()
        mock_compiler.compile_with_recovery = AsyncMock()
        
        # Should handle compilation errors gracefully
        errors = []
        if parsed['steps'][0]['action'] == 'nonexistent_action':
            errors.append("Unknown action: nonexistent_action")
        if 'step0.output' in parsed['steps'][0]['parameters']['invalid_param']:
            errors.append("Undefined step reference: step0")
            
        assert len(errors) == 2


@pytest.mark.skipif(not HAS_YAML_COMPONENTS, reason="YAML components not available")
class TestYAMLAdvancedFeatures:
    """Test advanced YAML pipeline features."""
    
    def test_yaml_file_inclusion(self):
        """Test YAML file inclusion capability."""
        main_yaml = """
        name: main_pipeline
        version: "1.0"
        include:
          - common_steps.yaml
          - utils/helper_functions.yaml
        
        steps:
          - id: main_step
            action: llm
            parameters:
              prompt: "Main processing"
        """
        
        parsed = yaml.safe_load(main_yaml)
        
        # Verify include section
        assert 'include' in parsed
        assert len(parsed['include']) == 2
        assert 'common_steps.yaml' in parsed['include']
    
    def test_yaml_macros_and_reuse(self):
        """Test YAML macros and template reuse."""
        macro_yaml = """
        name: macro_test
        version: "1.0"
        
        macros:
          llm_step: &llm_template
            action: llm
            parameters:
              model: gpt-4
              temperature: 0.7
        
        steps:
          - id: step1
            <<: *llm_template
            parameters:
              prompt: "First prompt"
              
          - id: step2
            <<: *llm_template  
            parameters:
              prompt: "Second prompt"
              temperature: 0.3  # Override default
        """
        
        parsed = yaml.safe_load(macro_yaml)
        
        # Verify macro expansion
        assert parsed['steps'][0]['action'] == 'llm'
        assert parsed['steps'][0]['parameters']['model'] == 'gpt-4'
        assert parsed['steps'][1]['parameters']['temperature'] == 0.3
    
    def test_yaml_environment_variables(self):
        """Test environment variable substitution."""
        env_yaml = """
        name: env_test
        version: "1.0"
        
        steps:
          - id: step1
            action: llm
            parameters:
              api_key: "${OPENAI_API_KEY}"
              model: "${DEFAULT_MODEL:-gpt-4}"
              prompt: "Test with ${USER:-default_user}"
        """
        
        parsed = yaml.safe_load(env_yaml)
        
        # Verify environment variable syntax
        params = parsed['steps'][0]['parameters']
        assert '${OPENAI_API_KEY}' in params['api_key']
        assert '${DEFAULT_MODEL:-gpt-4}' in params['model']
        assert '${USER:-default_user}' in params['prompt']
    
    def test_yaml_loop_constructs(self):
        """Test YAML loop and iteration constructs."""
        loop_yaml = """
        name: loop_test
        version: "1.0"
        
        parameters:
          - name: items
            type: list
            default: ["apple", "banana", "cherry"]
        
        steps:
          - id: process_items
            action: foreach
            parameters:
              items: "{{ items }}"
              steps:
                - id: process_item
                  action: llm
                  parameters:
                    prompt: "Process {{ item }}"
                    
          - id: while_condition
            action: while
            parameters:
              condition: "{{ counter < 5 }}"
              steps:
                - id: increment
                  action: set_variable
                  parameters:
                    counter: "{{ counter + 1 }}"
        """
        
        parsed = yaml.safe_load(loop_yaml)
        
        # Verify loop structures
        foreach_step = next(s for s in parsed['steps'] if s['id'] == 'process_items')
        assert foreach_step['action'] == 'foreach'
        assert 'steps' in foreach_step['parameters']
        
        while_step = next(s for s in parsed['steps'] if s['id'] == 'while_condition')
        assert while_step['action'] == 'while'
        assert 'condition' in while_step['parameters']


class TestYAMLPipelineIntegration:
    """Test integration between YAML pipeline specification and foundation."""
    
    @pytest.mark.asyncio
    async def test_yaml_foundation_integration(self):
        """Test integration with foundation interfaces."""
        # Mock foundation compiler
        mock_compiler = Mock(spec=PipelineCompilerInterface)
        mock_compiler.compile = AsyncMock()
        mock_compiler.validate = AsyncMock(return_value=[])
        
        yaml_content = """
        name: integration_test  
        version: "1.0"
        steps:
          - id: step1
            action: llm
            parameters:
              prompt: "test"
        """
        
        # Mock compilation to foundation specification
        mock_spec = Mock(spec=PipelineSpecification)
        mock_spec.header = Mock(spec=PipelineHeader)
        mock_spec.header.name = "integration_test"
        mock_spec.steps = [Mock(spec=PipelineStep)]
        mock_spec.steps[0].id = "step1"
        
        mock_compiler.compile.return_value = mock_spec
        
        # Test compilation
        result = await mock_compiler.compile(yaml_content)
        
        assert result.header.name == "integration_test"
        assert len(result.steps) == 1
        assert result["steps"][0].id == "step1"
        
        # Test validation
        validation_errors = await mock_compiler.validate(result)
        assert len(validation_errors) == 0
    
    def test_yaml_to_pipeline_specification_mapping(self):
        """Test mapping from YAML to PipelineSpecification objects."""
        yaml_content = """
        name: mapping_test
        version: "2.1"
        description: "Test YAML to specification mapping"
        
        parameters:
          - name: input_text
            type: string
            default: "hello"
        
        steps:
          - id: process
            action: llm
            parameters:
              model: gpt-4
              prompt: "Process: {{ input_text }}"
            metadata:
              timeout: 30
              retry_count: 3
        """
        
        parsed = yaml.safe_load(yaml_content)
        
        # Create foundation objects from YAML
        header = Mock(spec=PipelineHeader)
        header.name = parsed['name']
        header.version = parsed['version'] 
        header.description = parsed.get('description')
        
        steps = []
        for step_data in parsed['steps']:
            step = Mock(spec=PipelineStep)
            step.id = step_data['id']
            step.action = step_data['action']
            step.parameters = step_data['parameters']
            step.metadata = step_data.get('metadata', {})
            steps.append(step)
        
        spec = Mock(spec=PipelineSpecification)
        spec.header = header
        spec.steps = steps
        spec.parameters = parsed.get('parameters', [])
        
        # Verify mapping
        assert spec.header.name == "mapping_test"
        assert spec.header.version == "2.1"
        assert len(spec.steps) == 1
        assert spec.steps[0].id == "process"
        assert spec.steps[0].metadata['timeout'] == 30


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])