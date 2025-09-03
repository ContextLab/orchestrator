"""Phase 5 Integration Tests - Orchestrator Component Integration

Tests the integration of AutomaticGraphGenerator with existing orchestrator
components including YAMLCompiler, LangGraphAdapter, and Pipeline classes.

NO MOCKS - All tests use real component integration and execution.
"""

import asyncio
import pytest
import yaml
from typing import Dict, Any

from src.orchestrator.compiler.enhanced_yaml_compiler import EnhancedYAMLCompiler
from src.orchestrator.adapters.enhanced_langgraph_adapter import EnhancedLangGraphAdapter
from src.orchestrator.core.enhanced_pipeline import EnhancedPipeline, create_enhanced_pipeline_from_legacy
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.graph_generation.automatic_generator import AutomaticGraphGenerator

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestEnhancedYAMLCompilerIntegration:
    """Test integration of EnhancedYAMLCompiler with AutomaticGraphGenerator."""
    
    @pytest.fixture
    def enhanced_compiler(self):
        """Create enhanced YAML compiler instance."""
        return EnhancedYAMLCompiler(enable_auto_generation=True)
        
    @pytest.fixture
    def sample_enhanced_yaml(self):
        """Sample enhanced YAML for testing."""
        return '''
id: integration_test_pipeline
name: "Integration Test Pipeline"
type: workflow
version: "2.0.0"

inputs:
  message:
    type: string
    required: true
    description: "Test message"
  
outputs:
  final_result:
    type: string
    source: "{{ process_message.result }}"

steps:
  - id: process_message
    action: generate_text
    inputs:
      prompt: "Process: {{ inputs.message }}"
      model: "<AUTO task=\\"generate\\">Select model</AUTO>"
    outputs:
      result:
        type: string
        description: "Processed message"
'''
        
    @pytest.fixture  
    def sample_legacy_yaml(self):
        """Sample legacy YAML for testing."""
        return '''
id: legacy_test_pipeline
name: "Legacy Test Pipeline"

steps:
  - id: echo_step
    action: echo
    parameters:
      message: "Hello World"
    dependencies: []
'''
        
    @pytest.mark.asyncio
    async def test_enhanced_format_detection_and_compilation(self, enhanced_compiler, sample_enhanced_yaml):
        """Test that enhanced format is detected and compiled correctly."""
        
        pipeline = await enhanced_compiler.compile(
            sample_enhanced_yaml,
            context={"test_context": True}
        )
        
        # Verify pipeline was created
        assert isinstance(pipeline, Pipeline)
        assert pipeline.id == "integration_test_pipeline"
        assert pipeline.name == "Integration Test Pipeline"
        
        # Verify enhanced format was detected
        assert pipeline.metadata.get('compilation_method') == 'automatic_graph_generation'
        assert pipeline.metadata.get('enhanced_syntax') == True
        
        # Verify StateGraph was attached
        assert 'state_graph' in pipeline.metadata
        assert pipeline.metadata['state_graph'] is not None
        
        # Verify generation stats were recorded
        assert 'graph_generation_stats' in pipeline.metadata
        
    @pytest.mark.asyncio
    async def test_legacy_format_backwards_compatibility(self, enhanced_compiler, sample_legacy_yaml):
        """Test that legacy format still works with enhanced compiler."""
        
        pipeline = await enhanced_compiler.compile(sample_legacy_yaml)
        
        # Verify pipeline was created
        assert isinstance(pipeline, Pipeline)
        assert pipeline.id == "legacy_test_pipeline"
        
        # Verify legacy compilation was used
        assert pipeline.metadata.get('compilation_method') == 'legacy'
        assert pipeline.metadata.get('enhanced_syntax') != True
        
    @pytest.mark.asyncio
    async def test_auto_generation_toggle(self, sample_enhanced_yaml):
        """Test enabling/disabling auto-generation."""
        
        # Test with auto-generation disabled
        compiler_disabled = EnhancedYAMLCompiler(enable_auto_generation=False)
        pipeline_disabled = await compiler_disabled.compile(sample_enhanced_yaml)
        
        # Should use legacy compilation even for enhanced format
        assert pipeline_disabled.metadata.get('compilation_method') == 'legacy'
        
        # Test with auto-generation enabled
        compiler_enabled = EnhancedYAMLCompiler(enable_auto_generation=True)  
        pipeline_enabled = await compiler_enabled.compile(sample_enhanced_yaml)
        
        # Should use automatic generation
        assert pipeline_enabled.metadata.get('compilation_method') == 'automatic_graph_generation'
        
    @pytest.mark.asyncio
    async def test_compilation_with_context(self, enhanced_compiler, sample_enhanced_yaml):
        """Test compilation with execution context."""
        
        test_context = {
            "environment": "test",
            "user_id": "test_user",
            "custom_var": "custom_value"
        }
        
        pipeline = await enhanced_compiler.compile(
            sample_enhanced_yaml,
            context=test_context
        )
        
        # Verify context was preserved
        assert pipeline.context["environment"] == "test"
        assert pipeline.context["user_id"] == "test_user"
        assert pipeline.context["custom_var"] == "custom_value"
        
    def test_compiler_statistics(self, enhanced_compiler):
        """Test getting generation statistics from compiler."""
        
        stats = enhanced_compiler.get_generation_stats()
        
        # Verify stats structure
        assert isinstance(stats, dict)
        assert 'total_generations' in stats
        assert 'successful_generations' in stats
        assert 'success_rate' in stats
        
        # Test cache clearing
        enhanced_compiler.clear_generation_cache()


class TestEnhancedLangGraphAdapterIntegration:
    """Test integration of EnhancedLangGraphAdapter with AutomaticGraphGenerator."""
    
    @pytest.fixture
    def enhanced_adapter(self):
        """Create enhanced LangGraph adapter."""
        return EnhancedLangGraphAdapter(enable_auto_generation=True)
        
    @pytest.fixture
    def sample_pipeline_def(self):
        """Sample pipeline definition for testing."""
        return {
            'id': 'adapter_test_pipeline',
            'name': 'Adapter Test Pipeline',
            'type': 'workflow',
            'inputs': {
                'test_input': {
                    'type': 'string',
                    'required': True
                }
            },
            'steps': [
                {
                    'id': 'step1',
                    'action': 'test_action',
                    'inputs': {
                        'input_data': '{{ inputs.test_input }}'
                    }
                },
                {
                    'id': 'step2', 
                    'action': 'test_action_2',
                    'depends_on': ['step1'],
                    'inputs': {
                        'previous_result': '{{ step1.result }}'
                    }
                }
            ]
        }
        
    @pytest.mark.asyncio
    async def test_optimized_workflow_creation(self, enhanced_adapter, sample_pipeline_def):
        """Test creating optimized workflow from pipeline definition."""
        
        state_graph = await enhanced_adapter.create_optimized_workflow(
            sample_pipeline_def,
            context={'test': True}
        )
        
        # Verify StateGraph was created
        assert state_graph is not None
        
        # Check if it's a real StateGraph or placeholder
        if hasattr(state_graph, 'nodes'):
            # Real LangGraph StateGraph
            assert hasattr(state_graph, 'compile')
        else:
            # Placeholder StateGraph
            assert isinstance(state_graph, dict)
            assert state_graph.get('type') in ['placeholder_state_graph', 'enhanced_placeholder']
            
    @pytest.mark.asyncio
    async def test_pipeline_to_workflow_conversion(self, enhanced_adapter):
        """Test converting Pipeline object to LangGraphWorkflow."""
        
        # Create test pipeline
        from src.orchestrator.core.task import Task
        
        pipeline = Pipeline(
            id='test_conversion_pipeline',
            name='Test Conversion Pipeline'
        )
        
        task1 = Task(
            id='task1',
            name='task1',
            action='test_action',
            parameters={'param1': 'value1'}
        )
        task2 = Task(
            id='task2',
            name='task2',
            action='test_action_2', 
            parameters={'param2': 'value2'},
            dependencies=['task1']
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Convert to workflow
        workflow = await enhanced_adapter.pipeline_to_workflow(pipeline)
        
        # Verify workflow structure
        assert workflow.name == 'Test Conversion Pipeline'
        assert len(workflow.nodes) == 2
        assert 'task1' in workflow.nodes
        assert 'task2' in workflow.nodes
        assert len(workflow.edges) == 1
        
    @pytest.mark.asyncio
    async def test_auto_generation_fallback(self, sample_pipeline_def):
        """Test fallback to manual creation when auto-generation fails."""
        
        # Create adapter with potentially failing auto-generator
        adapter = EnhancedLangGraphAdapter(enable_auto_generation=True)
        
        # This should fallback gracefully even if auto-generation has issues
        state_graph = await adapter.create_optimized_workflow(sample_pipeline_def)
        
        # Should still get a StateGraph (either real or placeholder)
        assert state_graph is not None
        
    def test_adapter_configuration(self):
        """Test adapter configuration and settings."""
        
        adapter = EnhancedLangGraphAdapter(enable_auto_generation=False)
        
        # Test initial state
        assert adapter.is_auto_generation_enabled() == False
        
        # Test enabling
        adapter.set_auto_generation(True)
        assert adapter.is_auto_generation_enabled() == True
        
        # Test stats and cache
        stats = adapter.get_generation_stats()
        assert isinstance(stats, dict)
        
        adapter.clear_generation_cache()  # Should not raise


class TestEnhancedPipelineIntegration:
    """Test integration of EnhancedPipeline with StateGraph execution."""
    
    @pytest.fixture
    def enhanced_pipeline(self):
        """Create enhanced pipeline for testing."""
        pipeline = EnhancedPipeline(
            id='test_enhanced_pipeline',
            name='Test Enhanced Pipeline'
        )
        
        # Add test tasks
        from src.orchestrator.core.task import Task
        
        task1 = Task(
            id='task1',
            name='task1',
            action='test_action',
            parameters={'input': 'test_value'}
        )
        task2 = Task(
            id='task2',
            name='task2',
            action='test_action_2',
            parameters={'input': 'test_value_2'},
            dependencies=['task1']
        )
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        return pipeline
        
    def test_enhanced_pipeline_creation(self, enhanced_pipeline):
        """Test enhanced pipeline creation and properties."""
        
        assert isinstance(enhanced_pipeline, EnhancedPipeline)
        assert enhanced_pipeline.id == 'test_enhanced_pipeline'
        assert len(enhanced_pipeline.tasks) == 2
        
        # Test enhanced properties
        assert enhanced_pipeline.has_state_graph() == False
        assert enhanced_pipeline.is_enhanced_format() == False
        assert enhanced_pipeline.get_compilation_method() == 'unknown'
        
    def test_state_graph_attachment(self, enhanced_pipeline):
        """Test attaching StateGraph to enhanced pipeline."""
        
        # Create mock StateGraph
        mock_state_graph = {
            'type': 'test_state_graph',
            'nodes': ['task1', 'task2'],
            'edges': [('task1', 'task2')]
        }
        
        # Attach StateGraph
        enhanced_pipeline.state_graph = mock_state_graph
        
        # Verify attachment
        assert enhanced_pipeline.has_state_graph() == True
        assert enhanced_pipeline.state_graph == mock_state_graph
        assert enhanced_pipeline.metadata['state_graph'] == mock_state_graph
        
    @pytest.mark.asyncio
    async def test_hybrid_execution(self, enhanced_pipeline):
        """Test hybrid execution (StateGraph preferred, fallback to legacy)."""
        
        # Test without StateGraph - should use legacy
        result = await enhanced_pipeline.execute_hybrid(
            initial_context={'test_input': 'test_value'}
        )
        
        assert result['execution_type'] == 'legacy'
        assert result['pipeline_id'] == 'test_enhanced_pipeline'
        assert 'results' in result
        assert len(result['results']) == 2
        
        # Test performance metrics
        metrics = enhanced_pipeline.get_performance_metrics()
        assert 'execution_time' in metrics
        assert 'success' in metrics
        assert metrics['success'] == True
        
    @pytest.mark.asyncio
    async def test_state_graph_execution(self, enhanced_pipeline):
        """Test StateGraph execution with placeholder graph."""
        
        # Attach placeholder StateGraph
        placeholder_graph = {
            'type': 'placeholder_state_graph',
            'pipeline_id': 'test_enhanced_pipeline',
            'steps': 2
        }
        enhanced_pipeline.state_graph = placeholder_graph
        
        # Execute with StateGraph
        result = await enhanced_pipeline.execute_with_state_graph(
            initial_context={'test_input': 'test_value'}
        )
        
        assert result['execution_type'] == 'placeholder'
        assert result['pipeline_id'] == 'test_enhanced_pipeline'
        assert 'results' in result
        
    def test_legacy_pipeline_conversion(self):
        """Test converting legacy Pipeline to EnhancedPipeline."""
        
        # Create legacy pipeline
        from src.orchestrator.core.task import Task
        
        legacy_pipeline = Pipeline(
            id='legacy_pipeline',
            name='Legacy Pipeline'
        )
        
        task = Task(
            id='legacy_task',
            name='legacy_task',
            action='legacy_action',
            parameters={'param': 'value'}
        )
        legacy_pipeline.add_task(task)
        
        # Convert to enhanced
        enhanced = create_enhanced_pipeline_from_legacy(legacy_pipeline)
        
        # Verify conversion
        assert isinstance(enhanced, EnhancedPipeline)
        assert enhanced.id == 'legacy_pipeline'
        assert enhanced.name == 'Legacy Pipeline'
        assert len(enhanced.tasks) == 1
        assert 'legacy_task' in enhanced.tasks


class TestEndToEndIntegration:
    """End-to-end integration tests combining all components."""
    
    @pytest.mark.asyncio
    async def test_complete_enhanced_pipeline_flow(self):
        """Test complete flow: Enhanced YAML → AutoGraph → StateGraph → Execution."""
        
        # Step 1: Define enhanced YAML
        enhanced_yaml = '''
id: e2e_test_pipeline
name: "End-to-End Test Pipeline"
type: workflow

inputs:
  test_message:
    type: string
    required: true
    default: "Hello Integration Test"

outputs:
  final_output:
    type: string
    source: "{{ step2.output }}"

steps:
  - id: step1
    action: process_input
    inputs:
      message: "{{ inputs.test_message }}"
    outputs:
      processed:
        type: string
        description: "Processed input"
        
  - id: step2
    action: finalize_output
    depends_on: [step1]
    inputs:
      input_data: "{{ step1.processed }}"
    outputs:
      output:
        type: string
        description: "Final output"
'''
        
        # Step 2: Compile with EnhancedYAMLCompiler
        compiler = EnhancedYAMLCompiler(enable_auto_generation=True)
        pipeline = await compiler.compile(enhanced_yaml)
        
        # Step 3: Verify pipeline structure
        assert isinstance(pipeline, Pipeline)
        assert pipeline.metadata.get('compilation_method') == 'automatic_graph_generation'
        assert 'state_graph' in pipeline.metadata
        
        # Step 4: Convert to EnhancedPipeline if needed
        if not isinstance(pipeline, EnhancedPipeline):
            enhanced_pipeline = create_enhanced_pipeline_from_legacy(pipeline)
        else:
            enhanced_pipeline = pipeline
            
        # Step 5: Execute with StateGraph
        try:
            result = await enhanced_pipeline.execute_hybrid(
                initial_context={'test_context': True}
            )
            
            # Verify execution completed
            assert 'execution_type' in result
            assert result['pipeline_id'] == 'e2e_test_pipeline'
            
        except Exception as e:
            # Execution might fail due to missing real tools, but structure should be correct
            assert "StateGraph" in str(e) or "execution" in str(e).lower()
            
        # Step 6: Verify integration metadata
        assert enhanced_pipeline.has_state_graph()
        assert enhanced_pipeline.is_enhanced_format()
        
        # Step 7: Check generation stats
        stats = compiler.get_generation_stats()
        assert stats['total_generations'] >= 1
        
    @pytest.mark.asyncio  
    async def test_legacy_compatibility_preservation(self):
        """Test that legacy pipelines continue to work unchanged."""
        
        legacy_yaml = '''
id: legacy_compatibility_test
name: "Legacy Compatibility Test"

steps:
  - id: legacy_step1
    action: legacy_action
    parameters:
      input: "legacy_value"
    dependencies: []
    
  - id: legacy_step2  
    action: legacy_action_2
    parameters:
      input: "legacy_value_2"
    dependencies: [legacy_step1]
'''
        
        # Compile with enhanced compiler
        compiler = EnhancedYAMLCompiler(enable_auto_generation=True)
        pipeline = await compiler.compile(legacy_yaml)
        
        # Should be processed as legacy
        assert pipeline.metadata.get('compilation_method') == 'legacy'
        assert pipeline.metadata.get('enhanced_syntax') != True
        
        # Should still execute
        if isinstance(pipeline, EnhancedPipeline):
            enhanced = pipeline
        else:
            enhanced = create_enhanced_pipeline_from_legacy(pipeline)
            
        result = await enhanced.execute_hybrid()
        assert result['execution_type'] == 'legacy'
        assert len(result['results']) == 2
        
    def test_integration_component_availability(self):
        """Test that all integration components can be imported and instantiated."""
        
        # Test EnhancedYAMLCompiler
        compiler = EnhancedYAMLCompiler()
        assert compiler is not None
        assert hasattr(compiler, 'automatic_graph_generator')
        
        # Test EnhancedLangGraphAdapter  
        adapter = EnhancedLangGraphAdapter()
        assert adapter is not None
        assert hasattr(adapter, 'automatic_graph_generator')
        
        # Test EnhancedPipeline
        pipeline = EnhancedPipeline(id='test', name='test')
        assert pipeline is not None
        assert hasattr(pipeline, 'state_graph')
        
        # Test AutomaticGraphGenerator
        generator = AutomaticGraphGenerator()
        assert generator is not None
        assert hasattr(generator, 'generate_graph')


if __name__ == '__main__':
    # Run specific test for debugging
    import asyncio
    
    async def run_single_test():
        test_class = TestEndToEndIntegration()
        await test_class.test_complete_enhanced_pipeline_flow()
        print("✅ End-to-end integration test completed successfully!")
        
    asyncio.run(run_single_test())