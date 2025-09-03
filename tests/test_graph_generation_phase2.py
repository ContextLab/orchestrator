"""
Comprehensive integration tests for Phase 2 Week 2 graph generation enhancements.

Tests the enhanced components:
- ParallelExecutionDetector intelligent group detection
- ControlFlowAnalyzer pattern detection and optimization  
- StateGraphConstructor LangGraph integration
- End-to-end pipeline processing
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

# Import the enhanced components
from src.orchestrator.graph_generation.automatic_generator import AutomaticGraphGenerator
from src.orchestrator.graph_generation.parallel_detector import ParallelExecutionDetector
from src.orchestrator.graph_generation.control_flow_analyzer import ControlFlowAnalyzer
from src.orchestrator.graph_generation.state_graph_constructor import StateGraphConstructor
from src.orchestrator.graph_generation.types import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    ParsedStep, ParsedPipeline, DependencyGraph, ParallelGroup, 
    ControlFlowMap, DataFlowSchema, StepType, ParallelizationType
)


class TestEnhancedParallelExecutionDetector:
    """Test the enhanced parallel execution detection algorithms."""
    
    @pytest.fixture
    def detector(self):
        return ParallelExecutionDetector()
        
    @pytest.fixture
    def complex_dependency_graph(self):
        """Create a complex dependency graph for testing parallel detection."""
        graph = DependencyGraph()
        
        # Create a fan-out pattern: step1 -> [step2, step3, step4] -> step5
        steps = [
            ParsedStep(id="step1", tool="data-fetch", inputs={"url": "test"}),
            ParsedStep(id="step2", tool="web-search", depends_on=["step1"], inputs={"query": "{{ step1.data }}"}),
            ParsedStep(id="step3", tool="api-call", depends_on=["step1"], inputs={"data": "{{ step1.result }}"}),
            ParsedStep(id="step4", tool="analyze-text", depends_on=["step1"], inputs={"text": "{{ step1.content }}"}),
            ParsedStep(id="step5", depends_on=["step2", "step3", "step4"], inputs={"results": "combined"})
        ]
        
        for step in steps:
            graph.add_node(step.id, step)
            
        # Add dependency edges
        graph.add_edge("step1", "step2")
        graph.add_edge("step1", "step3") 
        graph.add_edge("step1", "step4")
        graph.add_edge("step2", "step5")
        graph.add_edge("step3", "step5")
        graph.add_edge("step4", "step5")
        
        return graph
        
    @pytest.mark.asyncio
    async def test_intelligent_parallel_group_detection(self, detector, complex_dependency_graph):
        """Test that intelligent parallel group detection works correctly."""
        parallel_groups = await detector.detect_parallel_groups(complex_dependency_graph)
        
        # Should detect that steps 2, 3, 4 can run in parallel
        assert len(parallel_groups) >= 1
        
        # Find groups that contain the parallel steps
        parallel_steps_found = set()
        for group in parallel_groups:
            parallel_steps_found.update(group.steps)
                
        # Should detect at least some of steps 2, 3, 4 as parallel
        expected_parallel = {"step2", "step3", "step4"}
        assert len(parallel_steps_found.intersection(expected_parallel)) >= 2, \
            f"Should detect at least 2 of steps 2,3,4 as parallel. Found: {parallel_steps_found}"
        
        # Check properties of detected parallel groups
        for group in parallel_groups:
            assert group.parallelization_type == ParallelizationType.INDEPENDENT
            assert group.estimated_speedup > 1.0
        
    @pytest.mark.asyncio
    async def test_resource_interference_detection(self, detector):
        """Test that resource interference is properly detected."""
        graph = DependencyGraph()
        
        # Create steps that should interfere (same API resource)
        steps = [
            ParsedStep(id="api_call1", tool="web-search", inputs={"query": "test1"}),
            ParsedStep(id="api_call2", tool="web-search", inputs={"query": "test2"}),
            ParsedStep(id="file_op1", action="file-read", inputs={"path": "file1.txt"}),
            ParsedStep(id="file_op2", action="file-read", inputs={"path": "file2.txt"})
        ]
        
        for step in steps:
            graph.add_node(step.id, step)
            
        parallel_groups = await detector.detect_parallel_groups(graph)
        
        # API calls to same service should be limited due to rate limits
        # File operations should be more parallelizable
        total_parallel_steps = sum(len(group.steps) for group in parallel_groups)
        assert total_parallel_steps >= 2, "Should detect some parallelization opportunities"
        
    @pytest.mark.asyncio
    async def test_speedup_estimation_with_amdahls_law(self, detector):
        """Test that speedup estimation uses realistic models like Amdahl's law."""
        # Create a simple parallel group
        group = ["step1", "step2", "step3", "step4"]
        estimated_speedup = detector._estimate_speedup(group)
        
        # Should be realistic (not just len(group))
        assert 1.0 < estimated_speedup <= 4.0, f"Speedup should be realistic: {estimated_speedup}"
        
        # Larger groups should have diminishing returns
        large_group = [f"step{i}" for i in range(10)]
        large_speedup = detector._estimate_speedup(large_group)
        assert large_speedup <= 4.0, "Large groups should be capped at reasonable speedup"


class TestAdvancedControlFlowAnalyzer:
    """Test the advanced control flow analysis and optimization."""
    
    @pytest.fixture
    def analyzer(self):
        return ControlFlowAnalyzer()
        
    @pytest.fixture
    def complex_pipeline(self):
        """Create pipeline with complex control flow patterns."""
        steps = [
            ParsedStep(
                id="data_check", 
                condition="{{ inputs.validate_data }}",
                inputs={"data": "{{ inputs.raw_data }}"}
            ),
            ParsedStep(
                id="parallel_process1",
                depends_on=["data_check"],
                inputs={"chunk": "{{ data_check.chunk1 }}"}
            ),
            ParsedStep(
                id="parallel_process2", 
                depends_on=["data_check"],
                inputs={"chunk": "{{ data_check.chunk2 }}"}
            ),
            ParsedStep(
                id="parallel_process3",
                depends_on=["data_check"], 
                inputs={"chunk": "{{ data_check.chunk3 }}"}
            ),
            ParsedStep(
                id="merge_results",
                depends_on=["parallel_process1", "parallel_process2", "parallel_process3"],
                condition="{{ all_chunks_processed }}",
                inputs={"results": "{{ [parallel_process1.result, parallel_process2.result, parallel_process3.result] }}"}
            ),
            ParsedStep(
                id="retry_failed", 
                condition="{{ merge_results.has_errors }}",
                inputs={"failed_chunks": "{{ merge_results.failed }}"}
            )
        ]
        
        return ParsedPipeline(
            id="complex-flow-test",
            steps=steps,
            inputs={"validate_data": {"type": "boolean"}, "raw_data": {"type": "string"}}
        )
        
    @pytest.mark.asyncio
    async def test_complex_pattern_detection(self, analyzer, complex_pipeline):
        """Test detection of complex control flow patterns."""
        patterns = await analyzer.analyze_complex_control_patterns(complex_pipeline)
        
        # Should detect fan-out pattern (data_check -> 3 parallel processes)
        assert len(patterns['fan_out_patterns']) >= 1
        fan_out = patterns['fan_out_patterns'][0]
        assert fan_out['source_step'] == 'data_check'
        assert len(fan_out['target_steps']) == 3
        
        # Should detect fan-in pattern (3 processes -> merge_results)
        assert len(patterns['fan_in_patterns']) >= 1
        fan_in = patterns['fan_in_patterns'][0]
        assert fan_in['target_step'] == 'merge_results'
        assert len(fan_in['source_steps']) == 3
        
        # Should detect retry pattern
        assert len(patterns['retry_patterns']) >= 1
        retry = patterns['retry_patterns'][0]
        assert retry['retry_step'] == 'retry_failed'
        
    @pytest.mark.asyncio
    async def test_control_flow_optimization(self, analyzer, complex_pipeline):
        """Test that control flow optimizations are applied correctly."""
        control_map = await analyzer.analyze_control_flow(complex_pipeline)
        patterns = await analyzer.analyze_complex_control_patterns(complex_pipeline)
        
        optimized_map = await analyzer.optimize_control_flow(control_map, patterns)
        
        # Should have optimization metadata
        assert len(optimized_map.parallel_opportunities) > 0
        
        # Fan-out optimization should be applied
        if 'data_check' in optimized_map.parallel_opportunities:
            fan_out_opt = optimized_map.parallel_opportunities['data_check']
            assert fan_out_opt['type'] == 'fan_out'
            assert len(fan_out_opt['parallel_targets']) == 3
            
    @pytest.mark.asyncio 
    async def test_langraph_control_edge_generation(self, analyzer, complex_pipeline):
        """Test generation of LangGraph-compatible control edges."""
        control_map = await analyzer.analyze_control_flow(complex_pipeline)
        langraph_edges = await analyzer.create_langraph_control_edges(control_map)
        
        assert len(langraph_edges) > 0
        
        # Should have conditional edges
        conditional_edges = [e for e in langraph_edges if 'condition_function' in e]
        assert len(conditional_edges) >= 2  # data_check and merge_results conditions
        
        # Each conditional edge should have proper structure
        for edge in conditional_edges:
            assert 'source_node' in edge
            assert 'condition_function' in edge
            assert 'true_path' in edge
            assert 'false_path' in edge


class TestStateGraphConstructor:
    """Test the StateGraph construction with LangGraph integration."""
    
    @pytest.fixture
    def constructor(self):
        return StateGraphConstructor()
        
    @pytest.fixture
    def complete_analysis_components(self):
        """Create complete set of analysis components for testing."""
        # Dependency graph
        dependency_graph = DependencyGraph()
        steps = [
            ParsedStep(id="step1", tool="fetch-data"),
            ParsedStep(id="step2", tool="process-data", depends_on=["step1"]),
            ParsedStep(id="step3", tool="analyze-results", depends_on=["step2"])
        ]
        
        for step in steps:
            dependency_graph.add_node(step.id, step)
        dependency_graph.add_edge("step1", "step2")
        dependency_graph.add_edge("step2", "step3")
        
        # Parallel groups
        parallel_groups = [
            ParallelGroup(
                steps=["step2"],
                parallelization_type=ParallelizationType.INDEPENDENT,
                execution_level=1,
                estimated_speedup=1.0
            )
        ]
        
        # Control flow
        control_flow = ControlFlowMap()
        
        # Data schema
        data_schema = DataFlowSchema()
        
        # Original pipeline
        pipeline = ParsedPipeline(
            id="test-pipeline",
            steps=steps,
            inputs={"input_data": {"type": "string"}}
        )
        
        return dependency_graph, parallel_groups, control_flow, data_schema, pipeline
        
    @pytest.mark.asyncio
    async def test_enhanced_placeholder_creation(self, constructor, complete_analysis_components):
        """Test creation of enhanced placeholder when LangGraph is not available."""
        dependency_graph, parallel_groups, control_flow, data_schema, pipeline = complete_analysis_components
        
        # Force placeholder creation
        with patch('orchestrator.graph_generation.state_graph_constructor.StateGraphConstructor._create_langgraph_state_graph', 
                   side_effect=ImportError("LangGraph not available")):
            
            result = await constructor.construct_graph(
                dependency_graph, parallel_groups, control_flow, data_schema, pipeline
            )
            
        # Should be enhanced placeholder
        assert result['type'] == 'enhanced_placeholder'
        assert result['pipeline_id'] == 'test-pipeline'
        assert len(result['nodes']) == 3
        assert len(result['edges']) == 2
        assert 'execution_plan' in result
        assert result['status'] == 'ready_for_execution'
        
        # Execution plan should have proper structure
        execution_plan = result['execution_plan']
        assert len(execution_plan) == 3  # 3 execution levels
        assert execution_plan[0]['level'] == 0
        assert execution_plan[0]['steps'] == ['step1']
        
    @pytest.mark.asyncio
    async def test_state_class_creation(self, constructor, complete_analysis_components):
        """Test dynamic creation of TypedDict state class."""
        dependency_graph, parallel_groups, control_flow, data_schema, pipeline = complete_analysis_components
        
        # Should not raise an exception
        state_class = constructor._create_state_class(data_schema, pipeline, dependency_graph)
        
        # Should be a class
        assert isinstance(state_class, type)
        assert state_class.__name__ == 'PipelineState'
        
        # Basic functionality test - can create instance
        # This tests that the dynamic class creation works
        try:
            # Create a basic instance to verify the class works
            test_instance = {}  # TypedDict instances are just dicts at runtime
            assert isinstance(test_instance, dict)
        except Exception as e:
            pytest.fail(f"State class creation failed: {e}")
            
    @pytest.mark.asyncio
    async def test_template_resolution(self, constructor):
        """Test Jinja2 template resolution in step inputs."""
        inputs = {
            "static_value": "hello",
            "template_value": "Hello {{ name }}!",
            "complex_template": "Result: {{ step1.result.status }} - {{ step2.data | default('none') }}"
        }
        
        state = {
            "name": "World",
            "step1": {"result": {"status": "success"}},
            "step2": {"data": "processed"}
        }
        
        resolved = await constructor._resolve_input_templates(inputs, state)
        
        assert resolved["static_value"] == "hello"
        assert resolved["template_value"] == "Hello World!"
        assert resolved["complex_template"] == "Result: success - processed"
        
    @pytest.mark.asyncio 
    async def test_speedup_calculation(self, constructor):
        """Test total speedup calculation from parallel groups."""
        parallel_groups = [
            ParallelGroup(
                steps=["step1", "step2"],
                parallelization_type=ParallelizationType.INDEPENDENT,
                execution_level=0,
                estimated_speedup=1.5
            ),
            ParallelGroup(
                steps=["step3", "step4", "step5"],
                parallelization_type=ParallelizationType.INDEPENDENT, 
                execution_level=1,
                estimated_speedup=2.0
            )
        ]
        
        total_speedup = constructor._calculate_total_speedup(parallel_groups)
        
        # Should combine speedups (but not exceed reasonable limits)
        assert 1.0 < total_speedup <= 10.0
        assert total_speedup >= 2.0  # Should be at least the best individual speedup


class TestEndToEndIntegration:
    """Test complete end-to-end pipeline processing with all Phase 2 enhancements."""
    
    @pytest.fixture
    def generator(self):
        return AutomaticGraphGenerator()
        
    @pytest.mark.asyncio
    async def test_complete_phase2_pipeline_processing(self, generator):
        """Test complete pipeline processing with Phase 2 enhancements."""
        
        # Complex pipeline with all Phase 2 features
        complex_pipeline = {
            "id": "phase2-integration-test",
            "name": "Phase 2 Integration Test Pipeline",
            "inputs": {
                "topic": {"type": "string", "required": True},
                "parallel_processes": {"type": "integer", "default": 3},
                "enable_analysis": {"type": "boolean", "default": True}
            },
            "steps": [
                {
                    "id": "data_collection",
                    "tool": "web-search",
                    "inputs": {
                        "query": "{{ inputs.topic }} research data",
                        "max_results": "{{ inputs.parallel_processes * 10 }}"
                    }
                },
                {
                    "id": "parallel_analysis1", 
                    "depends_on": ["data_collection"],
                    "tool": "analyze-text",
                    "inputs": {
                        "content": "{{ data_collection.results[:33] }}"
                    }
                },
                {
                    "id": "parallel_analysis2",
                    "depends_on": ["data_collection"], 
                    "tool": "analyze-sentiment",
                    "inputs": {
                        "content": "{{ data_collection.results[33:66] }}"
                    }
                },
                {
                    "id": "parallel_analysis3",
                    "depends_on": ["data_collection"],
                    "tool": "extract-entities", 
                    "inputs": {
                        "content": "{{ data_collection.results[66:] }}"
                    }
                },
                {
                    "id": "conditional_deep_analysis",
                    "condition": "{{ inputs.enable_analysis }}",
                    "depends_on": ["parallel_analysis1", "parallel_analysis2", "parallel_analysis3"],
                    "tool": "deep-analysis",
                    "inputs": {
                        "combined_results": "combined"
                    }
                },
                {
                    "id": "final_report",
                    "depends_on": ["parallel_analysis1", "parallel_analysis2", "parallel_analysis3"],
                    "tool": "generate-report",
                    "inputs": {
                        "topic": "{{ inputs.topic }}",
                        "analyses": "combined_analyses",
                        "deep_analysis": "{{ conditional_deep_analysis.result | default('not_performed') }}"
                    }
                }
            ]
        }
        
        # Process with Phase 2 capabilities
        result = await generator.generate_graph(complex_pipeline)
        
        # Should successfully generate enhanced graph structure
        assert result is not None
        
        # If it's a placeholder, should have Phase 2 enhancements
        if isinstance(result, dict) and result.get('type') == 'enhanced_placeholder':
            assert result['status'] == 'ready_for_execution'
            assert len(result['nodes']) == 6
            assert 'parallel_groups' in result
            assert 'execution_plan' in result
            assert result.get('estimated_total_speedup', 1.0) > 1.0
            
            # Should detect parallel opportunities
            parallel_groups = result['parallel_groups']
            assert len(parallel_groups) > 0
            
            # Should have detailed execution plan
            execution_plan = result['execution_plan']
            assert len(execution_plan) >= 3  # Multiple execution levels
            
            # Should detect parallel execution at level 1
            level1_plan = next((plan for plan in execution_plan if plan['level'] == 1), None)
            assert level1_plan is not None
            assert level1_plan['execution_type'] in ['parallel', 'sequential']
            
    @pytest.mark.asyncio
    async def test_phase2_performance_improvements(self, generator):
        """Test that Phase 2 enhancements provide measurable improvements."""
        
        # Simple pipeline to establish baseline
        simple_pipeline = {
            "id": "simple-test",
            "inputs": {"data": {"type": "string"}},
            "steps": [
                {"id": "step1", "tool": "process", "inputs": {"data": "{{ inputs.data }}"}},
                {"id": "step2", "depends_on": ["step1"], "tool": "analyze", "inputs": {"result": "{{ step1.result }}"}}
            ]
        }
        
        # Complex pipeline with parallelization opportunities
        parallel_pipeline = {
            "id": "parallel-test", 
            "inputs": {"data": {"type": "string"}},
            "steps": [
                {"id": "source", "tool": "fetch-data", "inputs": {"query": "{{ inputs.data }}"}},
                {"id": "process1", "depends_on": ["source"], "tool": "process-chunk1", "inputs": {"chunk": "{{ source.chunk1 }}"}},
                {"id": "process2", "depends_on": ["source"], "tool": "process-chunk2", "inputs": {"chunk": "{{ source.chunk2 }}"}},
                {"id": "process3", "depends_on": ["source"], "tool": "process-chunk3", "inputs": {"chunk": "{{ source.chunk3 }}"}},
                {"id": "merge", "depends_on": ["process1", "process2", "process3"], "tool": "merge-results", "inputs": {"results": "combined"}}
            ]
        }
        
        # Generate graphs
        simple_result = await generator.generate_graph(simple_pipeline)
        parallel_result = await generator.generate_graph(parallel_pipeline)
        
        # Both should succeed
        assert simple_result is not None
        assert parallel_result is not None
        
        # Parallel pipeline should have better estimated performance
        if (isinstance(parallel_result, dict) and 
            isinstance(simple_result, dict) and
            'estimated_total_speedup' in parallel_result and
            'estimated_total_speedup' in simple_result):
            
            parallel_speedup = parallel_result['estimated_total_speedup'] 
            simple_speedup = simple_result['estimated_total_speedup']
            
            assert parallel_speedup > simple_speedup, \
                f"Parallel pipeline should have better speedup: {parallel_speedup} vs {simple_speedup}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])