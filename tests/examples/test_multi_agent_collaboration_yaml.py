"""Tests for multi_agent_collaboration.yaml example.

This test file follows the NO MOCKS policy. Tests use real orchestration
when API keys are available, otherwise they skip gracefully.
"""
import pytest
from .test_base import BaseExampleTest


class TestMultiAgentCollaborationYAML(BaseExampleTest):
    """Test the multi-agent collaboration YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "multi_agent_collaboration.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "problem": "Design a sustainable smart city transportation system",
            "num_agents": 4,
            "max_rounds": 5,
            "agent_roles": "auto",
            "consensus_threshold": 0.8
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check agent-related steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'decompose_problem',
            'initialize_agents',
            'assign_tasks',
            'collaboration_round',
            'peer_review',
            'integrate_solutions',
            'final_review',
            'generate_report'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_agent_creation_logic(self, pipeline_name):
        """Test agent initialization step configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find initialize_agents step
        init_step = next(s for s in config['steps'] if s['id'] == 'initialize_agents')
        assert init_step is not None
        
        # Check agent roles mentioned
        assert 'Researcher agents' in init_step['action']
        assert 'Analyst agents' in init_step['action']
        assert 'Creative agents' in init_step['action']
        assert 'Critic agents' in init_step['action']
        assert 'Synthesizer agent' in init_step['action']
    
    @pytest.mark.asyncio
    async def test_collaboration_execution(self, orchestrator, pipeline_name, sample_inputs):
        """Test collaboration pipeline execution."""
        # Test pipeline execution with minimal responses
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'solution': str,
                'convergence_score': (str, float),
                'agent_contributions': (str, dict, list)
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert result is not None
        assert 'outputs' in result or 'steps' in result
    
    def test_consensus_building(self, pipeline_name):
        """Test consensus building mechanism configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check peer review step
        peer_review = next(s for s in config['steps'] if s['id'] == 'peer_review')
        assert peer_review is not None
        assert 'condition' in peer_review
        assert '{{consensus_threshold}}' in peer_review['condition']
        
        # Check conflict resolution step
        resolve_step = next(s for s in config['steps'] if s['id'] == 'resolve_conflicts')
        assert resolve_step is not None
        assert 'condition' in resolve_step
        assert 'conflicts' in resolve_step['condition']
    
    def test_output_structure(self, pipeline_name):
        """Test output definitions."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        assert 'outputs' in config
        
        # Check for multi-agent specific outputs (matching actual YAML)
        expected_outputs = [
            'solution',
            'convergence_achieved',
            'rounds_executed',
            'convergence_score',
            'agent_contributions',
            'emergent_patterns',
            'quality_score',
            'confidence_level'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"
    
    def test_emergent_behavior_analysis(self, pipeline_name):
        """Test emergent behavior analysis configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check analyze_emergence step
        emergence_step = next(s for s in config['steps'] if s['id'] == 'analyze_emergence')
        assert emergence_step is not None
        assert 'Communication patterns' in emergence_step['action']
        assert 'Collective intelligence metrics' in emergence_step['action']
        
        # Check agent learning step
        learning_step = next(s for s in config['steps'] if s['id'] == 'agent_learning')
        assert learning_step is not None
        assert 'successful collaboration patterns' in learning_step['action']
    
    def test_problem_decomposition(self, pipeline_name):
        """Test problem decomposition configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check decompose_problem step
        decompose_step = next(s for s in config['steps'] if s['id'] == 'decompose_problem')
        assert decompose_step is not None
        assert '{{problem}}' in decompose_step['action']
        assert 'dependency graph' in decompose_step['action']
        assert 'cache_results' in decompose_step
    
    def test_solution_integration(self, pipeline_name):
        """Test solution integration process."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check integrate_solutions step
        integrate_step = next(s for s in config['steps'] if s['id'] == 'integrate_solutions')
        assert integrate_step is not None
        assert 'synthesizer agent' in integrate_step['action']
        assert integrate_step['timeout'] == 60.0
    
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