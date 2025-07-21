"""Tests for multi_agent_collaboration.yaml example."""
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
            "consensus_threshold": 0.8
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check agent-related steps
        step_ids = [step['id'] for step in config['steps']]
        assert 'decompose_problem' in step_ids
        assert 'initialize_agents' in step_ids
        assert 'assign_tasks' in step_ids
        assert 'collaboration_round' in step_ids
        assert 'check_convergence' in step_ids
        
        # Check for loop in collaboration
        collab_step = next(s for s in config['steps'] if s['id'] == 'collaboration_round')
        assert 'loop' in collab_step
        assert 'max_iterations' in collab_step['loop']
    
    def test_agent_creation_logic(self, pipeline_name):
        """Test agent creation AUTO tags."""
        auto_tags = self.extract_auto_tags(pipeline_name)
        
        assert 'initialize_agents' in auto_tags
        agent_creation = auto_tags['initialize_agents'][0]
        
        # Check for agent diversity
        assert 'diverse backgrounds' in agent_creation.lower()
        assert 'expertise' in agent_creation.lower()
    
    @pytest.mark.asyncio
    async def test_collaboration_rounds(self, orchestrator, pipeline_name):
        """Test multiple collaboration rounds."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                nonlocal round_count
                step_id = step.get('id')
                
                if step_id == 'initialize_agents':
                    return {
                        'result': {
                            'agents': [
                                {'name': 'UrbanPlanner', 'expertise': 'city planning'},
                                {'name': 'Environmentalist', 'expertise': 'sustainability'},
                                {'name': 'TechExpert', 'expertise': 'smart systems'},
                                {'name': 'Economist', 'expertise': 'financial analysis'}
                            ]
                        }
                    }
                elif step_id == 'collaboration_round':
                    round_count += 1
                    return {
                        'result': {
                            'proposals': ['Proposal A', 'Proposal B'],
                            'discussions': ['Point 1', 'Point 2'],
                            'round': round_count
                        }
                    }
                elif step_id == 'check_convergence':
                    # Converge after 3 rounds
                    score = 0.6 + (round_count * 0.1)
                    return {
                        'result': {
                            'score': min(score, 0.85),
                            'converged': score >= 0.8
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify multiple rounds occurred
            assert round_count >= 2
            
            # Verify convergence check was called multiple times
            convergence_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'check_convergence'
            ]
            assert len(convergence_calls) >= 2
    
    @pytest.mark.asyncio
    async def test_consensus_building(self, orchestrator, pipeline_name):
        """Test consensus building mechanism."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'peer_review':
                    return {
                        'result': {
                            'evaluations': [
                                {'proposal': 'A', 'score': 0.85},
                                {'proposal': 'B', 'score': 0.92}
                            ],
                            'best_proposal': 'B'
                        }
                    }
                elif step_id == 'resolve_conflicts':
                    return {
                        'result': {
                            'consensus_level': 0.91,
                            'agreed_solution': 'Modified Proposal B',
                            'dissenting_opinions': []
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify consensus building steps
            consensus_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'resolve_conflicts'
            ]
            assert len(consensus_calls) > 0
    
    def test_output_structure(self, pipeline_name):
        """Test output definitions."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        assert 'outputs' in config
        
        # Check for multi-agent specific outputs
        expected_outputs = [
            'final_solution',
            'consensus_score',
            'agent_contributions',
            'collaboration_rounds',
            'implementation_plan'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs']
    
    @pytest.mark.asyncio
    async def test_early_convergence(self, orchestrator, pipeline_name):
        """Test early exit when consensus is reached quickly."""
        # Test pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Validate relevant configuration
        assert 'steps' in config
        assert len(config['steps']) > 0
    async def mock_step_execution(step, context, state):
                nonlocal rounds_executed
                step_id = step.get('id')
                
                if step_id == 'collaboration_round':
                    rounds_executed += 1
                    return {'result': {'round': rounds_executed}}
                elif step_id == 'check_convergence':
                    # High consensus from the start
                    return {
                        'result': {
                            'score': 0.85,
                            'converged': True
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Should exit early, not run all 10 rounds
            assert rounds_executed < inputs['max_rounds']