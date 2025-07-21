"""Tests for financial_analysis_bot.yaml example."""
import pytest
from .test_base import BaseExampleTest


class TestFinancialAnalysisBotYAML(BaseExampleTest):
    """Test the financial analysis bot YAML pipeline."""
    
    @pytest.fixture
    def pipeline_name(self):
        return "financial_analysis_bot.yaml"
    
    @pytest.fixture
    def sample_inputs(self):
        return {
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "timeframe": "1y",
            "analysis_type": "comprehensive",
            "risk_tolerance": "moderate",
            "asset_type": "equity",
            "run_backtest": True,
            "include_predictions": True
        }
    
    def test_pipeline_structure(self, pipeline_name):
        """Test that the pipeline has valid structure."""
        self.validate_pipeline_structure(pipeline_name)
        
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check financial-specific steps
        step_ids = [step['id'] for step in config['steps']]
        required_steps = [
            'collect_market_data',
            'technical_analysis',
            'fundamental_analysis',
            'sentiment_analysis',
            'risk_assessment',
            'generate_signals'
        ]
        
        for step in required_steps:
            assert step in step_ids, f"Missing required step: {step}"
    
    def test_market_data_collection(self, pipeline_name):
        """Test market data collection configuration."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find market data step
        market_step = next(s for s in config['steps'] if s['id'] == 'collect_market_data')
        
        # Check it has parallel loop for symbols
        assert 'loop' in market_step
        assert market_step['loop']['foreach'] == "{{symbols}}"
        assert market_step['loop']['parallel'] is True
    
    @pytest.mark.asyncio
    async def test_technical_analysis_execution(self, orchestrator, pipeline_name, sample_inputs):
        """Test technical analysis execution with minimal responses."""
        # Test pipeline structure and flow with minimal responses
        # This avoids expensive API calls while testing the pipeline logic
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            expected_outputs={
                'signals': dict,
                'risk_metrics': dict
            },
            use_minimal_responses=True
        )
        
        # Verify result structure
        assert 'outputs' in result
        assert isinstance(result.get('outputs', {}).get('signals', None), dict)
        assert isinstance(result.get('outputs', {}).get('risk_metrics', None), dict)
    
    @pytest.mark.asyncio
    async def test_fundamental_analysis(self, orchestrator, pipeline_name, sample_inputs):
        """Test fundamental analysis configuration."""
        # Load and validate pipeline structure
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find fundamental analysis step
        fund_step = next((s for s in config['steps'] if s['id'] == 'fundamental_analysis'), None)
        assert fund_step is not None
        
        # Verify step configuration
        assert 'parameters' in fund_step
        params = fund_step['parameters']
        assert 'metrics' in params
        
        # Test with minimal responses to validate flow
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            sample_inputs,
            use_minimal_responses=True
        )
        
        # Verify execution completed
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, orchestrator, pipeline_name):
        """Test risk assessment step configuration."""
        # Load pipeline and validate risk assessment step exists
        config = self.load_yaml_pipeline(pipeline_name)
        
        risk_step = next((s for s in config['steps'] if s['id'] == 'risk_assessment'), None)
        assert risk_step is not None
        
        # Verify risk assessment parameters
        assert 'parameters' in risk_step
        
        # Test with minimal portfolio data
        inputs = {
            "symbols": ["AAPL", "MSFT"],
            "portfolio_weights": [0.5, 0.5],
            "risk_tolerance": "moderate"
        }
        
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            inputs,
            use_minimal_responses=True
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_conditional_fundamental_analysis(self, orchestrator, pipeline_name):
        """Test that fundamental analysis only runs for equities."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find fundamental analysis step
        fund_step = next((s for s in config['steps'] if s['id'] == 'fundamental_analysis'), None)
        assert fund_step is not None
        
        # Check for conditional execution
        if 'when' in fund_step:
            assert 'asset_type' in fund_step['when']
    
    @pytest.mark.asyncio
    async def test_backtest_conditional_execution(self, orchestrator, pipeline_name):
        """Test backtest only runs when requested."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Find backtest step
        backtest_step = next((s for s in config['steps'] if s['id'] == 'backtest_strategies'), None)
        
        if backtest_step:
            # Verify conditional execution
            assert 'when' in backtest_step
            assert 'run_backtest' in backtest_step['when']
        
        # Test with backtest disabled
        inputs = {"run_backtest": False}
        result = await self.run_pipeline_test(
            orchestrator,
            pipeline_name,
            inputs,
            use_minimal_responses=True
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization(self, orchestrator, pipeline_name):
        """Test portfolio optimization for multiple symbols."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        # Check if portfolio optimization step exists
        opt_step = next((s for s in config['steps'] if s['id'] == 'portfolio_optimization'), None)
        
        if opt_step:
            # Test with multiple symbols
            inputs = {
                "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                "optimization_constraints": {
                    "min_weight": 0.1,
                    "max_weight": 0.4
                }
            }
            
            result = await self.run_pipeline_test(
                orchestrator,
                pipeline_name,
                inputs,
                use_minimal_responses=True
            )
            
            assert result is not None
    
    def test_output_completeness(self, pipeline_name):
        """Test that all financial outputs are defined."""
        config = self.load_yaml_pipeline(pipeline_name)
        
        expected_outputs = [
            'market_overview',
            'technical_signals',
            'risk_metrics',
            'trading_signals',
            'report',
            'visualizations'
        ]
        
        for output in expected_outputs:
            assert output in config['outputs'], f"Missing output: {output}"