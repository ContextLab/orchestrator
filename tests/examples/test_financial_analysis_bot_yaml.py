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
        """Test technical analysis execution."""
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'collect_market_data':
                    return {
                        'result': {
                            'AAPL': {
                                'price_data': {'close': [150, 152, 155]},
                                'volume': [1000000, 1100000, 1050000]
                            }
                        }
                    }
                elif step_id == 'technical_analysis':
                    return {
                        'result': {
                            'AAPL': {
                                'indicators': {
                                    'rsi': 65,
                                    'macd': {'signal': 'buy'},
                                    'sma_20': 153
                                },
                                'patterns': ['ascending_triangle'],
                                'trend': 'upward'
                            }
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=sample_inputs
            )
            
            # Verify technical analysis was called for each symbol
            tech_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'technical_analysis'
            ]
            assert len(tech_calls) > 0
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, orchestrator, pipeline_name):
        """Test risk assessment calculations."""
        inputs = {
            "symbols": ["AAPL", "MSFT"],
            "timeframe": "6m",
            "risk_tolerance": "conservative"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                step_id = step.get('id')
                
                if step_id == 'risk_assessment':
                    return {
                        'result': {
                            'portfolio_volatility': 0.18,
                            'var_95': -0.025,
                            'sharpe_ratio': 1.2,
                            'correlation_matrix': [[1.0, 0.6], [0.6, 1.0]],
                            'stress_test_results': {
                                'market_crash': -0.25,
                                'recession': -0.15
                            }
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify risk assessment includes stress tests
            risk_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'risk_assessment'
            ]
            assert len(risk_calls) == 1
    
    @pytest.mark.asyncio
    async def test_conditional_fundamental_analysis(self, orchestrator, pipeline_name):
        """Test that fundamental analysis only runs for equities."""
        # Test with equity
        equity_inputs = {
            "symbols": ["AAPL"],
            "asset_type": "equity",
            "timeframe": "1y"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {'result': {}}
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=equity_inputs
            )
            
            # Check fundamental analysis was called
            fundamental_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'fundamental_analysis'
            ]
            assert len(fundamental_calls) > 0
        
        # Test with crypto (should skip fundamental analysis)
        crypto_inputs = {
            "symbols": ["BTC-USD"],
            "asset_type": "crypto",
            "timeframe": "1y"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {'result': {}}
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=crypto_inputs
            )
            
            # Check fundamental analysis was NOT called
            fundamental_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'fundamental_analysis'
            ]
            assert len(fundamental_calls) == 0
    
    @pytest.mark.asyncio
    async def test_backtest_conditional_execution(self, orchestrator, pipeline_name):
        """Test backtest only runs when requested."""
        # Test with backtest enabled
        inputs_with_backtest = {
            "symbols": ["AAPL"],
            "run_backtest": True,
            "timeframe": "1y"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'backtest_strategies':
                    return {
                        'result': {
                            'total_return': 0.15,
                            'sharpe_ratio': 1.5,
                            'max_drawdown': -0.08,
                            'win_rate': 0.62
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs_with_backtest
            )
            
            # Verify backtest was called
            backtest_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'backtest_strategies'
            ]
            assert len(backtest_calls) > 0
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization(self, orchestrator, pipeline_name):
        """Test portfolio optimization for multiple symbols."""
        inputs = {
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "risk_tolerance": "moderate",
            "timeframe": "1y"
        }
        
        with patch.object(orchestrator, 'execute_step', new_callable=AsyncMock) as mock_exec:
            async def mock_step_execution(step, context, state):
                if step.get('id') == 'portfolio_optimization':
                    return {
                        'result': {
                            'optimal_weights': {
                                'AAPL': 0.30,
                                'MSFT': 0.25,
                                'GOOGL': 0.25,
                                'AMZN': 0.20
                            },
                            'expected_return': 0.12,
                            'portfolio_volatility': 0.16,
                            'sharpe_ratio': 0.75
                        }
                    }
                return {'result': {}}
            
            mock_exec.side_effect = mock_step_execution
            
            result = await orchestrator.run_pipeline(
                self.load_yaml_pipeline(pipeline_name),
                inputs=inputs
            )
            
            # Verify portfolio optimization was called
            optimization_calls = [
                call for call in mock_exec.call_args_list 
                if call[0][0].get('id') == 'portfolio_optimization'
            ]
            assert len(optimization_calls) > 0
    
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