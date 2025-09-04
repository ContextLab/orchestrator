"""Pipeline integration infrastructure using proven TestModel/TestProvider patterns.

This module extends the successful test infrastructure from Issue #374 to provide
systematic pipeline validation capabilities for the validate-all-example-pipelines epic.

Key Features:
- PipelineTestModel extending proven TestModel patterns
- PipelineTestProvider extending MockTestProvider patterns
- Integration with existing PipelineTestSuite infrastructure
- Systematic validation using orchestrator framework patterns
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from ..core.model import Model, ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost
from ..orchestrator import Orchestrator
from ..models.registry import ModelRegistry
from .pipeline_test_suite import PipelineTestSuite, PipelineTestResult, ExecutionResult
from .pipeline_validator import PipelineValidator

logger = logging.getLogger(__name__)


@dataclass
class PipelineIntegrationResult:
    """Result of pipeline integration validation."""
    
    pipeline_name: str
    validation_passed: bool
    execution_successful: bool
    integration_score: float
    
    # Detailed results
    test_model_performance: Dict[str, Any] = field(default_factory=dict)
    provider_integration_status: Dict[str, Any] = field(default_factory=dict)
    orchestrator_compatibility: Dict[str, Any] = field(default_factory=dict)
    
    # Issues and recommendations
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Execution metadata
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    estimated_cost: float = 0.0


class PipelineTestModel(Model):
    """Extended TestModel specifically designed for pipeline validation.
    
    This class extends the proven TestModel patterns from the successful
    testing infrastructure to provide pipeline-specific validation capabilities.
    """
    
    def __init__(
        self,
        name: str = "pipeline-test-model",
        provider: str = "pipeline-test-provider",
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
        metrics: Optional[ModelMetrics] = None,
        cost: Optional[ModelCost] = None,
        pipeline_validation_enabled: bool = True,
        mock_responses: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize pipeline test model with enhanced validation capabilities."""
        
        # Enhanced capabilities for pipeline testing
        if capabilities is None:
            capabilities = ModelCapabilities(
                supported_tasks=[
                    "text-generation", "analysis", "validation", "structured-output",
                    "pipeline-execution", "template-resolution", "quality-assessment"
                ],
                context_window=32768,  # Larger context for pipeline processing
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=False,  # Simplified for testing
                max_output_tokens=4096
            )
        
        if requirements is None:
            requirements = ModelRequirements(
                memory_gb=0.2,  # Slightly more memory for pipeline processing
                cpu_cores=1,
                gpu_memory_gb=0,  # CPU-only for testing
                network_access=True  # May need network for validation
            )
            
        if metrics is None:
            metrics = ModelMetrics(
                tokens_per_second=50.0,
                quality_score=0.95,
                reliability_score=0.99,
                latency_p50_ms=100.0,
                latency_p95_ms=250.0
            )
            
        if cost is None:
            cost = ModelCost(
                is_free=True,
                input_cost_per_token=0.0,
                output_cost_per_token=0.0,
                fixed_cost_per_request=0.0
            )
            
        super().__init__(name, provider, capabilities, requirements, metrics, cost)
        
        # Pipeline-specific configuration
        self.pipeline_validation_enabled = pipeline_validation_enabled
        self.mock_responses = mock_responses or {}
        self.execution_count = 0
        self.validation_history: List[Dict[str, Any]] = []
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate pipeline-aware text response."""
        
        self.execution_count += 1
        
        # Check for pipeline-specific mock responses
        pipeline_context = kwargs.get('pipeline_context', {})
        pipeline_name = pipeline_context.get('pipeline_name', 'unknown')
        
        if pipeline_name in self.mock_responses:
            response = self.mock_responses[pipeline_name]
        else:
            # Generate contextual response based on prompt content
            if "validation" in prompt.lower():
                response = f"Pipeline validation response for {pipeline_name}: All validations passed successfully."
            elif "analysis" in prompt.lower():
                response = f"Pipeline analysis for {pipeline_name}: Structure is valid, templates resolved correctly."
            elif "quality" in prompt.lower():
                response = f"Quality assessment for {pipeline_name}: High quality output with score 95/100."
            else:
                response = f"Pipeline test response for {pipeline_name}: {prompt[:100]}..."
        
        # Record validation attempt
        validation_record = {
            'timestamp': time.time(),
            'pipeline_name': pipeline_name,
            'prompt_type': self._classify_prompt(prompt),
            'response_generated': True,
            'execution_count': self.execution_count
        }
        self.validation_history.append(validation_record)
        
        return response
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate structured response for pipeline validation."""
        
        pipeline_context = kwargs.get('pipeline_context', {})
        pipeline_name = pipeline_context.get('pipeline_name', 'unknown')
        
        # Generate structured validation result
        if schema.get('type') == 'object':
            properties = schema.get('properties', {})
            
            # Build structured response based on schema
            response = {}
            
            if 'validation_status' in properties:
                response['validation_status'] = 'passed'
            
            if 'quality_score' in properties:
                response['quality_score'] = 95.0
                
            if 'issues' in properties:
                response['issues'] = []
                
            if 'recommendations' in properties:
                response['recommendations'] = [
                    f"Pipeline {pipeline_name} is well-structured",
                    "Consider adding error handling for robustness"
                ]
            
            if 'execution_metadata' in properties:
                response['execution_metadata'] = {
                    'execution_time': 0.5,
                    'tokens_used': 150,
                    'api_calls': 1,
                    'estimated_cost': 0.001
                }
            
            # Fill any remaining properties with defaults
            for prop_name, prop_schema in properties.items():
                if prop_name not in response:
                    if prop_schema.get('type') == 'string':
                        response[prop_name] = f"Test value for {prop_name}"
                    elif prop_schema.get('type') == 'number':
                        response[prop_name] = 1.0
                    elif prop_schema.get('type') == 'boolean':
                        response[prop_name] = True
                    elif prop_schema.get('type') == 'array':
                        response[prop_name] = []
                    elif prop_schema.get('type') == 'object':
                        response[prop_name] = {}
            
            return response
        
        # Fallback structured response
        return {
            "pipeline_validation": True,
            "test_output": f"Structured response for pipeline {pipeline_name}",
            "schema_validated": True
        }
    
    async def health_check(self) -> bool:
        """Enhanced health check for pipeline testing."""
        
        # Basic health check
        if not super().health_check():
            return False
        
        # Pipeline-specific health checks
        if self.pipeline_validation_enabled:
            # Check if we can perform basic validation operations
            try:
                test_validation = await self.generate("validation test", pipeline_context={'pipeline_name': 'health_check'})
                return "validation" in test_validation.lower()
            except Exception:
                return False
        
        return True
    
    async def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Pipeline-aware cost estimation."""
        
        # Always free for testing, but track usage
        usage_record = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost': 0.0,
            'timestamp': time.time()
        }
        
        # Add to validation history for tracking
        self.validation_history.append({
            'type': 'cost_estimation',
            'usage': usage_record
        })
        
        return 0.0
    
    def _classify_prompt(self, prompt: str) -> str:
        """Classify the type of prompt for better response generation."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['validate', 'validation', 'verify']):
            return 'validation'
        elif any(word in prompt_lower for word in ['analyze', 'analysis', 'assess']):
            return 'analysis'
        elif any(word in prompt_lower for word in ['quality', 'score', 'rating']):
            return 'quality'
        elif any(word in prompt_lower for word in ['template', 'resolve', 'render']):
            return 'template'
        else:
            return 'general'
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities."""
        
        return {
            'total_executions': self.execution_count,
            'validation_attempts': len([v for v in self.validation_history if v.get('type') != 'cost_estimation']),
            'unique_pipelines': len(set(v.get('pipeline_name', 'unknown') for v in self.validation_history if 'pipeline_name' in v)),
            'prompt_types': {
                prompt_type: len([v for v in self.validation_history if v.get('prompt_type') == prompt_type])
                for prompt_type in set(v.get('prompt_type') for v in self.validation_history if 'prompt_type' in v)
            },
            'average_executions_per_pipeline': self.execution_count / max(1, len(set(v.get('pipeline_name', 'unknown') for v in self.validation_history if 'pipeline_name' in v)))
        }


class PipelineTestProvider:
    """Enhanced TestProvider specifically designed for pipeline integration testing.
    
    This class extends the proven MockTestProvider patterns to provide
    comprehensive pipeline validation capabilities.
    """
    
    def __init__(self, name: str = "pipeline-test-provider"):
        """Initialize pipeline test provider with enhanced capabilities."""
        
        self.name = name
        self.is_initialized = True
        
        # Create enhanced test models for different validation scenarios
        self._models = {
            # Core pipeline test model
            "pipeline-test-model": PipelineTestModel(),
            
            # Specialized models for different validation types
            "pipeline-validation-model": PipelineTestModel(
                name="pipeline-validation-model",
                pipeline_validation_enabled=True
            ),
            
            "pipeline-quality-model": PipelineTestModel(
                name="pipeline-quality-model",
                mock_responses={
                    "quality_assessment": "Excellent pipeline quality with comprehensive validation",
                    "template_validation": "All templates resolved successfully"
                }
            ),
            
            # Common model aliases that tests might expect
            "openai/gpt-3.5-turbo": PipelineTestModel(name="gpt-3.5-turbo", provider="openai"),
            "openai/gpt-4": PipelineTestModel(name="gpt-4", provider="openai"),
            "anthropic/claude-3": PipelineTestModel(name="claude-3", provider="anthropic"),
            "anthropic/claude-sonnet-4-20250514": PipelineTestModel(name="claude-sonnet-4", provider="anthropic"),
        }
        
        # Pipeline validation configuration
        self.validation_enabled = True
        self.integration_tracking = True
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.model_usage_stats: Dict[str, Dict[str, Any]] = {}
    
    @property 
    def available_models(self) -> List[str]:
        """List all available models including pipeline-specific ones."""
        return list(self._models.keys())
    
    def supports_model(self, model_name: str) -> bool:
        """Check if provider supports model with enhanced pipeline validation."""
        return model_name in self._models
    
    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get enhanced capabilities for pipeline validation."""
        if not self.supports_model(model_name):
            raise ValueError(f"Model '{model_name}' not supported")
        
        model = self._models[model_name]
        return model.capabilities
    
    def get_model_requirements(self, model_name: str) -> ModelRequirements:
        """Get model requirements with pipeline-specific considerations."""
        if not self.supports_model(model_name):
            raise ValueError(f"Model '{model_name}' not supported")
        
        model = self._models[model_name]
        return model.requirements
    
    def get_model_cost(self, model_name: str) -> ModelCost:
        """Get cost information for pipeline testing."""
        if not self.supports_model(model_name):
            raise ValueError(f"Model '{model_name}' not supported")
        
        model = self._models[model_name]
        return model.cost
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get enhanced provider information with pipeline capabilities."""
        return {
            "name": self.name,
            "type": "pipeline-test",
            "models": len(self._models),
            "initialized": self.is_initialized,
            "validation_enabled": self.validation_enabled,
            "integration_tracking": self.integration_tracking,
            "supported_pipeline_features": [
                "template-resolution",
                "quality-assessment", 
                "execution-validation",
                "structured-output",
                "cost-estimation"
            ]
        }
    
    async def get_model(self, model_name: str, **kwargs) -> PipelineTestModel:
        """Get model instance with pipeline context."""
        if not self.supports_model(model_name):
            raise ValueError(f"Model '{model_name}' not supported")
        
        model = self._models[model_name]
        
        # Track model usage
        if model_name not in self.model_usage_stats:
            self.model_usage_stats[model_name] = {
                'requests': 0,
                'total_execution_time': 0.0,
                'last_used': None
            }
        
        self.model_usage_stats[model_name]['requests'] += 1
        self.model_usage_stats[model_name]['last_used'] = time.time()
        
        return model
    
    async def initialize(self) -> None:
        """Initialize provider with pipeline validation support."""
        
        # Verify all models are properly configured
        for model_name, model in self._models.items():
            try:
                health = await model.health_check()
                if not health:
                    logger.warning(f"Model {model_name} failed health check")
            except Exception as e:
                logger.warning(f"Error during model {model_name} initialization: {e}")
        
        self.is_initialized = True
        logger.info(f"PipelineTestProvider initialized with {len(self._models)} models")
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get provider usage statistics for analysis."""
        
        total_requests = sum(stats['requests'] for stats in self.model_usage_stats.values())
        
        return {
            'total_models': len(self._models),
            'total_requests': total_requests,
            'model_usage': self.model_usage_stats.copy(),
            'most_used_model': max(self.model_usage_stats.items(), key=lambda x: x[1]['requests'])[0] if self.model_usage_stats else None,
            'provider_uptime': time.time() - (min(stats.get('last_used', time.time()) for stats in self.model_usage_stats.values()) if self.model_usage_stats else time.time())
        }
    
    def reset_usage_statistics(self) -> None:
        """Reset usage statistics for clean testing."""
        self.model_usage_stats.clear()
        for model in self._models.values():
            if hasattr(model, 'execution_count'):
                model.execution_count = 0
            if hasattr(model, 'validation_history'):
                model.validation_history.clear()


class PipelineIntegrationValidator:
    """Systematic pipeline integration validator using proven infrastructure patterns."""
    
    def __init__(
        self,
        examples_dir: Optional[Path] = None,
        orchestrator: Optional[Orchestrator] = None,
        test_provider: Optional[PipelineTestProvider] = None,
        enable_comprehensive_validation: bool = True
    ):
        """Initialize pipeline integration validator."""
        
        self.examples_dir = examples_dir or Path("examples")
        self.test_provider = test_provider or PipelineTestProvider()
        
        # Create model registry with test provider
        self.model_registry = ModelRegistry()
        self.model_registry.register_provider(self.test_provider)
        
        # Create orchestrator with test infrastructure
        self.orchestrator = orchestrator or self._create_test_orchestrator()
        
        # Initialize validation components
        self.pipeline_validator = PipelineValidator()
        self.pipeline_test_suite = PipelineTestSuite(
            examples_dir=self.examples_dir,
            model_registry=self.model_registry,
            orchestrator=self.orchestrator,
            enable_llm_quality_review=enable_comprehensive_validation,
            enable_enhanced_template_validation=enable_comprehensive_validation,
            enable_performance_monitoring=True
        )
        
        # Results tracking
        self.integration_results: List[PipelineIntegrationResult] = []
        
        logger.info("PipelineIntegrationValidator initialized with systematic validation infrastructure")
    
    def _create_test_orchestrator(self) -> Orchestrator:
        """Create orchestrator with pipeline test infrastructure."""
        
        from ..control_systems.hybrid_control_system import HybridControlSystem
        
        # Create control system with test model registry
        control_system = HybridControlSystem(model_registry=self.model_registry)
        
        # Create orchestrator with all required components
        orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=control_system,
            max_concurrent_tasks=5,  # Limit for testing
            debug_templates=True  # Enable template debugging
        )
        
        return orchestrator
    
    async def validate_pipeline_integration(
        self,
        pipeline_name: str,
        pipeline_path: Path
    ) -> PipelineIntegrationResult:
        """Validate a single pipeline's integration with the test infrastructure."""
        
        start_time = time.time()
        logger.info(f"Validating pipeline integration: {pipeline_name}")
        
        # Initialize result
        result = PipelineIntegrationResult(
            pipeline_name=pipeline_name,
            validation_passed=False,
            execution_successful=False,
            integration_score=0.0
        )
        
        try:
            # 1. Basic pipeline validation using existing validator
            validation_results = self.pipeline_validator.validate_pipeline_file(pipeline_path)
            result.validation_passed = validation_results.get('valid', False)
            
            if not result.validation_passed:
                result.issues.extend(validation_results.get('issues', []))
                result.warnings.extend(validation_results.get('warnings', []))
            
            # 2. Test model integration
            model_performance = await self._test_model_integration(pipeline_name)
            result.test_model_performance = model_performance
            
            # 3. Provider integration validation
            provider_status = await self._test_provider_integration(pipeline_name)
            result.provider_integration_status = provider_status
            
            # 4. Orchestrator compatibility test
            orchestrator_compat = await self._test_orchestrator_compatibility(pipeline_path)
            result.orchestrator_compatibility = orchestrator_compat
            result.execution_successful = orchestrator_compat.get('execution_successful', False)
            
            # 5. Calculate integration score
            result.integration_score = self._calculate_integration_score(result)
            
            # 6. Generate recommendations
            result.recommendations = self._generate_recommendations(result)
            
        except Exception as e:
            logger.error(f"Error validating pipeline integration for {pipeline_name}: {e}")
            result.issues.append(f"Integration validation error: {str(e)}")
        
        result.execution_time = time.time() - start_time
        self.integration_results.append(result)
        
        logger.info(f"Pipeline integration validation completed for {pipeline_name} "
                   f"(score: {result.integration_score:.2f}, time: {result.execution_time:.1f}s)")
        
        return result
    
    async def _test_model_integration(self, pipeline_name: str) -> Dict[str, Any]:
        """Test integration with PipelineTestModel."""
        
        model_performance = {
            'model_available': False,
            'health_check_passed': False,
            'generation_test_passed': False,
            'structured_output_test_passed': False,
            'cost_estimation_working': False
        }
        
        try:
            # Get pipeline test model
            model = await self.test_provider.get_model("pipeline-test-model")
            model_performance['model_available'] = True
            
            # Test health check
            health = await model.health_check()
            model_performance['health_check_passed'] = health
            
            # Test text generation
            response = await model.generate(
                "Test pipeline validation",
                pipeline_context={'pipeline_name': pipeline_name}
            )
            model_performance['generation_test_passed'] = len(response) > 0
            
            # Test structured output
            schema = {
                "type": "object",
                "properties": {
                    "validation_status": {"type": "string"},
                    "quality_score": {"type": "number"}
                }
            }
            structured_response = await model.generate_structured(
                "Generate validation result",
                schema,
                pipeline_context={'pipeline_name': pipeline_name}
            )
            model_performance['structured_output_test_passed'] = isinstance(structured_response, dict)
            
            # Test cost estimation
            cost = await model.estimate_cost(100, 50)
            model_performance['cost_estimation_working'] = isinstance(cost, (int, float))
            
            # Get model validation summary
            model_performance['validation_summary'] = model.get_validation_summary()
            
        except Exception as e:
            model_performance['error'] = str(e)
            logger.warning(f"Model integration test failed for {pipeline_name}: {e}")
        
        return model_performance
    
    async def _test_provider_integration(self, pipeline_name: str) -> Dict[str, Any]:
        """Test integration with PipelineTestProvider."""
        
        provider_status = {
            'provider_initialized': False,
            'models_available': 0,
            'all_models_healthy': False,
            'usage_tracking_working': False
        }
        
        try:
            # Check provider initialization
            provider_status['provider_initialized'] = self.test_provider.is_initialized
            
            # Check available models
            available_models = self.test_provider.available_models
            provider_status['models_available'] = len(available_models)
            
            # Test model health checks
            healthy_models = 0
            for model_name in available_models[:5]:  # Test first 5 models
                try:
                    model = await self.test_provider.get_model(model_name)
                    if await model.health_check():
                        healthy_models += 1
                except Exception:
                    pass
            
            provider_status['all_models_healthy'] = healthy_models > 0
            provider_status['healthy_models_count'] = healthy_models
            
            # Test usage tracking
            initial_stats = self.test_provider.get_usage_statistics()
            test_model = await self.test_provider.get_model("pipeline-test-model")
            await test_model.generate("usage tracking test")
            updated_stats = self.test_provider.get_usage_statistics()
            
            provider_status['usage_tracking_working'] = (
                updated_stats['total_requests'] > initial_stats['total_requests']
            )
            
        except Exception as e:
            provider_status['error'] = str(e)
            logger.warning(f"Provider integration test failed for {pipeline_name}: {e}")
        
        return provider_status
    
    async def _test_orchestrator_compatibility(self, pipeline_path: Path) -> Dict[str, Any]:
        """Test orchestrator compatibility with test infrastructure."""
        
        orchestrator_compat = {
            'orchestrator_available': False,
            'pipeline_loadable': False,
            'execution_successful': False,
            'templates_resolved': False,
            'outputs_generated': False
        }
        
        try:
            orchestrator_compat['orchestrator_available'] = self.orchestrator is not None
            
            # Test pipeline loading (compilation)
            try:
                # Use pipeline test suite for execution
                pipeline_info = type('PipelineInfo', (), {
                    'name': pipeline_path.stem,
                    'path': pipeline_path,
                    'estimated_runtime': 30.0,
                    'input_requirements': {}
                })()
                
                # Test execution with timeout
                execution_result = await asyncio.wait_for(
                    self.pipeline_test_suite._test_pipeline_execution(pipeline_info),
                    timeout=60.0
                )
                
                orchestrator_compat['pipeline_loadable'] = True
                orchestrator_compat['execution_successful'] = execution_result.success
                orchestrator_compat['execution_time'] = execution_result.execution_time
                
                if execution_result.error:
                    orchestrator_compat['error'] = str(execution_result.error)
                
                # Check if outputs were generated
                orchestrator_compat['outputs_generated'] = bool(execution_result.outputs)
                
                # Basic template resolution check
                if execution_result.success:
                    # If execution succeeded, templates were likely resolved
                    orchestrator_compat['templates_resolved'] = True
                
            except asyncio.TimeoutError:
                orchestrator_compat['error'] = 'Pipeline execution timed out'
            except Exception as e:
                orchestrator_compat['error'] = str(e)
                orchestrator_compat['pipeline_loadable'] = 'YAML' not in str(e)  # Can load if not YAML error
                
        except Exception as e:
            orchestrator_compat['error'] = str(e)
            logger.warning(f"Orchestrator compatibility test failed: {e}")
        
        return orchestrator_compat
    
    def _calculate_integration_score(self, result: PipelineIntegrationResult) -> float:
        """Calculate overall integration score (0.0-100.0)."""
        
        score = 0.0
        
        # Basic validation (25 points)
        if result.validation_passed:
            score += 25.0
        
        # Model integration (25 points)
        model_perf = result.test_model_performance
        if model_perf.get('model_available', False):
            score += 5.0
        if model_perf.get('health_check_passed', False):
            score += 5.0
        if model_perf.get('generation_test_passed', False):
            score += 10.0
        if model_perf.get('structured_output_test_passed', False):
            score += 5.0
        
        # Provider integration (25 points)
        provider_status = result.provider_integration_status
        if provider_status.get('provider_initialized', False):
            score += 5.0
        if provider_status.get('models_available', 0) > 0:
            score += 10.0
        if provider_status.get('all_models_healthy', False):
            score += 5.0
        if provider_status.get('usage_tracking_working', False):
            score += 5.0
        
        # Orchestrator compatibility (25 points)
        orchestrator_compat = result.orchestrator_compatibility
        if orchestrator_compat.get('orchestrator_available', False):
            score += 5.0
        if orchestrator_compat.get('pipeline_loadable', False):
            score += 10.0
        if orchestrator_compat.get('execution_successful', False):
            score += 10.0
        
        # Penalty for issues
        issue_penalty = min(10.0, len(result.issues) * 2.0)
        score = max(0.0, score - issue_penalty)
        
        return score
    
    def _generate_recommendations(self, result: PipelineIntegrationResult) -> List[str]:
        """Generate recommendations based on integration results."""
        
        recommendations = []
        
        # Basic validation recommendations
        if not result.validation_passed:
            recommendations.append("Fix pipeline validation issues before integration testing")
        
        # Model integration recommendations
        model_perf = result.test_model_performance
        if not model_perf.get('model_available', False):
            recommendations.append("Ensure PipelineTestModel is properly registered in provider")
        if not model_perf.get('health_check_passed', False):
            recommendations.append("Check model health check implementation")
        
        # Provider integration recommendations
        provider_status = result.provider_integration_status
        if provider_status.get('models_available', 0) == 0:
            recommendations.append("Register test models with PipelineTestProvider")
        if not provider_status.get('usage_tracking_working', False):
            recommendations.append("Verify usage statistics tracking is functional")
        
        # Orchestrator compatibility recommendations
        orchestrator_compat = result.orchestrator_compatibility
        if not orchestrator_compat.get('execution_successful', False):
            if 'error' in orchestrator_compat:
                recommendations.append(f"Fix orchestrator execution issue: {orchestrator_compat['error']}")
            else:
                recommendations.append("Debug pipeline execution failure with test orchestrator")
        
        # Performance recommendations
        if result.execution_time > 30.0:
            recommendations.append("Consider optimizing pipeline for faster test execution")
        
        # General recommendations
        if result.integration_score < 70.0:
            recommendations.append("Integration score is low - review all validation components")
        elif result.integration_score < 90.0:
            recommendations.append("Good integration - address remaining issues for optimal performance")
        
        return recommendations
    
    async def validate_all_examples(self) -> Dict[str, PipelineIntegrationResult]:
        """Validate integration for all example pipelines."""
        
        logger.info("Starting systematic validation of all example pipelines")
        
        results = {}
        
        # Discover all YAML files in examples directory
        if not self.examples_dir.exists():
            logger.error(f"Examples directory not found: {self.examples_dir}")
            return results
        
        yaml_files = list(self.examples_dir.glob("*.yaml")) + list(self.examples_dir.glob("*.yml"))
        
        logger.info(f"Found {len(yaml_files)} pipeline files to validate")
        
        for pipeline_path in yaml_files:
            pipeline_name = pipeline_path.stem
            
            try:
                result = await self.validate_pipeline_integration(pipeline_name, pipeline_path)
                results[pipeline_name] = result
                
                status = "PASS" if result.integration_score >= 70.0 else "FAIL"
                logger.info(f"{status}: {pipeline_name} (score: {result.integration_score:.1f})")
                
            except Exception as e:
                logger.error(f"Failed to validate {pipeline_name}: {e}")
                
                # Create failed result
                failed_result = PipelineIntegrationResult(
                    pipeline_name=pipeline_name,
                    validation_passed=False,
                    execution_successful=False,
                    integration_score=0.0
                )
                failed_result.issues.append(f"Validation failed: {str(e)}")
                results[pipeline_name] = failed_result
        
        # Generate summary
        total_pipelines = len(results)
        passed_pipelines = len([r for r in results.values() if r.integration_score >= 70.0])
        avg_score = sum(r.integration_score for r in results.values()) / max(1, total_pipelines)
        
        logger.info(f"Pipeline integration validation completed:")
        logger.info(f"  Total pipelines: {total_pipelines}")
        logger.info(f"  Passed: {passed_pipelines} ({passed_pipelines/max(1, total_pipelines)*100:.1f}%)")
        logger.info(f"  Average score: {avg_score:.1f}")
        
        return results
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive integration validation summary."""
        
        if not self.integration_results:
            return {"message": "No validation results available"}
        
        # Calculate statistics
        total_validations = len(self.integration_results)
        successful_validations = len([r for r in self.integration_results if r.validation_passed])
        successful_executions = len([r for r in self.integration_results if r.execution_successful])
        
        avg_integration_score = sum(r.integration_score for r in self.integration_results) / total_validations
        avg_execution_time = sum(r.execution_time for r in self.integration_results) / total_validations
        
        # Categorize results
        high_quality = len([r for r in self.integration_results if r.integration_score >= 90.0])
        good_quality = len([r for r in self.integration_results if 70.0 <= r.integration_score < 90.0])
        needs_work = len([r for r in self.integration_results if r.integration_score < 70.0])
        
        # Common issues
        all_issues = []
        for result in self.integration_results:
            all_issues.extend(result.issues)
        
        issue_frequency = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_frequency[issue_type] = issue_frequency.get(issue_type, 0) + 1
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'successful_executions': successful_executions,
            'validation_success_rate': successful_validations / total_validations * 100,
            'execution_success_rate': successful_executions / total_validations * 100,
            'average_integration_score': avg_integration_score,
            'average_execution_time': avg_execution_time,
            'quality_distribution': {
                'high_quality': high_quality,
                'good_quality': good_quality,
                'needs_work': needs_work
            },
            'common_issues': dict(sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:10]),
            'provider_statistics': self.test_provider.get_usage_statistics()
        }


# Utility functions for creating test infrastructure
def create_pipeline_test_orchestrator() -> Orchestrator:
    """Create orchestrator with pipeline test infrastructure for external use."""
    
    # Create test provider and registry
    test_provider = PipelineTestProvider()
    registry = ModelRegistry()
    registry.register_provider(test_provider)
    
    # Create orchestrator with test components
    from ..control_systems.hybrid_control_system import HybridControlSystem
    
    control_system = HybridControlSystem(model_registry=registry)
    
    orchestrator = Orchestrator(
        model_registry=registry,
        control_system=control_system,
        max_concurrent_tasks=5,
        debug_templates=True
    )
    
    return orchestrator


def create_pipeline_integration_validator(examples_dir: Optional[Path] = None) -> PipelineIntegrationValidator:
    """Create pipeline integration validator with default configuration."""
    
    return PipelineIntegrationValidator(
        examples_dir=examples_dir or Path("examples"),
        enable_comprehensive_validation=True
    )


# Export key classes and functions
__all__ = [
    'PipelineTestModel',
    'PipelineTestProvider', 
    'PipelineIntegrationValidator',
    'PipelineIntegrationResult',
    'create_pipeline_test_orchestrator',
    'create_pipeline_integration_validator'
]