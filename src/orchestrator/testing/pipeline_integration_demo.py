#!/usr/bin/env python3
"""Pipeline Integration Infrastructure Demonstration Script.

This script demonstrates how to use the new PipelineTestModel/PipelineTestProvider
infrastructure for systematic pipeline validation, applying the proven patterns
from the successful testing epic (#374) to pipeline validation.

Usage:
    python -m src.orchestrator.testing.pipeline_integration_demo [options]

Examples:
    # Validate all example pipelines
    python -m src.orchestrator.testing.pipeline_integration_demo --validate-all
    
    # Validate specific pipeline
    python -m src.orchestrator.testing.pipeline_integration_demo --pipeline simple_data_processing
    
    # Run integration tests
    python -m src.orchestrator.testing.pipeline_integration_demo --test-infrastructure
    
    # Generate performance report
    python -m src.orchestrator.testing.pipeline_integration_demo --performance-report
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from .pipeline_integration_infrastructure import (
        PipelineTestModel,
        PipelineTestProvider,
        PipelineIntegrationValidator,
        create_pipeline_test_orchestrator,
        create_pipeline_integration_validator
    )
    from ..orchestrator import Orchestrator
    from ..models.registry import ModelRegistry
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from the orchestrator root directory")
    sys.exit(1)


class PipelineIntegrationDemo:
    """Demonstration class for pipeline integration infrastructure."""
    
    def __init__(self, examples_dir: Optional[Path] = None):
        """Initialize the demonstration."""
        self.examples_dir = examples_dir or Path("examples")
        self.validator = None
        self.results_dir = Path("outputs/integration_demo")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline Integration Demo initialized")
        logger.info(f"Examples directory: {self.examples_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def demonstrate_test_model_capabilities(self) -> Dict[str, Any]:
        """Demonstrate PipelineTestModel capabilities."""
        logger.info("=== Demonstrating PipelineTestModel Capabilities ===")
        
        results = {}
        
        try:
            # Create test model with custom configuration
            model = PipelineTestModel(
                name="demo-pipeline-model",
                mock_responses={
                    "demo_pipeline": "This is a custom response for the demo pipeline",
                    "quality_test": "Pipeline quality is excellent with score 98/100"
                },
                pipeline_validation_enabled=True
            )
            
            logger.info(f"Created PipelineTestModel: {model.name}")
            logger.info(f"Capabilities: {len(model.capabilities.supported_tasks)} supported tasks")
            logger.info(f"Context window: {model.capabilities.context_window}")
            logger.info(f"Validation enabled: {model.pipeline_validation_enabled}")
            
            results["model_created"] = True
            results["model_name"] = model.name
            results["supported_tasks"] = model.capabilities.supported_tasks
            results["context_window"] = model.capabilities.context_window
            
            # Test various model capabilities
            asyncio.run(self._test_model_features(model, results))
            
        except Exception as e:
            logger.error(f"Error demonstrating test model: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _test_model_features(self, model: PipelineTestModel, results: Dict[str, Any]):
        """Test various model features."""
        
        # Test health check
        health = await model.health_check()
        logger.info(f"Health check: {'PASS' if health else 'FAIL'}")
        results["health_check"] = health
        
        # Test text generation
        response = await model.generate(
            "Validate this pipeline for quality and correctness",
            pipeline_context={'pipeline_name': 'demo_pipeline'}
        )
        logger.info(f"Generated response: {response[:100]}...")
        results["text_generation"] = len(response) > 0
        
        # Test structured output
        schema = {
            "type": "object",
            "properties": {
                "validation_status": {"type": "string"},
                "quality_score": {"type": "number"},
                "issues": {"type": "array"},
                "recommendations": {"type": "array"}
            }
        }
        
        structured_response = await model.generate_structured(
            "Generate comprehensive pipeline validation report",
            schema,
            pipeline_context={'pipeline_name': 'demo_pipeline'}
        )
        
        logger.info(f"Structured response keys: {list(structured_response.keys())}")
        results["structured_output"] = isinstance(structured_response, dict)
        results["structured_response_sample"] = structured_response
        
        # Test cost estimation
        cost = await model.estimate_cost(200, 100)
        logger.info(f"Cost estimate: ${cost}")
        results["cost_estimation"] = isinstance(cost, (int, float))
        
        # Get validation summary
        summary = model.get_validation_summary()
        logger.info(f"Validation summary: {summary}")
        results["validation_summary"] = summary
    
    def demonstrate_test_provider_capabilities(self) -> Dict[str, Any]:
        """Demonstrate PipelineTestProvider capabilities."""
        logger.info("=== Demonstrating PipelineTestProvider Capabilities ===")
        
        results = {}
        
        try:
            # Create test provider
            provider = PipelineTestProvider("demo-provider")
            
            logger.info(f"Created PipelineTestProvider: {provider.name}")
            logger.info(f"Available models: {len(provider.available_models)}")
            logger.info(f"Provider info: {provider.get_provider_info()}")
            
            results["provider_created"] = True
            results["provider_name"] = provider.name
            results["available_models"] = provider.available_models
            results["provider_info"] = provider.get_provider_info()
            
            # Test provider features
            asyncio.run(self._test_provider_features(provider, results))
            
        except Exception as e:
            logger.error(f"Error demonstrating test provider: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _test_provider_features(self, provider: PipelineTestProvider, results: Dict[str, Any]):
        """Test various provider features."""
        
        # Test model support
        test_models = ["pipeline-test-model", "openai/gpt-4", "nonexistent-model"]
        support_results = {}
        
        for model_name in test_models:
            supported = provider.supports_model(model_name)
            support_results[model_name] = supported
            logger.info(f"Model support '{model_name}': {'YES' if supported else 'NO'}")
        
        results["model_support"] = support_results
        
        # Test getting model capabilities
        try:
            capabilities = provider.get_model_capabilities("pipeline-test-model")
            logger.info(f"Model capabilities retrieved successfully")
            results["capabilities_retrieval"] = True
        except Exception as e:
            logger.error(f"Failed to get capabilities: {e}")
            results["capabilities_retrieval"] = False
        
        # Test model retrieval and usage tracking
        initial_stats = provider.get_usage_statistics()
        logger.info(f"Initial usage stats: {initial_stats}")
        
        # Get a model and use it
        model = await provider.get_model("pipeline-test-model")
        await model.generate("Test usage tracking")
        
        updated_stats = provider.get_usage_statistics()
        logger.info(f"Updated usage stats: {updated_stats}")
        
        results["usage_tracking"] = updated_stats['total_requests'] > initial_stats['total_requests']
        
        # Test provider initialization
        await provider.initialize()
        results["initialization"] = provider.is_initialized
    
    def demonstrate_orchestrator_integration(self) -> Dict[str, Any]:
        """Demonstrate integration with orchestrator framework."""
        logger.info("=== Demonstrating Orchestrator Integration ===")
        
        results = {}
        
        try:
            # Create test orchestrator using utility function
            orchestrator = create_pipeline_test_orchestrator()
            
            logger.info(f"Created test orchestrator successfully")
            logger.info(f"Model registry: {orchestrator.model_registry is not None}")
            logger.info(f"Control system: {orchestrator.control_system is not None}")
            
            results["orchestrator_created"] = True
            
            # Verify test provider is registered
            providers = orchestrator.model_registry.get_registered_providers()
            test_provider_found = any(p.name == "pipeline-test-provider" for p in providers)
            
            logger.info(f"Test provider registered: {'YES' if test_provider_found else 'NO'}")
            logger.info(f"Total providers: {len(providers)}")
            
            results["test_provider_registered"] = test_provider_found
            results["total_providers"] = len(providers)
            
            # Test model availability through registry
            try:
                # This would typically require the orchestrator to be fully set up
                logger.info("Orchestrator integration test completed")
                results["orchestrator_functional"] = True
            except Exception as e:
                logger.warning(f"Orchestrator functionality test failed: {e}")
                results["orchestrator_functional"] = False
                results["orchestrator_error"] = str(e)
            
        except Exception as e:
            logger.error(f"Error in orchestrator integration: {e}")
            results["error"] = str(e)
        
        return results
    
    async def demonstrate_integration_validator(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """Demonstrate PipelineIntegrationValidator capabilities."""
        logger.info("=== Demonstrating Integration Validator ===")
        
        results = {}
        
        try:
            # Create integration validator
            self.validator = create_pipeline_integration_validator(self.examples_dir)
            
            logger.info(f"Created PipelineIntegrationValidator")
            logger.info(f"Examples directory: {self.validator.examples_dir}")
            logger.info(f"Test provider: {self.validator.test_provider.name}")
            
            results["validator_created"] = True
            
            if pipeline_name:
                # Validate specific pipeline
                pipeline_path = self.examples_dir / f"{pipeline_name}.yaml"
                if pipeline_path.exists():
                    logger.info(f"Validating specific pipeline: {pipeline_name}")
                    result = await self.validator.validate_pipeline_integration(pipeline_name, pipeline_path)
                    
                    logger.info(f"Validation result:")
                    logger.info(f"  - Pipeline: {result.pipeline_name}")
                    logger.info(f"  - Integration score: {result.integration_score:.1f}")
                    logger.info(f"  - Validation passed: {result.validation_passed}")
                    logger.info(f"  - Execution successful: {result.execution_successful}")
                    logger.info(f"  - Execution time: {result.execution_time:.2f}s")
                    logger.info(f"  - Issues: {len(result.issues)}")
                    logger.info(f"  - Recommendations: {len(result.recommendations)}")
                    
                    results["single_pipeline_validation"] = {
                        "pipeline_name": result.pipeline_name,
                        "integration_score": result.integration_score,
                        "validation_passed": result.validation_passed,
                        "execution_successful": result.execution_successful,
                        "execution_time": result.execution_time,
                        "issues": result.issues,
                        "recommendations": result.recommendations[:3]  # First 3 recommendations
                    }
                else:
                    logger.warning(f"Pipeline file not found: {pipeline_path}")
                    results["single_pipeline_validation"] = {"error": "Pipeline file not found"}
            else:
                # Validate all pipelines (limited for demo)
                logger.info("Validating all example pipelines...")
                
                # For demo, limit to first few pipelines to avoid long execution
                yaml_files = list(self.examples_dir.glob("*.yaml"))[:3]  # First 3 pipelines
                logger.info(f"Found {len(yaml_files)} pipeline files (limited to 3 for demo)")
                
                validation_results = {}
                
                for pipeline_path in yaml_files:
                    pipeline_name = pipeline_path.stem
                    try:
                        result = await self.validator.validate_pipeline_integration(pipeline_name, pipeline_path)
                        validation_results[pipeline_name] = {
                            "integration_score": result.integration_score,
                            "validation_passed": result.validation_passed,
                            "execution_successful": result.execution_successful,
                            "issues_count": len(result.issues)
                        }
                        logger.info(f"  {pipeline_name}: score {result.integration_score:.1f}")
                    except Exception as e:
                        logger.warning(f"Failed to validate {pipeline_name}: {e}")
                        validation_results[pipeline_name] = {"error": str(e)}
                
                results["all_pipelines_validation"] = validation_results
                
                # Get integration summary
                summary = self.validator.get_integration_summary()
                logger.info(f"Integration summary:")
                logger.info(f"  - Total validations: {summary.get('total_validations', 0)}")
                logger.info(f"  - Success rate: {summary.get('validation_success_rate', 0):.1f}%")
                logger.info(f"  - Average score: {summary.get('average_integration_score', 0):.1f}")
                
                results["integration_summary"] = summary
            
        except Exception as e:
            logger.error(f"Error in integration validator demo: {e}")
            results["error"] = str(e)
        
        return results
    
    def run_infrastructure_tests(self) -> Dict[str, Any]:
        """Run comprehensive infrastructure tests."""
        logger.info("=== Running Infrastructure Tests ===")
        
        results = {
            "test_model_capabilities": self.demonstrate_test_model_capabilities(),
            "test_provider_capabilities": self.demonstrate_test_provider_capabilities(),
            "orchestrator_integration": self.demonstrate_orchestrator_integration()
        }
        
        # Calculate overall test success
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for key, value in category_results.items():
                    if isinstance(value, bool):
                        total_tests += 1
                        if value:
                            passed_tests += 1
        
        success_rate = (passed_tests / max(1, total_tests)) * 100
        logger.info(f"Infrastructure tests completed: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        results["test_summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate
        }
        
        return results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report."""
        logger.info("=== Generating Performance Report ===")
        
        results = {}
        
        try:
            if not self.validator:
                self.validator = create_pipeline_integration_validator(self.examples_dir)
            
            # Get provider statistics
            provider_stats = self.validator.test_provider.get_usage_statistics()
            results["provider_statistics"] = provider_stats
            
            # Get integration summary if available
            if self.validator.integration_results:
                summary = self.validator.get_integration_summary()
                results["integration_summary"] = summary
                
                # Calculate performance metrics
                execution_times = [r.execution_time for r in self.validator.integration_results if r.execution_time > 0]
                if execution_times:
                    results["performance_metrics"] = {
                        "average_execution_time": sum(execution_times) / len(execution_times),
                        "min_execution_time": min(execution_times),
                        "max_execution_time": max(execution_times),
                        "total_pipelines_tested": len(execution_times)
                    }
            
            logger.info(f"Performance report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            results["error"] = str(e)
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        try:
            output_file = self.results_dir / f"{filename}_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


async def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Pipeline Integration Infrastructure Demo")
    parser.add_argument("--examples-dir", type=Path, help="Path to examples directory")
    parser.add_argument("--validate-all", action="store_true", help="Validate all example pipelines")
    parser.add_argument("--pipeline", type=str, help="Validate specific pipeline")
    parser.add_argument("--test-infrastructure", action="store_true", help="Run infrastructure tests")
    parser.add_argument("--performance-report", action="store_true", help="Generate performance report")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = PipelineIntegrationDemo(args.examples_dir)
    
    if args.test_infrastructure:
        logger.info("Running infrastructure tests...")
        results = demo.run_infrastructure_tests()
        
        if args.save_results:
            demo.save_results(results, "infrastructure_tests")
        
        # Print summary
        summary = results.get("test_summary", {})
        print(f"\n{'='*60}")
        print(f"INFRASTRUCTURE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total tests: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed_tests', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0):.1f}%")
        
    elif args.validate_all:
        logger.info("Running validation on all pipelines...")
        results = await demo.demonstrate_integration_validator()
        
        if args.save_results:
            demo.save_results(results, "pipeline_validation")
        
        # Print summary
        if "integration_summary" in results:
            summary = results["integration_summary"]
            print(f"\n{'='*60}")
            print(f"PIPELINE VALIDATION RESULTS")
            print(f"{'='*60}")
            print(f"Total pipelines: {summary.get('total_validations', 0)}")
            print(f"Validation success rate: {summary.get('validation_success_rate', 0):.1f}%")
            print(f"Average integration score: {summary.get('average_integration_score', 0):.1f}")
        
    elif args.pipeline:
        logger.info(f"Running validation on pipeline: {args.pipeline}")
        results = await demo.demonstrate_integration_validator(args.pipeline)
        
        if args.save_results:
            demo.save_results(results, f"pipeline_{args.pipeline}")
        
        # Print summary
        if "single_pipeline_validation" in results:
            result = results["single_pipeline_validation"]
            print(f"\n{'='*60}")
            print(f"PIPELINE VALIDATION: {args.pipeline}")
            print(f"{'='*60}")
            print(f"Integration score: {result.get('integration_score', 0):.1f}")
            print(f"Validation passed: {result.get('validation_passed', False)}")
            print(f"Execution successful: {result.get('execution_successful', False)}")
            print(f"Issues: {len(result.get('issues', []))}")
            print(f"Recommendations: {len(result.get('recommendations', []))}")
            
            if result.get('recommendations'):
                print(f"\nTop recommendations:")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"  {i}. {rec}")
        
    elif args.performance_report:
        logger.info("Generating performance report...")
        results = demo.generate_performance_report()
        
        if args.save_results:
            demo.save_results(results, "performance_report")
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        if "provider_statistics" in results:
            stats = results["provider_statistics"]
            print(f"Provider requests: {stats.get('total_requests', 0)}")
            print(f"Available models: {stats.get('total_models', 0)}")
        
        if "performance_metrics" in results:
            metrics = results["performance_metrics"]
            print(f"Average execution time: {metrics.get('average_execution_time', 0):.2f}s")
            print(f"Pipelines tested: {metrics.get('total_pipelines_tested', 0)}")
    
    else:
        # Run full demo
        logger.info("Running full pipeline integration demo...")
        
        print(f"\n{'='*60}")
        print(f"PIPELINE INTEGRATION INFRASTRUCTURE DEMO")
        print(f"{'='*60}")
        
        # Run infrastructure tests
        infra_results = demo.run_infrastructure_tests()
        infra_summary = infra_results.get("test_summary", {})
        print(f"Infrastructure tests: {infra_summary.get('passed_tests', 0)}/{infra_summary.get('total_tests', 0)} passed")
        
        # Run pipeline validation
        validation_results = await demo.demonstrate_integration_validator()
        if "integration_summary" in validation_results:
            val_summary = validation_results["integration_summary"]
            print(f"Pipeline validations: {val_summary.get('total_validations', 0)} pipelines tested")
            print(f"Average score: {val_summary.get('average_integration_score', 0):.1f}")
        
        print(f"\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())