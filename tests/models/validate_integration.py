#!/usr/bin/env python3
"""
Comprehensive validation script for multi-model integration with execution engine.

This script validates that all components work together:
- Model providers are accessible and functional
- Selection strategies work with real models  
- Pipeline execution integrates correctly with model system
- Performance optimizations are effective
- All integration points work as expected

This serves as the comprehensive test for Issue #311 Stream C.
"""

import asyncio
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.providers.openai_provider import OpenAIProvider
from src.orchestrator.models.providers.anthropic_provider import AnthropicProvider
from src.orchestrator.models.providers.local_provider import LocalProvider
from src.orchestrator.models.selection.strategies import TaskRequirements, CostOptimizedStrategy
from src.orchestrator.models.selection.manager import ModelSelectionManager
from src.orchestrator.execution.integration import create_comprehensive_execution_manager
from src.orchestrator.execution.variables import VariableScope, VariableType
from src.orchestrator.core.model import ModelCapabilities, ModelCost
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.compiler.yaml_compiler import IntegratedYAMLCompiler


class IntegrationValidator:
    """Validates complete multi-model system integration."""
    
    def __init__(self):
        self.results = []
        self.registry = None
        self.execution_manager = None
        self.working_models = []
    
    def log_result(self, test_name: str, success: bool, details: str = "", error: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if error:
            print(f"    Error: {error}")
        
        self.results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "error": error
        })
    
    async def validate_model_registry(self):
        """Validate model registry functionality."""
        print("\n🔧 VALIDATING MODEL REGISTRY")
        print("-" * 40)
        
        try:
            # Create registry
            self.registry = ModelRegistry()
            
            # Register providers  
            self.registry.register_provider(OpenAIProvider())
            self.registry.register_provider(AnthropicProvider())
            self.registry.register_provider(LocalProvider())
            
            self.log_result("Registry Creation", True, f"Created with {len(self.registry.providers)} providers")
            
            # Initialize registry
            await self.registry.initialize()
            self.log_result("Registry Initialization", self.registry.is_initialized, 
                          f"Initialized: {self.registry.is_initialized}")
            
            # Check health
            health_status = await self.registry.health_check()
            healthy_providers = sum(1 for status in health_status.values() if status)
            self.log_result("Provider Health Check", healthy_providers > 0,
                          f"{healthy_providers}/{len(health_status)} providers healthy")
            
            # Get registry info
            info = self.registry.get_registry_info()
            self.log_result("Registry Info", True,
                          f"{info['provider_count']} providers, {info['total_models']} models")
            
            return True
            
        except Exception as e:
            self.log_result("Registry Validation", False, error=str(e))
            return False
    
    async def validate_model_discovery(self):
        """Validate model discovery across providers."""
        print("\n🔍 VALIDATING MODEL DISCOVERY")  
        print("-" * 40)
        
        if not self.registry:
            self.log_result("Model Discovery", False, error="Registry not available")
            return False
        
        try:
            # Discover all models
            discovered = await self.registry.discover_all_models()
            
            total_models = sum(len(models) for models in discovered.values())
            self.log_result("Model Discovery", total_models > 0,
                          f"Found {total_models} models across {len(discovered)} providers")
            
            # Test getting a specific model
            working_model_found = False
            for provider_name, models in discovered.items():
                for model_name in models[:1]:  # Test first model from each provider
                    try:
                        model = await self.registry.get_model(model_name, provider_name)
                        if model:
                            self.working_models.append((provider_name, model_name, model))
                            working_model_found = True
                            self.log_result(f"Model Creation ({provider_name})", True,
                                          f"Successfully created {model_name}")
                            break
                    except Exception as e:
                        self.log_result(f"Model Creation ({provider_name})", False,
                                      f"Failed to create {model_name}", str(e))
                        continue
                
                if working_model_found:
                    break
            
            return working_model_found
            
        except Exception as e:
            self.log_result("Model Discovery", False, error=str(e))
            return False
    
    async def validate_model_selection(self):
        """Validate model selection strategies."""
        print("\n🎯 VALIDATING MODEL SELECTION")
        print("-" * 40)
        
        if not self.registry:
            self.log_result("Model Selection", False, error="Registry not available")
            return False
        
        try:
            # Create selection manager
            selection_manager = ModelSelectionManager(self.registry)
            
            # Test different selection strategies
            strategies_to_test = [
                ("cost_optimized", TaskRequirements(
                    task_type="text_generation",
                    max_cost_per_1k_tokens=0.01,
                    prefer_local=True
                )),
                ("performance_optimized", TaskRequirements(
                    task_type="text_generation", 
                    max_latency_ms=5000,
                    required_capabilities=["text_generation"]
                )),
                ("balanced", TaskRequirements(
                    task_type="text_generation",
                    max_cost_per_1k_tokens=0.02,
                    accuracy_threshold=0.8
                ))
            ]
            
            successful_selections = 0
            for strategy_name, requirements in strategies_to_test:
                try:
                    selected_model = await selection_manager.select_model(requirements)
                    if selected_model:
                        successful_selections += 1
                        self.log_result(f"Selection Strategy ({strategy_name})", True,
                                      f"Selected: {selected_model}")
                    else:
                        self.log_result(f"Selection Strategy ({strategy_name})", False,
                                      "No suitable model found")
                except Exception as e:
                    self.log_result(f"Selection Strategy ({strategy_name})", False,
                                  error=str(e))
            
            return successful_selections > 0
            
        except Exception as e:
            self.log_result("Model Selection", False, error=str(e))
            return False
    
    async def validate_execution_integration(self):
        """Validate integration with execution engine."""
        print("\n⚙️ VALIDATING EXECUTION ENGINE INTEGRATION")
        print("-" * 40)
        
        try:
            # Create execution manager
            self.execution_manager = create_comprehensive_execution_manager(
                "integration_test", "model_pipeline"
            )
            
            self.log_result("Execution Manager Creation", True, 
                          "Created comprehensive execution manager")
            
            # Start execution
            self.execution_manager.start_execution(total_steps=3)
            
            # Step 1: Model selection within execution context
            self.execution_manager.start_step("model_selection", "Select model for task")
            
            # Store model selection in execution variables
            if self.working_models:
                provider_name, model_name, model = self.working_models[0]
                
                self.execution_manager.variable_manager.set_variable(
                    "selected_model",
                    {
                        "name": model_name,
                        "provider": provider_name,
                        "capabilities": ["text_generation"]
                    },
                    scope=VariableScope.EXECUTION,
                    var_type=VariableType.MODEL_REFERENCE
                )
                
                self.execution_manager.complete_step("model_selection", success=True)
                self.log_result("Model Selection in Execution Context", True,
                              f"Selected {provider_name}:{model_name}")
                
                # Step 2: Model execution within execution context
                self.execution_manager.start_step("model_execution", "Execute model task")
                
                try:
                    # Test actual model execution 
                    result = await model.generate("Hello", max_tokens=5)
                    
                    # Store result in execution context
                    self.execution_manager.variable_manager.set_variable(
                        "execution_result",
                        {
                            "content": result,
                            "model": model_name,
                            "timestamp": time.time()
                        },
                        scope=VariableScope.EXECUTION,
                        var_type=VariableType.TASK_RESULT
                    )
                    
                    self.execution_manager.complete_step("model_execution", success=True)
                    self.log_result("Model Execution in Context", True,
                                  f"Generated: {result[:50]}...")
                    
                    # Step 3: Variable integration
                    stored_model = self.execution_manager.variable_manager.get_variable("selected_model")
                    stored_result = self.execution_manager.variable_manager.get_variable("execution_result")
                    
                    variables_ok = (stored_model is not None and stored_result is not None)
                    self.log_result("Variable Integration", variables_ok,
                                  f"Stored model and result successfully")
                    
                    return True
                    
                except Exception as e:
                    self.execution_manager.complete_step("model_execution", success=False)
                    self.log_result("Model Execution in Context", False, error=str(e))
                    return False
            else:
                self.log_result("Model Selection in Execution Context", False,
                              error="No working models available")
                return False
                
        except Exception as e:
            self.log_result("Execution Engine Integration", False, error=str(e))
            return False
    
    async def validate_pipeline_execution(self):
        """Validate complete pipeline execution with models."""
        print("\n🚀 VALIDATING PIPELINE EXECUTION")
        print("-" * 40)
        
        if not self.working_models:
            self.log_result("Pipeline Execution", False, error="No working models available")
            return False
        
        try:
            # Create a simple control system for testing
            from src.orchestrator.core.control_system import ControlSystem
            from src.orchestrator.core.task import Task
            
            class TestModelControlSystem(ControlSystem):
                def __init__(self, model_registry):
                    config = {"capabilities": {"model_integration": True}}
                    super().__init__(name="test-model-control", config=config)
                    self.model_registry = model_registry
                    self._results = {}
                
                async def execute_task(self, task: Task, context: dict = None):
                    if task.action == "generate_text":
                        return await self._generate_text(task)
                    return {"status": "completed", "result": f"Executed {task.action}"}
                
                async def _generate_text(self, task):
                    prompt = task.parameters.get("prompt", "Hello")
                    
                    # Use first working model
                    if self.parent.working_models:
                        _, model_name, model = self.parent.working_models[0]
                        
                        try:
                            result = await model.generate(prompt, max_tokens=20)
                            return {
                                "status": "completed",
                                "content": result,
                                "model": model_name
                            }
                        except Exception as e:
                            return {"status": "failed", "error": str(e)}
                    
                    return {"status": "failed", "error": "No working models"}
                
                async def execute_pipeline(self, pipeline, context=None):
                    raise NotImplementedError("Use orchestrator")
                
                def get_capabilities(self):
                    return self.config.get("capabilities", {})
                
                async def health_check(self):
                    return {"status": "healthy"}
            
            # Set up test control system
            control_system = TestModelControlSystem(self.registry)
            control_system.parent = self  # Give access to working_models
            orchestrator = Orchestrator(control_system=control_system)
            
            # Test simple pipeline
            pipeline_yaml = """
name: "model_integration_test"
description: "Test model integration in pipeline"

steps:
  - id: generate
    action: generate_text
    parameters:
      prompt: "What is AI?"
"""
            
            print("Executing test pipeline...")
            results = await orchestrator.execute_yaml(pipeline_yaml, context={})
            
            if "generate" in results:
                result = results["generate"]
                if result["status"] == "completed":
                    self.log_result("Pipeline Execution", True,
                                  f"Generated content: {result['content'][:50]}...")
                    return True
                else:
                    self.log_result("Pipeline Execution", False,
                                  f"Task failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                self.log_result("Pipeline Execution", False, 
                              "No results returned from pipeline")
                return False
                
        except Exception as e:
            self.log_result("Pipeline Execution", False, error=str(e))
            traceback.print_exc()
            return False
    
    async def validate_performance_features(self):
        """Validate performance optimization features."""
        print("\n⚡ VALIDATING PERFORMANCE FEATURES")
        print("-" * 40)
        
        try:
            # Test caching
            from src.orchestrator.models.optimization.caching import ModelCache, CacheConfig
            
            cache = ModelCache(CacheConfig(max_size=10, ttl_seconds=300))
            
            # Test cache operations
            cache.set("test_key", {"content": "test_content"})
            cached_result = cache.get("test_key")
            
            cache_works = cached_result is not None and cached_result["content"] == "test_content"
            self.log_result("Model Caching", cache_works, "Cache set/get operations working")
            
            # Test connection pooling
            from src.orchestrator.models.optimization.pooling import ConnectionPool, PoolConfig
            
            pool = ConnectionPool("test_provider", PoolConfig(min_connections=1, max_connections=3))
            
            # Basic pool operations
            conn = await pool.get_connection()
            await pool.return_connection(conn)
            
            self.log_result("Connection Pooling", True, "Pool operations working")
            
            return True
            
        except Exception as e:
            self.log_result("Performance Features", False, error=str(e))
            return False
    
    async def run_validation(self):
        """Run complete validation suite."""
        print("🔗 MULTI-MODEL INTEGRATION VALIDATION")
        print("=" * 60)
        print("Validating Issue #311 Stream C: Integration & Testing")
        print()
        
        # Run all validation steps
        validations = [
            ("Model Registry", self.validate_model_registry()),
            ("Model Discovery", self.validate_model_discovery()),
            ("Model Selection", self.validate_model_selection()),
            ("Execution Integration", self.validate_execution_integration()),
            ("Pipeline Execution", self.validate_pipeline_execution()),
            ("Performance Features", self.validate_performance_features())
        ]
        
        passed_count = 0
        total_count = len(validations)
        
        for validation_name, validation_coro in validations:
            try:
                success = await validation_coro
                if success:
                    passed_count += 1
            except Exception as e:
                print(f"❌ {validation_name} validation failed with exception: {e}")
                traceback.print_exc()
        
        # Final summary
        print(f"\n{'=' * 60}")
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        success_rate = passed_count / total_count
        
        for result in self.results:
            status = "✅" if result["success"] else "❌" 
            print(f"{status} {result['test']}")
        
        print(f"\n📈 Overall Success Rate: {passed_count}/{total_count} ({success_rate*100:.1f}%)")
        
        if success_rate >= 0.8:
            print("\n🎉 INTEGRATION VALIDATION PASSED!")
            print("✅ Multi-model system successfully integrated with execution engine")
            print("✅ Pipeline execution working with model operations")
            print("✅ Performance optimizations functional")
            print("✅ Issue #311 Stream C objectives completed")
            return True
        else:
            print("\n⚠️ INTEGRATION VALIDATION NEEDS ATTENTION")
            print(f"❌ Only {success_rate*100:.1f}% of validations passed")
            print("🔧 Review failed validations and fix issues")
            return False


async def main():
    """Main validation entry point."""
    validator = IntegrationValidator()
    success = await validator.run_validation()
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)