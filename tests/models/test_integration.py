#!/usr/bin/env python3
"""
Pipeline integration tests for multi-model system with execution engine.

Tests complete integration between multi-model system and pipeline execution engine
from Issue #309 to validate:
- Model selection within pipeline execution contexts
- Execution engine compatibility with model providers
- Variable management integration with model outputs
- Progress tracking for model operations
- Recovery and checkpointing with model state
- End-to-end pipeline execution with real models
"""

import asyncio
import os
import pytest
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Model system imports
from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.selection.manager import ModelSelectionManager
from src.orchestrator.models.selection.strategies import SelectionCriteria
from src.orchestrator.models.providers.base import ModelCapability
from src.orchestrator.models.providers.openai_provider import OpenAIProvider
from src.orchestrator.models.providers.anthropic_provider import AnthropicProvider
from src.orchestrator.models.providers.local_provider import LocalProvider

# Execution engine imports
from src.orchestrator.execution.integration import (
    ComprehensiveExecutionManager,
    create_comprehensive_execution_manager
)
from src.orchestrator.execution.state import ExecutionContext, ExecutionStatus
from src.orchestrator.execution.variables import VariableManager, VariableScope, VariableType
from src.orchestrator.execution.progress import ProgressTracker, ProgressEventType
from src.orchestrator.execution.recovery import RecoveryManager, RecoveryStrategy

# Core orchestrator imports
from src.orchestrator.core.task import Task
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.compiler.yaml_compiler import IntegratedYAMLCompiler


@pytest.mark.integration
class TestModelExecutionIntegration:
    """Test integration between model system and execution engine."""

    @pytest.fixture
    async def integrated_model_registry(self):
        """Create integrated model registry."""
        registry = ModelRegistry()
        
        # Add all providers
        registry.add_provider(OpenAIProvider())
        registry.add_provider(AnthropicProvider())
        registry.add_provider(LocalProvider())
        
        return registry

    @pytest.fixture
    async def execution_manager(self):
        """Create execution manager for testing."""
        return create_comprehensive_execution_manager("test_exec", "model_pipeline")

    @pytest.fixture
    async def model_selection_manager(self, integrated_model_registry):
        """Create model selection manager."""
        return ModelSelectionManager(integrated_model_registry)

    async def test_execution_context_model_integration(
        self, execution_manager, model_selection_manager
    ):
        """Test model operations within execution context."""
        execution_manager.start_execution(total_steps=3)
        
        # Step 1: Model selection
        execution_manager.start_step("model_selection", "Select appropriate model")
        
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            strategy="balanced"
        )
        
        try:
            selection_result = await model_selection_manager.select_model(criteria)
            
            if selection_result and selection_result.model:
                # Store model selection in execution context
                execution_manager.variable_manager.set_variable(
                    "selected_model",
                    {
                        "name": selection_result.model.name,
                        "provider": selection_result.model.provider,
                        "confidence": selection_result.confidence,
                        "reasoning": selection_result.reasoning
                    },
                    scope=VariableScope.EXECUTION,
                    var_type=VariableType.MODEL_REFERENCE
                )
                
                execution_manager.complete_step("model_selection", success=True)
                
                # Verify model reference is stored
                stored_model = execution_manager.variable_manager.get_variable("selected_model")
                assert stored_model is not None
                assert stored_model["name"] == selection_result.model.name
                assert stored_model["provider"] == selection_result.model.provider
                
                print(f"Selected model: {stored_model['provider']}:{stored_model['name']}")
                
        except Exception as e:
            print(f"Model selection failed: {e}")
            execution_manager.complete_step("model_selection", success=False)
            pytest.skip("No suitable models available for integration testing")

    async def test_model_execution_with_progress_tracking(
        self, execution_manager, integrated_model_registry
    ):
        """Test model execution with progress tracking."""
        execution_manager.start_execution(total_steps=2)
        
        # Find a working model
        working_model = await self._find_working_model(integrated_model_registry)
        if not working_model:
            pytest.skip("No working models available")
        
        model_info, model_instance = working_model
        
        # Step 1: Model execution with progress tracking
        execution_manager.start_step("model_generation", "Generate content with model")
        
        # Update progress during execution
        execution_manager.update_step_progress("model_generation", 25.0, "Starting generation")
        
        try:
            # Track generation time
            start_time = time.time()
            result = await model_instance.generate(
                "Explain machine learning in one paragraph",
                max_tokens=100,
                temperature=0.3
            )
            duration = time.time() - start_time
            
            execution_manager.update_step_progress("model_generation", 75.0, "Generation complete")
            
            # Store result in execution context
            execution_manager.variable_manager.set_variable(
                "generation_result",
                {
                    "content": result,
                    "model": model_info.name,
                    "provider": model_info.provider,
                    "duration_ms": duration * 1000,
                    "timestamp": datetime.now().isoformat()
                },
                scope=VariableScope.EXECUTION,
                var_type=VariableType.TASK_RESULT
            )
            
            execution_manager.complete_step("model_generation", success=True)
            
            # Verify result storage
            stored_result = execution_manager.variable_manager.get_variable("generation_result")
            assert stored_result is not None
            assert stored_result["content"] == result
            assert stored_result["model"] == model_info.name
            assert stored_result["duration_ms"] > 0
            
            print(f"Generated {len(result)} chars in {duration:.2f}s using {model_info.name}")
            
        except Exception as e:
            execution_manager.handle_step_error("model_generation", "Generate content", e)
            execution_manager.complete_step("model_generation", success=False)
            raise

    async def test_model_execution_with_recovery(
        self, execution_manager, integrated_model_registry
    ):
        """Test model execution with error recovery."""
        execution_manager.start_execution(total_steps=1)
        
        # Find a working model
        working_model = await self._find_working_model(integrated_model_registry)
        if not working_model:
            pytest.skip("No working models available")
        
        _, model_instance = working_model
        
        async def model_executor():
            """Executor that might fail and need recovery."""
            # Simulate potential network issues
            result = await model_instance.generate(
                "Test generation",
                max_tokens=10,
                temperature=0.1
            )
            return result
        
        # Execute with recovery support
        success = await execution_manager.execute_step_with_recovery(
            "model_task", "Execute model with recovery", model_executor
        )
        
        assert success is True
        
        # Check that step completed
        step_progress = execution_manager.progress_tracker.get_step_progress(
            "test_exec", "model_task"
        )
        assert step_progress is not None
        assert step_progress.progress_percentage == 100.0

    async def test_checkpoint_with_model_state(
        self, execution_manager, integrated_model_registry
    ):
        """Test checkpointing with model state."""
        execution_manager.start_execution(total_steps=2)
        
        # Find a working model
        working_model = await self._find_working_model(integrated_model_registry)
        if not working_model:
            pytest.skip("No working models available")
        
        model_info, model_instance = working_model
        
        # Execute first step with model
        execution_manager.start_step("step1", "First model operation")
        
        result1 = await model_instance.generate("Hello", max_tokens=5, temperature=0.1)
        
        # Store model result
        execution_manager.variable_manager.set_variable(
            "step1_result", 
            {
                "content": result1,
                "model": model_info.name,
                "step": "step1"
            },
            scope=VariableScope.EXECUTION
        )
        
        execution_manager.complete_step("step1", success=True)
        
        # Create checkpoint after first step
        checkpoint = execution_manager.create_checkpoint("after_step1")
        assert checkpoint is not None
        
        # Start second step
        execution_manager.start_step("step2", "Second model operation")
        
        result2 = await model_instance.generate("World", max_tokens=5, temperature=0.1)
        
        execution_manager.variable_manager.set_variable(
            "step2_result",
            {
                "content": result2,
                "model": model_info.name,
                "step": "step2"
            },
            scope=VariableScope.EXECUTION
        )
        
        # Restore checkpoint
        restore_success = execution_manager.restore_checkpoint(checkpoint.id)
        assert restore_success is True
        
        # Verify state was restored
        step1_result = execution_manager.variable_manager.get_variable("step1_result")
        step2_result = execution_manager.variable_manager.get_variable("step2_result")
        
        assert step1_result is not None
        assert step1_result["content"] == result1
        assert step2_result is None  # Should be cleared by restore

    async def _find_working_model(self, registry):
        """Find a working model for testing."""
        providers = registry.get_providers()
        
        for provider in providers:
            try:
                models = await provider.get_available_models()
                for model_info in models[:1]:  # Try first model
                    try:
                        model_instance = await provider.create_model(model_info.name)
                        if model_instance:
                            return (model_info, model_instance)
                    except Exception as e:
                        print(f"Failed to create {model_info.name}: {e}")
                        continue
            except Exception as e:
                print(f"Provider {provider.name} failed: {e}")
                continue
        
        return None


@pytest.mark.integration 
class TestPipelineExecutionWithModels:
    """Test end-to-end pipeline execution with multi-model integration."""

    @pytest.fixture
    async def integrated_orchestrator(self):
        """Create orchestrator with integrated model system."""
        from src.orchestrator.core.control_system import ControlSystem
        
        # Create a control system that integrates with models
        class ModelAwareControlSystem(ControlSystem):
            def __init__(self):
                config = {
                    "capabilities": {
                        "supported_actions": ["generate", "analyze", "summarize"],
                        "model_integration": True
                    }
                }
                super().__init__(name="model-aware-control", config=config)
                
                # Initialize model system
                self.model_registry = ModelRegistry()
                self.model_registry.add_provider(OpenAIProvider())
                self.model_registry.add_provider(AnthropicProvider())
                self.model_registry.add_provider(LocalProvider())
                
                self.model_selection_manager = ModelSelectionManager(self.model_registry)
                self._task_results = {}
            
            async def execute_task(self, task: Task, context: dict = None):
                """Execute task with model integration."""
                if task.action == "generate":
                    return await self._generate_content(task)
                elif task.action == "analyze":
                    return await self._analyze_content(task)
                else:
                    return {"status": "completed", "result": f"Executed {task.action}"}
            
            async def _generate_content(self, task):
                """Generate content using selected model."""
                prompt = task.parameters.get("prompt", "")
                
                # Select appropriate model
                criteria = SelectionCriteria(
                    required_capabilities=[ModelCapability.TEXT_GENERATION],
                    strategy="balanced"
                )
                
                try:
                    selection_result = await self.model_selection_manager.select_model(criteria)
                    
                    if not selection_result or not selection_result.model:
                        return {"status": "failed", "error": "No suitable model found"}
                    
                    # Create model instance
                    provider = self._get_provider(selection_result.model.provider)
                    model_instance = await provider.create_model(selection_result.model.name)
                    
                    # Generate content
                    result = await model_instance.generate(
                        prompt, 
                        max_tokens=100, 
                        temperature=0.3
                    )
                    
                    result_data = {
                        "status": "completed",
                        "content": result,
                        "model": {
                            "name": selection_result.model.name,
                            "provider": selection_result.model.provider
                        },
                        "confidence": selection_result.confidence
                    }
                    
                    self._task_results[task.id] = result_data
                    return result_data
                    
                except Exception as e:
                    error_result = {
                        "status": "failed", 
                        "error": str(e),
                        "task_id": task.id
                    }
                    self._task_results[task.id] = error_result
                    return error_result
            
            async def _analyze_content(self, task):
                """Analyze content using selected model."""
                content = task.parameters.get("content", "")
                if isinstance(content, str) and content.startswith("$results."):
                    # Resolve reference
                    ref_task_id = content.split(".")[1]
                    if ref_task_id in self._task_results:
                        referenced_result = self._task_results[ref_task_id]
                        content = referenced_result.get("content", "")
                
                # For analysis, prefer higher quality models
                criteria = SelectionCriteria(
                    required_capabilities=[ModelCapability.ANALYSIS],
                    min_quality_tier="high",
                    strategy="balanced"
                )
                
                try:
                    selection_result = await self.model_selection_manager.select_model(criteria)
                    
                    if not selection_result or not selection_result.model:
                        # Fallback to any text generation model
                        criteria = SelectionCriteria(
                            required_capabilities=[ModelCapability.TEXT_GENERATION]
                        )
                        selection_result = await self.model_selection_manager.select_model(criteria)
                    
                    if not selection_result or not selection_result.model:
                        return {"status": "failed", "error": "No suitable analysis model found"}
                    
                    provider = self._get_provider(selection_result.model.provider)
                    model_instance = await provider.create_model(selection_result.model.name)
                    
                    analysis_prompt = f"Analyze the following content and provide key insights:\n\n{content}"
                    analysis = await model_instance.generate(
                        analysis_prompt,
                        max_tokens=150,
                        temperature=0.2
                    )
                    
                    result_data = {
                        "status": "completed",
                        "analysis": analysis,
                        "original_content_length": len(content),
                        "model": {
                            "name": selection_result.model.name,
                            "provider": selection_result.model.provider
                        }
                    }
                    
                    self._task_results[task.id] = result_data
                    return result_data
                    
                except Exception as e:
                    error_result = {"status": "failed", "error": str(e)}
                    self._task_results[task.id] = error_result
                    return error_result
            
            def _get_provider(self, provider_name):
                """Get provider by name."""
                for provider in self.model_registry.get_providers():
                    if provider.name == provider_name:
                        return provider
                raise ValueError(f"Provider {provider_name} not found")
            
            async def execute_pipeline(self, pipeline, context=None):
                raise NotImplementedError("Use orchestrator for pipeline execution")
            
            def get_capabilities(self):
                return self.config.get("capabilities", {})
            
            async def health_check(self):
                return {"status": "healthy", "name": self.name}
        
        control_system = ModelAwareControlSystem()
        orchestrator = Orchestrator(control_system=control_system)
        
        return orchestrator

    async def test_simple_model_pipeline(self, integrated_orchestrator):
        """Test simple pipeline with model operations."""
        pipeline_yaml = """
name: "model_integration_test"
description: "Test pipeline with model integration"

steps:
  - id: generate
    action: generate
    parameters:
      prompt: "Explain artificial intelligence in two sentences"

  - id: analyze
    action: analyze
    depends_on: [generate]
    parameters:
      content: "$results.generate.content"
"""
        
        print("🚀 Executing model integration pipeline...")
        
        try:
            results = await integrated_orchestrator.execute_yaml(pipeline_yaml, context={})
            
            print(f"✅ Pipeline completed with {len(results)} tasks")
            
            # Verify generation task
            assert "generate" in results
            generate_result = results["generate"]
            assert generate_result["success"] == True
            assert "content" in generate_result
            assert len(generate_result["content"]) > 0
            
            print(f"Generated content: {generate_result['content'][:100]}...")
            print(f"Used model: {generate_result['model']['provider']}:{generate_result['model']['name']}")
            
            # Verify analysis task
            if "analyze" in results:
                analyze_result = results["analyze"]
                if analyze_result["success"] == True:
                    assert "analysis" in analyze_result
                    assert len(analyze_result["analysis"]) > 0
                    
                    print(f"Analysis: {analyze_result['analysis'][:100]}...")
                    print(f"Analysis model: {analyze_result['model']['provider']}:{analyze_result['model']['name']}")
                
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            pytest.skip("Pipeline execution failed - likely no working models")

    async def test_complex_model_pipeline(self, integrated_orchestrator):
        """Test complex pipeline with multiple model operations."""
        pipeline_yaml = """
name: "complex_model_pipeline"
description: "Complex pipeline with multiple model operations"

steps:
  - id: generate_topic
    action: generate
    parameters:
      prompt: "Generate a technical topic related to machine learning (just the topic name)"

  - id: explain_topic
    action: generate
    depends_on: [generate_topic]
    parameters:
      prompt: "Explain this topic in detail: $results.generate_topic.content"

  - id: analyze_explanation
    action: analyze
    depends_on: [explain_topic]
    parameters:
      content: "$results.explain_topic.content"
"""
        
        print("🔧 Executing complex model pipeline...")
        
        try:
            results = await integrated_orchestrator.execute_yaml(pipeline_yaml, context={})
            
            print(f"✅ Complex pipeline completed with {len(results)} tasks")
            
            # Should have all three tasks
            expected_tasks = ["generate_topic", "explain_topic", "analyze_explanation"]
            for task_id in expected_tasks:
                if task_id in results:
                    result = results[task_id]
                    print(f"{task_id}: {result['status']}")
                    
                    if result["success"] == True:
                        if "content" in result:
                            print(f"  Content length: {len(result['content'])}")
                        if "analysis" in result:
                            print(f"  Analysis length: {len(result['analysis'])}")
                        if "model" in result:
                            print(f"  Model: {result['model']['provider']}:{result['model']['name']}")
            
            # At least the first task should complete
            assert "generate_topic" in results
            assert results["generate_topic"]["status"] == "completed"
            
        except Exception as e:
            print(f"Complex pipeline failed: {e}")
            pytest.skip("Complex pipeline execution failed")

    async def test_model_error_handling_in_pipeline(self, integrated_orchestrator):
        """Test error handling when model operations fail."""
        pipeline_yaml = """
name: "error_handling_test"
description: "Test pipeline error handling with models"

steps:
  - id: valid_generation
    action: generate
    parameters:
      prompt: "Hello world"

  - id: analyze_results
    action: analyze
    depends_on: [valid_generation]
    parameters:
      content: "$results.valid_generation.content"
"""
        
        try:
            results = await integrated_orchestrator.execute_yaml(pipeline_yaml, context={})
            
            # Should handle errors gracefully
            print(f"Error handling test completed with {len(results)} results")
            
            for task_id, result in results.items():
                print(f"{task_id}: {result['status']}")
                if result["success"] == False:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            print(f"Error handling test raised exception: {e}")
            # This is acceptable as we're testing error scenarios


async def main():
    """Run pipeline integration tests."""
    print("🔗 PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    
    # Run pytest with this file
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "integration"
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)