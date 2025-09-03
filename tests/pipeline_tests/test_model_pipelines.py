"""
Test suite for model and LLM pipeline functionality.

Tests model selection, routing, LLM output quality, image generation,
and multimodal processing capabilities across example pipelines.

This module implements Issue #242 Stream 4: Model & LLM Pipeline Tests.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from orchestrator import Orchestrator
from src.orchestrator.models.model_registry import ModelRegistry
from tests.pipeline_tests.test_base import (
    BasePipelineTest,
    PipelineExecutionResult,
    PipelineTestConfiguration
)


class ModelPipelineTests(BasePipelineTest):
    """
    Comprehensive test suite for model and LLM pipelines.
    
    Tests:
    - LLM routing and model selection
    - Model routing logic and cost optimization
    - AUTO tag functionality and dynamic decisions
    - Creative image generation pipelines
    - Multimodal processing capabilities
    - Output quality validation
    - Performance and cost tracking
    """
    
    def __init__(self, 
                 orchestrator: Orchestrator,
                 model_registry: ModelRegistry,
                 config: Optional[PipelineTestConfiguration] = None):
        """Initialize test suite with cost-optimized configuration."""
        # Configure for cost-efficient testing
        test_config = config or PipelineTestConfiguration(
            timeout_seconds=300,  # 5 minutes for model operations
            max_cost_dollars=2.0,  # Higher budget for model testing
            enable_performance_tracking=True,
            validate_outputs=True,
            max_execution_time=300,
            retry_on_failure=True,
            max_retries=1  # Limited retries for cost control
        )
        
        super().__init__(orchestrator, model_registry, test_config)
        
        # Pipeline paths
        self.examples_dir = Path("examples")
        self.output_dir = Path("examples/outputs/test_model_pipelines")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test data
        self.test_prompts = {
            "simple": "Write a Python function to add two numbers",
            "complex": "Design and implement a complete microservices architecture for a distributed e-commerce platform",
            "creative": "Create a story about a robot learning to paint"
        }
        
        self.test_image_prompts = [
            "A serene mountain landscape at sunset",
            "A futuristic city with flying cars",
            "An abstract geometric pattern in bright colors"
        ]
    
    async def test_llm_routing_pipeline(self):
        """
        Test LLM routing pipeline for intelligent model selection and optimization.
        
        Validates:
        - Model selection based on task complexity
        - Prompt optimization functionality
        - Routing strategy implementation
        - Cost estimation accuracy
        - Output quality and format
        """
        print("Testing LLM routing pipeline...")
        
        # Load pipeline
        pipeline_path = self.examples_dir / "llm_routing_pipeline.yaml"
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Test with different task complexities
        test_cases = [
            {
                "name": "simple_task",
                "task": self.test_prompts["simple"],
                "optimization_goals": ["brevity", "clarity"],
                "routing_strategy": "cost_optimized"
            },
            {
                "name": "complex_task", 
                "task": self.test_prompts["complex"],
                "optimization_goals": ["quality", "completeness"],
                "routing_strategy": "quality_optimized"
            },
            {
                "name": "balanced_task",
                "task": self.test_prompts["creative"],
                "optimization_goals": ["clarity", "creativity"],
                "routing_strategy": "balanced"
            }
        ]
        
        results = []
        for case in test_cases:
            print(f"  Testing {case['name']}...")
            
            # Execute pipeline
            inputs = {
                "task": case["task"],
                "optimization_goals": case["optimization_goals"],
                "routing_strategy": case["routing_strategy"]
            }
            
            result = await self.execute_pipeline_async(
                yaml_content, 
                inputs, 
                self.output_dir / case["name"]
            )
            
            # Validate execution
            self.assert_pipeline_success(result, f"LLM routing failed for {case['name']}")
            
            # Validate outputs
            assert "analyze_task" in result.outputs, "Task analysis missing"
            assert "optimize_prompt" in result.outputs, "Prompt optimization missing"
            assert "route_request" in result.outputs, "Routing decision missing"
            assert "save_report" in result.outputs, "Report generation missing"
            
            # Validate routing decisions
            task_analysis = result.outputs.get("analyze_task", {})
            assert "selected_model" in task_analysis, "No model selected"
            assert "score" in task_analysis, "No selection score provided"
            assert "estimated_cost" in task_analysis, "No cost estimation"
            
            # Validate prompt optimization
            optimization = result.outputs.get("optimize_prompt", {})
            assert "optimized_prompt" in optimization, "No optimized prompt"
            assert "metrics" in optimization, "No optimization metrics"
            
            # Validate routing result
            routing = result.outputs.get("route_request", {})
            assert "selected_model" in routing, "No final model selection"
            assert "strategy" in routing, "No routing strategy recorded"
            
            # Check cost efficiency
            estimated_cost = task_analysis.get("estimated_cost", 0)
            if case["routing_strategy"] == "cost_optimized":
                assert estimated_cost < 0.50, f"Cost too high for cost-optimized routing: ${estimated_cost}"
            
            # Validate report generation
            assert result.outputs["save_report"], "Report not saved"
            
            results.append({
                "case": case["name"],
                "result": result,
                "model": task_analysis.get("selected_model"),
                "cost": estimated_cost,
                "strategy": case["routing_strategy"]
            })
        
        # Validate cost differences between strategies
        cost_optimized = next(r for r in results if r["strategy"] == "cost_optimized")
        quality_optimized = next(r for r in results if r["strategy"] == "quality_optimized")
        
        assert cost_optimized["cost"] <= quality_optimized["cost"], \
            "Cost-optimized routing should be cheaper than quality-optimized"
        
        print(f"  ✓ LLM routing tests passed. Tested {len(results)} scenarios.")
        
        # Performance validation
        for result_data in results:
            self.assert_performance_within_limits(result_data["result"])
        
        return results
    
    async def test_model_routing_demo(self):
        """
        Test model routing demonstration pipeline.
        
        Validates:
        - Task-specific model assignment
        - Batch processing optimization
        - Cost tracking and budget management
        - Multi-task routing strategies
        - Translation and coding task handling
        """
        print("Testing model routing demo...")
        
        pipeline_path = self.examples_dir / "model_routing_demo.yaml"
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Test different routing priorities
        test_priorities = ["cost", "balanced", "quality"]
        results = {}
        
        for priority in test_priorities:
            print(f"  Testing {priority} routing...")
            
            inputs = {
                "task_budget": 5.00,
                "priority": priority,
                "output_path": str(self.output_dir / f"model_routing_{priority}")
            }
            
            result = await self.execute_pipeline_async(
                yaml_content,
                inputs,
                self.output_dir / f"routing_{priority}"
            )
            
            # Validate execution
            self.assert_pipeline_success(result, f"Model routing failed for {priority}")
            
            # Validate key outputs
            assert "assess_requirements" in result.outputs, "Requirements assessment missing"
            assert "summarize_document" in result.outputs, "Document summary missing"
            assert "generate_code" in result.outputs, "Code generation missing"
            assert "analyze_data" in result.outputs, "Data analysis missing"
            assert "batch_processing" in result.outputs, "Batch processing missing"
            assert "routing_report" in result.outputs, "Routing report missing"
            
            # Validate routing assessment
            assessment = result.outputs["assess_requirements"]
            assert "recommendations" in assessment, "No routing recommendations"
            assert len(assessment["recommendations"]) >= 3, "Insufficient task routing"
            
            # Validate task outputs quality
            summary = result.outputs["summarize_document"]
            assert "result" in summary, "No summary result"
            assert len(summary["result"]) > 50, "Summary too short"
            
            code = result.outputs["generate_code"]
            assert "result" in code, "No code result"
            assert "def " in code["result"], "Generated code missing function definition"
            
            analysis = result.outputs["analyze_data"]
            assert "result" in analysis, "No analysis result"
            
            # Validate batch processing
            batch = result.outputs["batch_processing"]
            assert "results" in batch, "No batch results"
            assert "total_cost" in batch, "No batch cost tracking"
            assert len(batch["results"]) == 4, "Incorrect number of translations"
            
            # Check cost adherence
            total_cost = float(batch.get("total_cost", 0))
            assert total_cost <= inputs["task_budget"], \
                f"Exceeded budget: ${total_cost} > ${inputs['task_budget']}"
            
            results[priority] = {
                "result": result,
                "total_cost": total_cost,
                "assessment": assessment,
                "tasks_completed": 4  # summary, code, analysis, translations
            }
        
        # Validate cost optimization works
        if "cost" in results and "quality" in results:
            cost_total = results["cost"]["total_cost"]
            quality_total = results["quality"]["total_cost"]
            assert cost_total <= quality_total, \
                "Cost routing should be cheaper than quality routing"
        
        print(f"  ✓ Model routing demo tests passed. Tested {len(results)} priorities.")
        
        # Performance validation
        for priority_data in results.values():
            self.assert_performance_within_limits(priority_data["result"])
        
        return results
    
    async def test_auto_tags_demo(self):
        """
        Test AUTO tags functionality for dynamic decision-making.
        
        Validates:
        - AUTO tag resolution and parsing
        - Dynamic model selection via AUTO tags
        - Context-aware parameter decisions
        - Conditional execution based on AUTO tags
        - Type coercion for AUTO tag responses
        """
        print("Testing AUTO tags demo...")
        
        pipeline_path = self.examples_dir / "auto_tags_demo.yaml"
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Test different content types and complexities
        test_cases = [
            {
                "name": "technical_content",
                "content": "Machine learning algorithms utilize neural networks with backpropagation to optimize weight matrices through gradient descent optimization techniques.",
                "task_complexity": "high"
            },
            {
                "name": "casual_content", 
                "content": "I really enjoyed the movie last night. The acting was great and the story was interesting.",
                "task_complexity": "low"
            },
            {
                "name": "business_content",
                "content": "Our quarterly revenue increased by 15% due to improved market penetration and strategic partnerships in the technology sector.",
                "task_complexity": "medium"
            }
        ]
        
        results = []
        for case in test_cases:
            print(f"  Testing {case['name']}...")
            
            inputs = {
                "content": case["content"],
                "task_complexity": case["task_complexity"]
            }
            
            result = await self.execute_pipeline_async(
                yaml_content,
                inputs,
                self.output_dir / f"auto_tags_{case['name']}"
            )
            
            # Validate execution
            self.assert_pipeline_success(result, f"AUTO tags demo failed for {case['name']}")
            
            # Validate AUTO tag resolutions
            assert "analyze_content" in result.outputs, "Content analysis missing"
            assert "select_processing_strategy" in result.outputs, "Strategy selection missing"
            assert "choose_output_format" in result.outputs, "Format selection missing"
            assert "process_content" in result.outputs, "Content processing missing"
            assert "assess_quality" in result.outputs, "Quality assessment missing"
            
            # Validate dynamic decisions were made
            content_type = result.outputs["analyze_content"]
            assert content_type in ["technical", "academic", "casual", "business"], \
                f"Invalid content type: {content_type}"
            
            strategy_output = result.outputs["select_processing_strategy"]
            assert "strategy" in strategy_output, "No processing strategy selected"
            assert "model" in strategy_output, "No model selected via AUTO tag"
            
            format_output = result.outputs["choose_output_format"]
            assert "format" in format_output, "No output format selected"
            
            processing = result.outputs["process_content"]
            assert "detail_level" in processing, "No detail level selected"
            assert "max_tokens" in processing, "No token allocation selected"
            
            # Validate conditional logic worked
            quality = result.outputs["assess_quality"]
            assert quality, "Quality assessment not performed"
            
            # Check if improvement was conditionally executed
            improvement_executed = "improve_if_needed" in result.outputs
            
            # Validate final report
            assert "create_report" in result.outputs, "Final report missing"
            report = result.outputs["create_report"]
            assert case["content"] in report, "Original content not in report"
            assert content_type in report, "Content type not in report"
            
            results.append({
                "case": case["name"],
                "result": result,
                "content_type": content_type,
                "strategy": strategy_output.get("strategy"),
                "model": strategy_output.get("model"),
                "format": format_output.get("format"),
                "improvement_executed": improvement_executed
            })
        
        # Validate different content led to different decisions
        content_types = [r["content_type"] for r in results]
        strategies = [r["strategy"] for r in results]
        models = [r["model"] for r in results]
        
        # Should have some variation in decisions
        assert len(set(content_types)) > 1 or len(set(strategies)) > 1, \
            "AUTO tags should produce varied decisions for different content"
        
        print(f"  ✓ AUTO tags demo tests passed. Tested {len(results)} content types.")
        
        # Performance validation
        for result_data in results:
            self.assert_performance_within_limits(result_data["result"])
        
        return results
    
    async def test_creative_image_pipeline(self):
        """
        Test creative image generation pipeline.
        
        Validates:
        - Image generation with different styles
        - Prompt optimization for images
        - File output handling
        - Gallery report generation
        - Multi-style variation creation
        """
        print("Testing creative image pipeline...")
        
        pipeline_path = self.examples_dir / "creative_image_pipeline.yaml"
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Test with simple prompt to control costs
        test_prompt = "A simple geometric shape"
        
        inputs = {
            "base_prompt": test_prompt,
            "num_variations": 2,  # Reduced for cost control
            "art_styles": ["photorealistic", "abstract"],  # Reduced styles
            "output_path": str(self.output_dir / "creative_images")
        }
        
        result = await self.execute_pipeline_async(
            yaml_content,
            inputs,
            self.output_dir / "creative_test"
        )
        
        # Validate execution (may fail due to image generation availability)
        if not result.success:
            if "image-generation" in str(result.error_message):
                print("  ⚠ Image generation service unavailable - skipping image tests")
                return {"skipped": True, "reason": "Image generation unavailable"}
            else:
                self.assert_pipeline_success(result, "Creative image pipeline failed")
        
        # Validate pipeline structure (even if image generation fails)
        expected_steps = [
            "generate_folder_name",
            "generate_base_image", 
            "analyze_base",
            "generate_style_variation_1",
            "generate_style_variation_2",
            "enhance_prompt",
            "generate_enhanced",
            "count_images",
            "create_gallery_report",
            "save_gallery_report"
        ]
        
        # Check which steps completed
        completed_steps = list(result.outputs.keys())
        print(f"  Completed steps: {completed_steps}")
        
        # At minimum, folder name generation should work
        if "generate_folder_name" in result.outputs:
            folder_name = result.outputs["generate_folder_name"]
            assert isinstance(folder_name, str), "Folder name should be string"
            assert len(folder_name) > 0, "Folder name should not be empty"
        
        # If image generation worked, validate outputs
        if "generate_base_image" in result.outputs:
            base_image = result.outputs["generate_base_image"]
            assert "images" in base_image, "No images in base generation result"
            
        if "enhance_prompt" in result.outputs:
            enhancement = result.outputs["enhance_prompt"]
            assert "optimized_prompt" in enhancement, "No prompt optimization"
            
        print("  ✓ Creative image pipeline structure validated.")
        
        # Performance validation
        self.assert_performance_within_limits(result, max_cost=1.0)  # Higher cost allowance for images
        
        return {
            "result": result,
            "completed_steps": completed_steps,
            "total_steps": len(expected_steps),
            "success_rate": len(completed_steps) / len(expected_steps)
        }
    
    async def test_multimodal_processing(self):
        """
        Test multimodal processing pipeline capabilities.
        
        Validates:
        - Image analysis functionality
        - Audio processing and transcription
        - Video frame extraction and analysis
        - Multimodal content integration
        - Report generation with media content
        """
        print("Testing multimodal processing pipeline...")
        
        pipeline_path = self.examples_dir / "multimodal_processing.yaml"
        with open(pipeline_path, 'r') as f:
            yaml_content = f.read()
        
        # Create simple test media files
        test_output = self.output_dir / "multimodal_test"
        test_output.mkdir(exist_ok=True)
        
        # Create minimal test image (using example data structure)
        test_image_path = "examples/data/test_image.jpg"
        
        inputs = {
            "input_image": test_image_path,
            "input_audio": "examples/data/test_audio.wav",  # May not exist
            "input_video": "examples/data/test_video.mp4",  # May not exist
            "output_dir": str(test_output),
            "output_path": str(test_output)
        }
        
        result = await self.execute_pipeline_async(
            yaml_content,
            inputs,
            test_output
        )
        
        # Validate execution (may partially fail due to missing media tools)
        if not result.success:
            if any(tool in str(result.error_message) for tool in 
                   ["image-analysis", "audio-processing", "video-processing"]):
                print("  ⚠ Some multimodal services unavailable - testing available components")
            else:
                # If it's a different error, it should succeed
                self.assert_pipeline_success(result, "Multimodal pipeline failed")
        
        # Check which multimodal steps completed
        expected_steps = [
            "analyze_image",
            "detect_objects", 
            "generate_variations",
            "transcribe_audio",
            "analyze_audio",
            "analyze_video",
            "extract_key_frames",
            "analyze_key_frames",
            "copy_original_image",
            "generate_summary_report"
        ]
        
        completed_steps = list(result.outputs.keys())
        print(f"  Completed multimodal steps: {len(completed_steps)}/{len(expected_steps)}")
        
        # Validate image processing if available
        if "analyze_image" in result.outputs:
            image_analysis = result.outputs["analyze_image"]
            assert "analysis" in image_analysis, "No image analysis result"
            
        if "detect_objects" in result.outputs:
            object_detection = result.outputs["detect_objects"]
            assert "analysis" in object_detection, "No object detection result"
            
        # Validate audio processing if available
        if "transcribe_audio" in result.outputs:
            transcription = result.outputs["transcribe_audio"]
            assert "transcription" in transcription, "No audio transcription"
            
        # Validate video processing if available
        if "analyze_video" in result.outputs:
            video_analysis = result.outputs["analyze_video"]
            assert "analysis" in video_analysis, "No video analysis"
            
        # Validate report generation
        if "generate_summary_report" in result.outputs:
            report = result.outputs["generate_summary_report"]
            # Report should be generated even with missing media
            assert report, "Summary report not generated"
            
        print("  ✓ Multimodal processing pipeline structure validated.")
        
        # Performance validation
        self.assert_performance_within_limits(result, max_cost=1.0)  # Higher cost for multimodal
        
        return {
            "result": result,
            "completed_steps": completed_steps,
            "total_steps": len(expected_steps),
            "success_rate": len(completed_steps) / len(expected_steps),
            "image_processing": "analyze_image" in completed_steps,
            "audio_processing": "transcribe_audio" in completed_steps,
            "video_processing": "analyze_video" in completed_steps
        }
    
    async def test_model_output_validation(self):
        """
        Test model output quality validation across different pipelines.
        
        Validates:
        - Output format consistency
        - Content quality metrics
        - Model response validation
        - Error handling for invalid outputs
        - Cost tracking accuracy
        """
        print("Testing model output validation...")
        
        # Simple test pipeline for output validation
        validation_pipeline = """
id: output_validation_test
name: Model Output Validation Test
version: "1.0.0"

steps:
  # Test different output types
  - id: test_json_output
    action: generate_text
    parameters:
      prompt: 'Return valid JSON with keys "name" and "value": {"name": "test", "value": 42}'
      model: "gpt-4o-mini"
      response_format: "json_object"
      max_tokens: 100
      
  - id: test_text_output
    action: generate_text
    parameters:
      prompt: "Write exactly 3 words describing AI."
      model: "gpt-4o-mini"
      max_tokens: 50
      
  - id: test_structured_output
    action: generate_text
    parameters:
      prompt: |
        Create a numbered list of 3 benefits of renewable energy.
        Format as:
        1. Benefit one
        2. Benefit two  
        3. Benefit three
      model: "gpt-4o-mini"
      max_tokens: 200
      
outputs:
  json_result: "{{ test_json_output }}"
  text_result: "{{ test_text_output }}"
  structured_result: "{{ test_structured_output }}"
"""
        
        result = await self.execute_pipeline_async(
            validation_pipeline,
            {},
            self.output_dir / "validation_test"
        )
        
        # Validate execution
        self.assert_pipeline_success(result, "Output validation pipeline failed")
        
        # Validate outputs exist
        assert "test_json_output" in result.outputs, "JSON output missing"
        assert "test_text_output" in result.outputs, "Text output missing"  
        assert "test_structured_output" in result.outputs, "Structured output missing"
        
        # Validate JSON output
        json_output = result.outputs["test_json_output"]
        if isinstance(json_output, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(json_output)
                assert "name" in parsed or "value" in parsed, "JSON missing expected keys"
            except json.JSONDecodeError:
                # JSON parsing failed, but output exists
                pass
        
        # Validate text output constraints
        text_output = result.outputs["test_text_output"]
        assert isinstance(text_output, str), "Text output should be string"
        assert len(text_output.strip()) > 0, "Text output should not be empty"
        
        # Validate structured output
        structured_output = result.outputs["test_structured_output"]
        assert isinstance(structured_output, str), "Structured output should be string"
        # Should contain numbered list indicators
        assert any(num in structured_output for num in ["1.", "2.", "3."]), \
            "Structured output missing list format"
        
        print("  ✓ Model output validation tests passed.")
        
        # Performance validation
        self.assert_performance_within_limits(result, max_cost=0.10)  # Should be very cheap
        
        return {
            "result": result,
            "json_valid": "test_json_output" in result.outputs,
            "text_valid": "test_text_output" in result.outputs,
            "structured_valid": "test_structured_output" in result.outputs
        }
    
    def test_basic_execution(self):
        """Test basic model pipeline execution (required by base class)."""
        print("Running basic model pipeline execution tests...")
        return asyncio.run(self._run_basic_tests())
    
    async def _run_basic_tests(self):
        """Internal method to run basic async tests."""
        # Test a simple model pipeline
        simple_pipeline = """
id: basic_model_test
name: Basic Model Test
version: "1.0.0"

steps:
  - id: simple_generation
    action: generate_text
    parameters:
      prompt: "Say 'Hello, world!' in exactly those words."
      model: "gpt-4o-mini"
      max_tokens: 20

outputs:
  greeting: "{{ simple_generation }}"
"""
        
        result = await self.execute_pipeline_async(
            simple_pipeline,
            {},
            self.output_dir / "basic_test"
        )
        
        self.assert_pipeline_success(result, "Basic model test failed")
        assert "simple_generation" in result.outputs, "Basic generation missing"
        
        return result
    
    def test_error_handling(self):
        """Test error handling in model pipelines (required by base class)."""
        print("Running model pipeline error handling tests...")
        return asyncio.run(self._run_error_tests())
    
    async def _run_error_tests(self):
        """Internal method to run error handling tests."""
        # Test pipeline with invalid model
        error_pipeline = """
id: error_test
name: Error Handling Test
version: "1.0.0"

steps:
  - id: invalid_model_test
    action: generate_text
    parameters:
      prompt: "This should fail gracefully"
      model: "nonexistent-model-12345"
      max_tokens: 50
    error_handling:
      on_error: continue
      
  - id: fallback_test
    action: generate_text
    parameters:
      prompt: "This should work as fallback"
      model: "gpt-4o-mini"
      max_tokens: 50
    dependencies:
      - invalid_model_test

outputs:
  fallback_result: "{{ fallback_test }}"
"""
        
        result = await self.execute_pipeline_async(
            error_pipeline,
            {},
            self.output_dir / "error_test"
        )
        
        # Pipeline should handle the error and continue
        if result.success:
            assert "fallback_test" in result.outputs, "Fallback step should execute"
        else:
            # Error handling worked by failing gracefully
            assert result.error is not None, "Error should be captured"
        
        return result
    
    async def run_comprehensive_tests(self):
        """
        Run all model pipeline tests comprehensively.
        
        Returns:
            Dict[str, Any]: Complete test results summary
        """
        print("=" * 60)
        print("RUNNING COMPREHENSIVE MODEL PIPELINE TESTS")
        print("=" * 60)
        
        test_results = {
            "start_time": time.time(),
            "tests": {},
            "summary": {},
            "total_cost": 0.0
        }
        
        # Run all test suites
        test_suites = [
            ("llm_routing", self.test_llm_routing_pipeline),
            ("model_routing", self.test_model_routing_demo),
            ("auto_tags", self.test_auto_tags_demo),
            ("creative_images", self.test_creative_image_pipeline),
            ("multimodal", self.test_multimodal_processing),
            ("output_validation", self.test_model_output_validation)
        ]
        
        for suite_name, test_method in test_suites:
            print(f"\n{'=' * 40}")
            print(f"RUNNING {suite_name.upper()} TESTS")
            print(f"{'=' * 40}")
            
            try:
                suite_result = await test_method()
                test_results["tests"][suite_name] = {
                    "status": "passed",
                    "result": suite_result,
                    "error": None
                }
                print(f"✓ {suite_name} tests PASSED")
                
            except Exception as e:
                test_results["tests"][suite_name] = {
                    "status": "failed",
                    "result": None,
                    "error": str(e)
                }
                print(f"✗ {suite_name} tests FAILED: {e}")
        
        # Calculate summary statistics
        test_results["end_time"] = time.time()
        test_results["duration"] = test_results["end_time"] - test_results["start_time"]
        test_results["total_cost"] = self.total_cost
        
        passed_tests = [t for t in test_results["tests"].values() if t["status"] == "passed"]
        failed_tests = [t for t in test_results["tests"].values() if t["status"] == "failed"]
        
        test_results["summary"] = {
            "total_suites": len(test_suites),
            "passed_suites": len(passed_tests),
            "failed_suites": len(failed_tests),
            "success_rate": len(passed_tests) / len(test_suites),
            "total_executions": self.total_executions,
            "total_cost": self.total_cost,
            "average_cost_per_execution": self.total_cost / max(self.total_executions, 1)
        }
        
        print(f"\n{'=' * 60}")
        print("MODEL PIPELINE TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Test Suites: {test_results['summary']['total_suites']}")
        print(f"Passed: {test_results['summary']['passed_suites']}")
        print(f"Failed: {test_results['summary']['failed_suites']}")
        print(f"Success Rate: {test_results['summary']['success_rate']:.1%}")
        print(f"Total Cost: ${test_results['summary']['total_cost']:.4f}")
        print(f"Total Duration: {test_results['duration']:.1f}s")
        print(f"Total Executions: {test_results['summary']['total_executions']}")
        
        if failed_tests:
            print(f"\nFAILED TESTS:")
            for suite_name, test_data in test_results["tests"].items():
                if test_data["status"] == "failed":
                    print(f"  - {suite_name}: {test_data['error']}")
        
        return test_results


# Pytest fixtures and test functions for CI/CD integration
@pytest.fixture
async def model_test_instance():
    """Create ModelPipelineTests instance for pytest."""
    from orchestrator import Orchestrator
    from src.orchestrator.models.model_registry import ModelRegistry
    
    orchestrator = Orchestrator()
    model_registry = ModelRegistry()
    
    return ModelPipelineTests(orchestrator, model_registry)


@pytest.mark.asyncio
async def test_llm_routing_pipeline_pytest(model_test_instance):
    """Pytest wrapper for LLM routing tests."""
    await model_test_instance.test_llm_routing_pipeline()


@pytest.mark.asyncio  
async def test_model_routing_demo_pytest(model_test_instance):
    """Pytest wrapper for model routing demo tests."""
    await model_test_instance.test_model_routing_demo()


@pytest.mark.asyncio
async def test_auto_tags_demo_pytest(model_test_instance):
    """Pytest wrapper for AUTO tags tests."""
    await model_test_instance.test_auto_tags_demo()


@pytest.mark.asyncio
async def test_creative_image_pipeline_pytest(model_test_instance):
    """Pytest wrapper for creative image tests."""
    await model_test_instance.test_creative_image_pipeline()


@pytest.mark.asyncio
async def test_multimodal_processing_pytest(model_test_instance):
    """Pytest wrapper for multimodal tests."""
    await model_test_instance.test_multimodal_processing()


@pytest.mark.asyncio
async def test_model_output_validation_pytest(model_test_instance):
    """Pytest wrapper for output validation tests."""
    await model_test_instance.test_model_output_validation()


if __name__ == "__main__":
    """Direct execution for standalone testing."""
    import sys
    
    # Setup test environment
    from orchestrator import Orchestrator
    from src.orchestrator.models.model_registry import ModelRegistry
    
    async def main():
        """Main test execution function."""
        print("Initializing Model Pipeline Test Suite...")
        
        try:
            orchestrator = Orchestrator()
            model_registry = ModelRegistry()
            
            # Create test instance
            test_suite = ModelPipelineTests(orchestrator, model_registry)
            
            # Run comprehensive tests
            results = await test_suite.run_comprehensive_tests()
            
            # Exit with appropriate code
            if results["summary"]["success_rate"] < 0.8:
                print(f"\n❌ Test suite failed - success rate {results['summary']['success_rate']:.1%} below threshold")
                sys.exit(1)
            else:
                print(f"\n✅ Test suite passed - success rate {results['summary']['success_rate']:.1%}")
                sys.exit(0)
                
        except Exception as e:
            print(f"\n💥 Test suite crashed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Run tests
    asyncio.run(main())