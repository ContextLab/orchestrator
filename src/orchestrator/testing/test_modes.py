"""
Test execution mode management with time-based optimization.

This module provides intelligent test mode selection and execution time optimization
for the pipeline testing infrastructure, enabling CI/CD integration with time constraints.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from statistics import mean, median

logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Test execution modes with different coverage and time characteristics."""
    
    QUICK = "quick"      # 5-10 critical pipelines, ~5-10 minutes
    CORE = "core"        # 15-20 essential pipelines, ~15-30 minutes  
    FULL = "full"        # All pipelines, ~60-120 minutes
    SINGLE = "single"    # Single pipeline, ~2-5 minutes
    SMOKE = "smoke"      # Ultra-fast validation, ~2-5 minutes
    REGRESSION = "regression"  # Performance-focused subset, ~20-40 minutes


@dataclass
class TestModeConfig:
    """Configuration for a specific test mode."""
    
    name: str
    description: str
    max_pipelines: Optional[int]
    target_time_minutes: int
    max_time_minutes: int
    priority_categories: List[str]
    required_pipelines: Set[str] = field(default_factory=set)
    excluded_pipelines: Set[str] = field(default_factory=set)
    enable_quality_validation: bool = True
    enable_performance_monitoring: bool = True
    enable_regression_detection: bool = False
    fail_fast: bool = False


@dataclass 
class PipelineExecutionEstimate:
    """Execution time and resource estimates for a pipeline."""
    
    pipeline_name: str
    estimated_time_seconds: float
    confidence: float  # 0.0 to 1.0
    historical_samples: int
    complexity_factor: float
    category: str
    priority_score: float  # Higher = more important to test


@dataclass
class TestSuiteComposition:
    """Composition of a test suite for a specific mode."""
    
    mode: TestMode
    selected_pipelines: List[str]
    estimated_total_time_minutes: float
    estimated_cost_dollars: float
    coverage_percentage: float  # % of total pipelines
    priority_score: float  # Average priority of selected pipelines
    reasoning: List[str]  # Human-readable selection reasoning


class TestModeManager:
    """
    Manages test execution modes with intelligent pipeline selection and time optimization.
    
    Features:
    - Time-based test suite optimization
    - Pipeline execution time prediction
    - Smart pipeline selection algorithms
    - Historical performance analysis
    - CI/CD integration support
    """
    
    def __init__(self, performance_tracker=None):
        """Initialize test mode manager."""
        self.performance_tracker = performance_tracker
        self._mode_configs = self._initialize_mode_configs()
        self._execution_estimates = {}
        self._historical_data_loaded = False
        
    def _initialize_mode_configs(self) -> Dict[TestMode, TestModeConfig]:
        """Initialize test mode configurations."""
        return {
            TestMode.SMOKE: TestModeConfig(
                name="smoke",
                description="Ultra-fast smoke test for immediate feedback",
                max_pipelines=3,
                target_time_minutes=3,
                max_time_minutes=5,
                priority_categories=["data_processing"],
                required_pipelines={"simple_data_processing"},
                enable_quality_validation=False,
                enable_performance_monitoring=False,
                fail_fast=True
            ),
            
            TestMode.QUICK: TestModeConfig(
                name="quick",
                description="Quick validation for development workflow",
                max_pipelines=8,
                target_time_minutes=8,
                max_time_minutes=12,
                priority_categories=["data_processing", "research", "creative"],
                required_pipelines={
                    "simple_data_processing", 
                    "control_flow_conditional",
                    "research_minimal"
                },
                enable_quality_validation=True,
                enable_performance_monitoring=True,
                fail_fast=False
            ),
            
            TestMode.CORE: TestModeConfig(
                name="core", 
                description="Essential pipeline validation for CI/CD",
                max_pipelines=18,
                target_time_minutes=25,
                max_time_minutes=35,
                priority_categories=["data_processing", "research", "creative", "control_flow"],
                required_pipelines={
                    "simple_data_processing",
                    "data_processing_pipeline", 
                    "control_flow_conditional",
                    "control_flow_for_loop",
                    "research_minimal",
                    "creative_image_pipeline"
                },
                enable_regression_detection=True
            ),
            
            TestMode.REGRESSION: TestModeConfig(
                name="regression",
                description="Performance regression testing",
                max_pipelines=15,
                target_time_minutes=30,
                max_time_minutes=45,
                priority_categories=["data_processing", "research"],
                enable_performance_monitoring=True,
                enable_regression_detection=True,
                excluded_pipelines={"creative_image_pipeline"}  # Skip expensive creative tasks
            ),
            
            TestMode.FULL: TestModeConfig(
                name="full",
                description="Comprehensive validation for releases",
                max_pipelines=None,  # No limit
                target_time_minutes=90,
                max_time_minutes=150,
                priority_categories=["data_processing", "research", "creative", "control_flow"],
                enable_regression_detection=True
            )
        }
    
    def get_mode_config(self, mode: TestMode) -> TestModeConfig:
        """Get configuration for a specific test mode."""
        return self._mode_configs[mode]
    
    def load_historical_performance_data(self) -> bool:
        """Load historical performance data for execution time estimation."""
        if not self.performance_tracker:
            logger.warning("No performance tracker available for historical data")
            return False
            
        try:
            # Get all pipeline performance profiles
            all_pipelines = self.performance_tracker.get_all_pipeline_names()
            
            for pipeline_name in all_pipelines:
                profile = self.performance_tracker.get_pipeline_performance_profile(pipeline_name)
                
                if profile and profile.total_executions > 0:
                    # Calculate confidence based on sample size and consistency
                    confidence = min(1.0, profile.total_executions / 10.0)  # Max confidence at 10+ samples
                    if profile.execution_time_std > 0:
                        cv = profile.execution_time_std / profile.execution_time_mean
                        confidence *= max(0.1, 1.0 - cv)  # Reduce confidence for high variability
                    
                    self._execution_estimates[pipeline_name] = PipelineExecutionEstimate(
                        pipeline_name=pipeline_name,
                        estimated_time_seconds=profile.execution_time_mean,
                        confidence=confidence,
                        historical_samples=profile.total_executions,
                        complexity_factor=self._estimate_complexity_factor(pipeline_name),
                        category=self._get_pipeline_category(pipeline_name),
                        priority_score=self._calculate_priority_score(pipeline_name, profile)
                    )
                    
            self._historical_data_loaded = True
            logger.info(f"Loaded historical performance data for {len(self._execution_estimates)} pipelines")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load historical performance data: {e}")
            return False
    
    def _estimate_complexity_factor(self, pipeline_name: str) -> float:
        """Estimate complexity factor for a pipeline based on name and characteristics."""
        complexity_indicators = {
            "simple": 0.5,
            "basic": 0.6, 
            "minimal": 0.4,
            "advanced": 1.5,
            "complex": 2.0,
            "comprehensive": 2.5,
            "full": 2.0,
            "creative": 3.0,  # Creative tasks often take longer
            "image": 2.5,
            "multimodal": 3.0,
            "research": 1.5,
            "web": 2.0,
            "statistical": 1.8,
            "batch": 2.2
        }
        
        base_factor = 1.0
        name_lower = pipeline_name.lower()
        
        for indicator, factor in complexity_indicators.items():
            if indicator in name_lower:
                base_factor = max(base_factor, factor)
                
        return base_factor
    
    def _get_pipeline_category(self, pipeline_name: str) -> str:
        """Determine pipeline category from name."""
        if any(term in pipeline_name for term in ["data", "processing", "statistical", "csv"]):
            return "data_processing"
        elif any(term in pipeline_name for term in ["research", "web"]):
            return "research"  
        elif any(term in pipeline_name for term in ["creative", "image", "multimodal"]):
            return "creative"
        elif "control_flow" in pipeline_name:
            return "control_flow"
        else:
            return "data_processing"  # Default
            
    def _calculate_priority_score(self, pipeline_name: str, profile) -> float:
        """Calculate priority score for pipeline selection (higher = more important)."""
        base_score = 5.0
        
        # Core pipelines get higher priority
        if pipeline_name in {
            "simple_data_processing", "data_processing_pipeline", 
            "control_flow_conditional", "control_flow_for_loop",
            "research_minimal"
        }:
            base_score += 3.0
        
        # Frequently failing pipelines need testing
        if profile.success_rate < 0.9:
            base_score += 2.0
        elif profile.success_rate < 0.95:
            base_score += 1.0
            
        # Recently changed pipelines (if we had that data)
        # This would require integration with git history
        
        # Pipelines with performance issues
        if hasattr(profile, 'has_performance_issues') and profile.has_performance_issues:
            base_score += 1.5
            
        return base_score
    
    def estimate_pipeline_execution_time(self, pipeline_name: str) -> PipelineExecutionEstimate:
        """Estimate execution time for a specific pipeline."""
        if pipeline_name in self._execution_estimates:
            return self._execution_estimates[pipeline_name]
            
        # Fallback to heuristic-based estimation
        complexity_factor = self._estimate_complexity_factor(pipeline_name)
        category = self._get_pipeline_category(pipeline_name)
        
        # Base time estimates by category (in seconds)
        base_times = {
            "data_processing": 45,
            "research": 90,
            "creative": 180,
            "control_flow": 30
        }
        
        base_time = base_times.get(category, 60)
        estimated_time = base_time * complexity_factor
        
        return PipelineExecutionEstimate(
            pipeline_name=pipeline_name,
            estimated_time_seconds=estimated_time,
            confidence=0.3,  # Low confidence for heuristic estimates
            historical_samples=0,
            complexity_factor=complexity_factor,
            category=category,
            priority_score=5.0  # Default priority
        )
    
    def select_optimal_pipeline_suite(
        self, 
        mode: TestMode,
        available_pipelines: List[str],
        time_budget_minutes: Optional[int] = None
    ) -> TestSuiteComposition:
        """
        Select optimal pipeline suite for given mode and constraints.
        
        Args:
            mode: Test mode configuration
            available_pipelines: List of available pipeline names
            time_budget_minutes: Override time budget (uses mode default if None)
            
        Returns:
            TestSuiteComposition with selected pipelines and metadata
        """
        config = self.get_mode_config(mode)
        time_budget = time_budget_minutes or config.target_time_minutes
        max_time_budget = config.max_time_minutes
        
        logger.info(f"Selecting optimal pipeline suite for {mode.value} mode")
        logger.info(f"Time budget: {time_budget} minutes (max: {max_time_budget})")
        logger.info(f"Available pipelines: {len(available_pipelines)}")
        
        # Load historical data if not already loaded
        if not self._historical_data_loaded:
            self.load_historical_performance_data()
        
        # Get execution estimates for all available pipelines
        pipeline_estimates = []
        for pipeline in available_pipelines:
            if pipeline not in config.excluded_pipelines:
                estimate = self.estimate_pipeline_execution_time(pipeline)
                pipeline_estimates.append(estimate)
        
        # Filter by category priorities
        if config.priority_categories:
            pipeline_estimates = [
                est for est in pipeline_estimates 
                if est.category in config.priority_categories
            ]
        
        # Always include required pipelines
        selected_pipelines = list(config.required_pipelines & set(available_pipelines))
        remaining_estimates = [
            est for est in pipeline_estimates 
            if est.pipeline_name not in selected_pipelines
        ]
        
        # Calculate time used by required pipelines
        required_time = sum(
            self.estimate_pipeline_execution_time(p).estimated_time_seconds 
            for p in selected_pipelines
        ) / 60.0  # Convert to minutes
        
        logger.info(f"Required pipelines: {len(selected_pipelines)} ({required_time:.1f} min)")
        
        # Select additional pipelines using greedy algorithm
        remaining_time = time_budget - required_time
        remaining_estimates.sort(key=lambda x: x.priority_score, reverse=True)
        
        reasoning = [f"Starting with {len(selected_pipelines)} required pipelines ({required_time:.1f} min)"]
        
        for estimate in remaining_estimates:
            if config.max_pipelines and len(selected_pipelines) >= config.max_pipelines:
                reasoning.append(f"Reached maximum pipeline limit of {config.max_pipelines}")
                break
                
            estimated_minutes = estimate.estimated_time_seconds / 60.0
            if estimated_minutes <= remaining_time:
                selected_pipelines.append(estimate.pipeline_name)
                remaining_time -= estimated_minutes
                reasoning.append(
                    f"Added {estimate.pipeline_name} (priority: {estimate.priority_score:.1f}, "
                    f"time: {estimated_minutes:.1f} min)"
                )
            else:
                reasoning.append(
                    f"Skipped {estimate.pipeline_name} - would exceed time budget "
                    f"({estimated_minutes:.1f} min > {remaining_time:.1f} min remaining)"
                )
        
        # Calculate final metrics
        total_estimated_time = sum(
            self.estimate_pipeline_execution_time(p).estimated_time_seconds 
            for p in selected_pipelines
        ) / 60.0
        
        total_pipelines = len(available_pipelines)
        coverage_percentage = (len(selected_pipelines) / total_pipelines) * 100 if total_pipelines > 0 else 0
        
        avg_priority = mean([
            self.estimate_pipeline_execution_time(p).priority_score 
            for p in selected_pipelines
        ]) if selected_pipelines else 0
        
        # Estimate cost (rough approximation)
        estimated_cost = total_estimated_time * 0.02  # $0.02 per minute estimate
        
        composition = TestSuiteComposition(
            mode=mode,
            selected_pipelines=selected_pipelines,
            estimated_total_time_minutes=total_estimated_time,
            estimated_cost_dollars=estimated_cost,
            coverage_percentage=coverage_percentage,
            priority_score=avg_priority,
            reasoning=reasoning
        )
        
        logger.info(f"Selected {len(selected_pipelines)} pipelines for {mode.value} mode:")
        logger.info(f"  Estimated time: {total_estimated_time:.1f} minutes")
        logger.info(f"  Coverage: {coverage_percentage:.1f}%")
        logger.info(f"  Average priority: {avg_priority:.1f}")
        
        return composition
    
    def get_recommended_mode_for_time_budget(self, time_budget_minutes: int) -> TestMode:
        """Recommend the best test mode for a given time budget."""
        mode_times = []
        for mode, config in self._mode_configs.items():
            if time_budget_minutes >= config.target_time_minutes:
                mode_times.append((mode, config.target_time_minutes))
                
        if not mode_times:
            return TestMode.SMOKE  # Fallback to fastest mode
            
        # Select the mode with highest target time that fits in budget
        mode_times.sort(key=lambda x: x[1], reverse=True)
        return mode_times[0][0]
    
    def get_execution_summary(self, composition: TestSuiteComposition) -> Dict[str, Any]:
        """Get detailed execution summary for a test suite composition."""
        estimates = [
            self.estimate_pipeline_execution_time(p) 
            for p in composition.selected_pipelines
        ]
        
        # Category breakdown
        categories = {}
        for estimate in estimates:
            if estimate.category not in categories:
                categories[estimate.category] = []
            categories[estimate.category].append(estimate)
        
        category_summary = {}
        for category, cat_estimates in categories.items():
            total_time = sum(e.estimated_time_seconds for e in cat_estimates) / 60.0
            avg_confidence = mean([e.confidence for e in cat_estimates])
            category_summary[category] = {
                "pipelines": len(cat_estimates),
                "estimated_time_minutes": total_time,
                "average_confidence": avg_confidence,
                "pipeline_names": [e.pipeline_name for e in cat_estimates]
            }
        
        # Confidence analysis  
        high_confidence = sum(1 for e in estimates if e.confidence > 0.7)
        medium_confidence = sum(1 for e in estimates if 0.3 <= e.confidence <= 0.7)
        low_confidence = sum(1 for e in estimates if e.confidence < 0.3)
        
        return {
            "mode": composition.mode.value,
            "total_pipelines": len(composition.selected_pipelines),
            "estimated_time_minutes": composition.estimated_total_time_minutes,
            "estimated_cost_dollars": composition.estimated_cost_dollars,
            "coverage_percentage": composition.coverage_percentage,
            "average_priority_score": composition.priority_score,
            "category_breakdown": category_summary,
            "confidence_breakdown": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence, 
                "low_confidence": low_confidence
            },
            "reasoning": composition.reasoning,
            "pipeline_list": composition.selected_pipelines
        }


def get_available_test_modes() -> Dict[str, str]:
    """Get available test modes with descriptions."""
    manager = TestModeManager()
    return {
        mode.value: config.description 
        for mode, config in manager._mode_configs.items()
    }


def select_pipelines_for_mode(
    mode: str, 
    available_pipelines: List[str],
    performance_tracker=None,
    time_budget_minutes: Optional[int] = None
) -> TestSuiteComposition:
    """
    Convenience function to select pipelines for a specific mode.
    
    Args:
        mode: Test mode name (e.g., "quick", "core", "full")
        available_pipelines: List of available pipeline names
        performance_tracker: Optional performance tracker for historical data
        time_budget_minutes: Optional time budget override
        
    Returns:
        TestSuiteComposition with selected pipelines and metadata
    """
    manager = TestModeManager(performance_tracker)
    test_mode = TestMode(mode)
    return manager.select_optimal_pipeline_suite(
        test_mode, 
        available_pipelines, 
        time_budget_minutes
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Test Mode Manager - Example Usage")
    print("=" * 50)
    
    # Example pipeline list
    example_pipelines = [
        "simple_data_processing",
        "data_processing_pipeline", 
        "control_flow_conditional",
        "control_flow_for_loop",
        "research_minimal",
        "creative_image_pipeline",
        "statistical_analysis",
        "web_research_pipeline"
    ]
    
    manager = TestModeManager()
    
    for mode in [TestMode.SMOKE, TestMode.QUICK, TestMode.CORE]:
        print(f"\n{mode.value.upper()} Mode:")
        composition = manager.select_optimal_pipeline_suite(mode, example_pipelines)
        summary = manager.get_execution_summary(composition)
        
        print(f"  Selected: {len(composition.selected_pipelines)} pipelines")
        print(f"  Time: {composition.estimated_total_time_minutes:.1f} minutes")
        print(f"  Coverage: {composition.coverage_percentage:.1f}%")
        print(f"  Pipelines: {', '.join(composition.selected_pipelines[:3])}...")