"""Intelligence Layer - Phase 3 Advanced Features

Advanced intelligence capabilities for the orchestrator including:
- Intelligent model selection with multi-dimensional optimization
- Performance profiling and monitoring
- Model lifecycle management with instance caching
- Health monitoring and automatic recovery
- Advanced optimization algorithms
"""

from .intelligent_model_selector import (
    IntelligentModelSelector,
    ModelRequirements,
    ModelScore,
    OptimizationObjective,
    create_intelligent_selector,
    select_optimal_model_for_task
)

from .model_health_monitor import (
    ModelHealthMonitor,
    HealthStatus,
    HealthCheck,
    HealthMetrics,
    create_health_monitor,
    setup_basic_health_monitoring
)

__all__ = [
    # Intelligent Model Selection
    "IntelligentModelSelector",
    "ModelRequirements", 
    "ModelScore",
    "OptimizationObjective",
    "create_intelligent_selector",
    "select_optimal_model_for_task",
    
    # Health Monitoring
    "ModelHealthMonitor",
    "HealthStatus",
    "HealthCheck", 
    "HealthMetrics",
    "create_health_monitor",
    "setup_basic_health_monitoring"
]