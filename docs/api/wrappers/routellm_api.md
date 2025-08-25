# RouteLLM Integration API Reference

## Overview

This document provides comprehensive API reference for the RouteLLM integration, including configuration, routing decisions, cost tracking, and monitoring capabilities.

## Core Classes

### RouteLLMConfig

Configuration class for RouteLLM integration.

```python
from src.orchestrator.models.routellm_integration import RouteLLMConfig, RouterType

@dataclass
class RouteLLMConfig:
    # Core routing settings
    enabled: bool = False
    router_type: RouterType = RouterType.MATRIX_FACTORIZATION
    threshold: float = 0.11593
    
    # Model configuration  
    strong_model: str = "gpt-4-1106-preview"
    weak_model: str = "gpt-3.5-turbo"
    
    # Fallback and reliability
    fallback_enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Cost optimization
    cost_tracking_enabled: bool = True
    cost_optimization_target: float = 0.5
    
    # Performance monitoring
    performance_monitoring_enabled: bool = True
    metrics_retention_days: int = 30
    
    # Domain-specific settings
    domain_specific_routing: bool = True
    domain_routing_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

#### Methods

##### get_router_model_string()

Get the router model string for RouteLLM API calls.

```python
def get_router_model_string(self) -> str
```

**Returns:** Router model string in format `"router-{type}-{threshold}"`

**Example:**
```python
config = RouteLLMConfig(router_type=RouterType.MATRIX_FACTORIZATION, threshold=0.11593)
model_string = config.get_router_model_string()
# Returns: "router-mf-0.11593"
```

##### get_domain_override()

Get domain-specific routing overrides.

```python
def get_domain_override(self, domain: str) -> Optional[Dict[str, Any]]
```

**Parameters:**
- `domain`: Domain name to check for overrides

**Returns:** Dictionary of domain-specific settings or `None`

**Example:**
```python
config = RouteLLMConfig(
    domain_routing_overrides={
        "medical": {"threshold": 0.05, "strong_model": "gpt-4"}
    }
)
override = config.get_domain_override("medical")
# Returns: {"threshold": 0.05, "strong_model": "gpt-4"}
```

### RouterType

Enumeration of available router types.

```python
class RouterType(Enum):
    MATRIX_FACTORIZATION = "mf"
    BERT_CLASSIFIER = "bert"
    CAUSAL_LLM = "causal_llm"
    SIMILARITY_WEIGHTED = "sw_ranking"
    RANDOM = "random"
```

**Router Descriptions:**
- `MATRIX_FACTORIZATION`: Default router using matrix factorization (best general purpose)
- `BERT_CLASSIFIER`: BERT-based classification router (good for text classification)
- `CAUSAL_LLM`: Causal language model router (optimized for generation tasks)
- `SIMILARITY_WEIGHTED`: Similarity weighted ranking (semantic similarity based)
- `RANDOM`: Random routing (for testing and benchmarking)

### RoutingDecision

Result of a RouteLLM routing decision.

```python
@dataclass
class RoutingDecision:
    should_use_routellm: bool
    recommended_model: Optional[str] = None
    confidence: float = 0.0
    estimated_cost: float = 0.0
    reasoning: str = ""
    domains: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None
```

#### Properties

##### is_high_confidence

Check if this is a high-confidence routing decision.

```python
@property
def is_high_confidence(self) -> bool
```

**Returns:** `True` if confidence >= 0.8

##### is_cost_effective

Check if this routing decision is cost-effective.

```python
@property
def is_cost_effective(self) -> bool
```

**Returns:** `True` if estimated cost savings > 0

### RoutingMetrics

Metrics for tracking routing decisions and performance.

```python
@dataclass
class RoutingMetrics:
    # Request identification
    tracking_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Input characteristics
    input_text_length: int = 0
    detected_domains: List[str] = field(default_factory=list)
    domain_confidence: float = 0.0
    
    # Routing decision
    routing_method: str = ""  # 'routellm' or 'domain_selector'
    selected_model: str = ""
    routing_confidence: float = 0.0
    
    # Cost metrics
    estimated_cost: float = 0.0
    actual_cost: Optional[float] = None
    cost_savings_vs_baseline: Optional[float] = None
    
    # Performance metrics  
    routing_latency_ms: float = 0.0
    model_response_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    # Quality metrics
    response_quality_score: Optional[float] = None
    user_satisfaction: Optional[float] = None
```

## Feature Flag System

### FeatureFlags

Feature flags for RouteLLM integration rollout.

```python
class FeatureFlags:
    # Core feature flags
    ROUTELLM_ENABLED = "routellm_enabled"
    ROUTELLM_COST_TRACKING = "routellm_cost_tracking"
    ROUTELLM_PERFORMANCE_MONITORING = "routellm_performance_monitoring"
    
    # Domain-specific flags
    ROUTELLM_MEDICAL_DOMAIN = "routellm_medical_domain"
    ROUTELLM_LEGAL_DOMAIN = "routellm_legal_domain"
    ROUTELLM_TECHNICAL_DOMAIN = "routellm_technical_domain"
    ROUTELLM_CREATIVE_DOMAIN = "routellm_creative_domain"
    ROUTELLM_SCIENTIFIC_DOMAIN = "routellm_scientific_domain"
    ROUTELLM_FINANCIAL_DOMAIN = "routellm_financial_domain"
    ROUTELLM_EDUCATIONAL_DOMAIN = "routellm_educational_domain"
    
    # Experimental flags
    ROUTELLM_DYNAMIC_THRESHOLD = "routellm_dynamic_threshold"
    ROUTELLM_A_B_TESTING = "routellm_a_b_testing"
    ROUTELLM_QUALITY_FEEDBACK = "routellm_quality_feedback"
```

#### Methods

##### is_enabled()

Check if a feature flag is enabled.

```python
def is_enabled(self, flag: str) -> bool
```

**Example:**
```python
flags = FeatureFlags()
if flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED):
    # Use RouteLLM routing
    pass
```

##### enable()

Enable a feature flag.

```python
def enable(self, flag: str) -> None
```

##### disable()

Disable a feature flag.

```python
def disable(self, flag: str) -> None
```

##### update_flags()

Update multiple feature flags at once.

```python
def update_flags(self, flags: Dict[str, bool]) -> None
```

**Example:**
```python
flags.update_flags({
    FeatureFlags.ROUTELLM_ENABLED: True,
    FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN: True
})
```

##### is_domain_enabled()

Check if RouteLLM is enabled for a specific domain.

```python
def is_domain_enabled(self, domain: str) -> bool
```

**Parameters:**
- `domain`: Domain name (e.g., "medical", "technical")

**Returns:** `True` if RouteLLM is enabled for the domain

## Cost Tracking

### CostTracker

Track routing decisions and calculate cost savings.

```python
class CostTracker:
    def __init__(self, retention_days: int = 30)
```

#### Methods

##### track_routing_decision()

Track a routing decision and return tracking ID.

```python
def track_routing_decision(
    self,
    text: str,
    domains: List[str], 
    routing_method: str,
    selected_model: str,
    estimated_cost: float,
    routing_latency_ms: float = 0.0,
    routing_confidence: float = 0.0,
    success: bool = True,
    error_message: Optional[str] = None,
) -> str
```

**Parameters:**
- `text`: Input text that was routed
- `domains`: Detected domains for the text
- `routing_method`: Method used for routing ("routellm" or "domain_selector")
- `selected_model`: Model selected by routing
- `estimated_cost`: Estimated cost for the operation
- `routing_latency_ms`: Time taken for routing decision
- `routing_confidence`: Confidence score for routing decision  
- `success`: Whether the operation succeeded
- `error_message`: Error message if operation failed

**Returns:** Unique tracking ID for the decision

**Example:**
```python
tracker = CostTracker()
tracking_id = tracker.track_routing_decision(
    text="What is machine learning?",
    domains=["technical"],
    routing_method="routellm",
    selected_model="gpt-3.5-turbo",
    estimated_cost=0.002,
    routing_latency_ms=15.0,
    routing_confidence=0.85
)
```

##### update_actual_cost()

Update the actual cost for a tracked request.

```python
def update_actual_cost(self, tracking_id: str, actual_cost: float) -> None
```

**Example:**
```python
tracker.update_actual_cost(tracking_id, 0.0018)
```

##### update_quality_score()

Update the response quality score for a tracked request.

```python
def update_quality_score(self, tracking_id: str, quality_score: float) -> None
```

**Parameters:**
- `tracking_id`: ID from `track_routing_decision()`
- `quality_score`: Quality score from 0.0 to 1.0

##### get_cost_savings_report()

Generate comprehensive cost savings report.

```python
def get_cost_savings_report(self, period_days: int = 30) -> CostSavingsReport
```

**Parameters:**
- `period_days`: Number of days to include in report

**Returns:** `CostSavingsReport` with comprehensive metrics

**Example:**
```python
report = tracker.get_cost_savings_report(period_days=7)
print(f"Saved ${report.estimated_savings:.2f} over 7 days")
print(f"Success rate: {report.success_rate:.2%}")
```

##### get_metrics_summary()

Get a summary of all tracked metrics.

```python
def get_metrics_summary(self) -> Dict[str, Any]
```

**Returns:** Dictionary with summary metrics including:
- `total_requests`: Total number of tracked requests
- `success_rate`: Overall success rate
- `routellm_usage_rate`: Percentage of requests using RouteLLM
- `average_cost`: Average cost per request
- `average_latency_ms`: Average routing latency

### CostSavingsReport

Report on cost savings achieved through RouteLLM routing.

```python
@dataclass
class CostSavingsReport:
    period_days: int
    total_requests: int
    routellm_requests: int
    traditional_requests: int
    
    # Cost metrics
    total_estimated_cost: float
    routellm_estimated_cost: float
    traditional_estimated_cost: float
    estimated_savings: float
    savings_percentage: float
    
    # Performance metrics
    average_routing_latency_ms: float
    success_rate: float
    
    # Quality metrics  
    average_quality_score: Optional[float] = None
    quality_maintained_percentage: Optional[float] = None
```

## Usage Examples

### Basic RouteLLM Configuration

```python
from src.orchestrator.models.routellm_integration import (
    RouteLLMConfig, RouterType, FeatureFlags, CostTracker
)

# Create configuration
config = RouteLLMConfig(
    enabled=True,
    router_type=RouterType.MATRIX_FACTORIZATION,
    threshold=0.11593,
    strong_model="gpt-4-1106-preview", 
    weak_model="gpt-3.5-turbo",
    cost_tracking_enabled=True,
    domain_specific_routing=True
)

# Setup feature flags
flags = FeatureFlags()
flags.enable(FeatureFlags.ROUTELLM_ENABLED)
flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)

# Initialize cost tracking
tracker = CostTracker(retention_days=30)
```

### Domain-Specific Routing

```python
# Configure domain-specific overrides
config = RouteLLMConfig(
    domain_routing_overrides={
        "medical": {
            "threshold": 0.05,  # Lower threshold for medical domain
            "strong_model": "gpt-4",
            "enabled": True
        },
        "creative": {
            "threshold": 0.15,  # Higher threshold for creative tasks
            "strong_model": "gpt-4",
            "enabled": True
        }
    }
)

# Check domain override
medical_override = config.get_domain_override("medical")
if medical_override:
    print(f"Medical domain uses threshold: {medical_override['threshold']}")
```

### Cost Tracking and Reporting

```python
import asyncio
from datetime import datetime, timedelta

async def track_routing_usage():
    tracker = CostTracker()
    
    # Simulate some routing decisions
    test_cases = [
        {"text": "What is 2+2?", "model": "gpt-3.5-turbo", "cost": 0.001},
        {"text": "Explain quantum computing", "model": "gpt-4", "cost": 0.006}, 
        {"text": "Simple math problem", "model": "gpt-3.5-turbo", "cost": 0.001},
    ]
    
    tracking_ids = []
    for case in test_cases:
        tracking_id = tracker.track_routing_decision(
            text=case["text"],
            domains=["general"],
            routing_method="routellm",
            selected_model=case["model"],
            estimated_cost=case["cost"],
            routing_latency_ms=10.0,
            success=True
        )
        tracking_ids.append(tracking_id)
    
    # Update with actual costs
    for i, tracking_id in enumerate(tracking_ids):
        actual_cost = test_cases[i]["cost"] * 0.95  # Slightly lower actual cost
        tracker.update_actual_cost(tracking_id, actual_cost)
    
    # Generate report
    report = tracker.get_cost_savings_report(period_days=1)
    
    print("Cost Savings Report:")
    print(f"  Total requests: {report.total_requests}")
    print(f"  RouteLLM requests: {report.routellm_requests}")
    print(f"  Estimated savings: ${report.estimated_savings:.4f}")
    print(f"  Success rate: {report.success_rate:.2%}")
    
    # Get metrics summary
    summary = tracker.get_metrics_summary()
    print(f"\nMetrics Summary:")
    print(f"  Average cost: ${summary['average_cost']:.4f}")
    print(f"  RouteLLM usage: {summary['routellm_usage_rate']:.1%}")

asyncio.run(track_routing_usage())
```

### Feature Flag Rollout

```python
def gradual_routellm_rollout():
    """Example of gradual RouteLLM rollout using feature flags."""
    
    flags = FeatureFlags()
    
    # Phase 1: Enable monitoring only
    print("Phase 1: Monitoring only")
    flags.enable(FeatureFlags.ROUTELLM_COST_TRACKING)
    flags.enable(FeatureFlags.ROUTELLM_PERFORMANCE_MONITORING)
    
    # Phase 2: Enable for technical domain
    print("Phase 2: Technical domain")
    flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)
    
    # Phase 3: Enable more domains
    print("Phase 3: Multiple domains")
    flags.enable(FeatureFlags.ROUTELLM_CREATIVE_DOMAIN)
    flags.enable(FeatureFlags.ROUTELLM_EDUCATIONAL_DOMAIN)
    
    # Phase 4: Full rollout
    print("Phase 4: Full rollout")
    flags.enable(FeatureFlags.ROUTELLM_ENABLED)
    
    # Check final state
    enabled_flags = [flag for flag in flags.get_all_flags() 
                    if flags.is_enabled(flag)]
    print(f"Enabled flags: {enabled_flags}")

gradual_routellm_rollout()
```

### Advanced Configuration with Validation

```python
def create_validated_config():
    """Create and validate RouteLLM configuration."""
    
    # Create configuration with validation
    config = RouteLLMConfig(
        enabled=True,
        router_type=RouterType.BERT_CLASSIFIER,
        threshold=0.12,
        strong_model="gpt-4-turbo",
        weak_model="gpt-3.5-turbo",
        timeout_seconds=45.0,
        max_retry_attempts=2,
        cost_optimization_target=0.6,  # Target 60% cost reduction
        metrics_retention_days=45
    )
    
    # Validate configuration
    errors = []
    
    if config.threshold < 0 or config.threshold > 1:
        errors.append("Threshold must be between 0 and 1")
    
    if config.timeout_seconds <= 0:
        errors.append("Timeout must be positive")
        
    if config.max_retry_attempts < 0:
        errors.append("Retry attempts cannot be negative")
    
    if errors:
        print(f"Configuration errors: {errors}")
        return None
    
    print("Configuration validated successfully")
    print(f"Router model string: {config.get_router_model_string()}")
    
    return config

create_validated_config()
```

## Error Handling

### RouteLLM-Specific Exceptions

```python
class RouteLLMException(Exception):
    """Base exception for RouteLLM errors."""
    pass

class RoutingDecisionError(RouteLLMException):
    """Raised when routing decision cannot be made."""
    pass

class ModelNotAvailableError(RouteLLMException):
    """Raised when selected model is not available."""
    pass

class CostTrackingError(RouteLLMException):
    """Raised when cost tracking fails."""
    pass
```

### Error Handling Examples

```python
from src.orchestrator.models.routellm_integration import RoutingDecisionError

async def handle_routing_errors():
    """Example of handling RouteLLM routing errors."""
    
    config = RouteLLMConfig(enabled=True)
    tracker = CostTracker()
    
    try:
        # Simulate routing decision
        routing_decision = make_routing_decision("test query")
        
        if routing_decision.should_use_routellm:
            # Track successful routing
            tracking_id = tracker.track_routing_decision(
                text="test query",
                domains=routing_decision.domains,
                routing_method="routellm",
                selected_model=routing_decision.recommended_model,
                estimated_cost=routing_decision.estimated_cost,
                success=True
            )
            print(f"Routing successful, tracking ID: {tracking_id}")
        
    except RoutingDecisionError as e:
        print(f"Routing decision failed: {e}")
        # Fall back to default model
        tracking_id = tracker.track_routing_decision(
            text="test query",
            domains=["unknown"],
            routing_method="fallback",
            selected_model=config.strong_model,
            estimated_cost=0.006,  # Assume higher cost for fallback
            success=False,
            error_message=str(e)
        )
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Log error and continue with fallback
```

## Environment Variables

RouteLLM integration supports these environment variables:

```bash
# Core RouteLLM settings
ROUTELLM_ENABLED=false
ROUTELLM_ROUTER_TYPE=mf
ROUTELLM_THRESHOLD=0.11593

# Model configuration
ROUTELLM_STRONG_MODEL=gpt-4-1106-preview
ROUTELLM_WEAK_MODEL=gpt-3.5-turbo

# Performance settings
ROUTELLM_TIMEOUT_SECONDS=30.0
ROUTELLM_MAX_RETRIES=3

# Cost tracking
ROUTELLM_COST_TRACKING=true
ROUTELLM_COST_TARGET=0.5

# Monitoring
ROUTELLM_PERFORMANCE_MONITORING=true
ROUTELLM_METRICS_RETENTION_DAYS=30

# Feature flags
ROUTELLM_MEDICAL_DOMAIN=false
ROUTELLM_TECHNICAL_DOMAIN=false
ROUTELLM_CREATIVE_DOMAIN=false
```

## REST API Endpoints

When used with the orchestrator's REST API, these endpoints are available:

### GET /api/v1/routellm/config

Get current RouteLLM configuration.

**Response:**
```json
{
  "enabled": false,
  "router_type": "mf", 
  "threshold": 0.11593,
  "strong_model": "gpt-4-1106-preview",
  "weak_model": "gpt-3.5-turbo",
  "cost_tracking_enabled": true
}
```

### POST /api/v1/routellm/config

Update RouteLLM configuration.

**Request:**
```json
{
  "enabled": true,
  "threshold": 0.12
}
```

### GET /api/v1/routellm/cost-report

Get cost savings report.

**Query Parameters:**
- `period_days` (optional): Number of days for report (default: 30)

**Response:**
```json
{
  "period_days": 30,
  "total_requests": 1500,
  "estimated_savings": 245.67,
  "savings_percentage": 35.2,
  "success_rate": 0.987
}
```

### GET /api/v1/routellm/health

Get RouteLLM health status.

**Response:**
```json
{
  "status": "healthy",
  "routing_enabled": true,
  "success_rate": 0.987,
  "average_latency_ms": 12.5,
  "cost_tracking_active": true
}
```

This comprehensive API reference provides all the tools needed to integrate, configure, and monitor RouteLLM routing within your applications.