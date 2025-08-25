# Wrapper Configuration Guide

## Overview

This comprehensive guide covers all configuration options for the wrapper framework, including RouteLLM integration, POML processing, feature flags, monitoring, and environment-specific settings. Learn how to configure, validate, and manage wrapper configurations across different environments.

## Table of Contents

- [Configuration Architecture](#configuration-architecture)
- [Core Wrapper Configuration](#core-wrapper-configuration)
- [RouteLLM Configuration](#routellm-configuration)
- [POML Configuration](#poml-configuration)
- [Feature Flags Configuration](#feature-flags-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Environment Management](#environment-management)
- [Configuration Examples](#configuration-examples)
- [Validation and Troubleshooting](#validation-and-troubleshooting)

## Configuration Architecture

### Configuration Hierarchy

The wrapper framework uses a hierarchical configuration system:

```
1. Default Values (in code)
2. Configuration Files (YAML/JSON)
3. Environment Variables
4. Runtime Overrides
5. Command Line Arguments (if applicable)
```

### Configuration Sources

```python
from src.orchestrator.core.wrapper_config import ConfigSource

# Configuration can come from multiple sources
class ConfigSource(Enum):
    DEFAULT = "default"        # Built-in defaults
    FILE = "file"             # Configuration files
    ENVIRONMENT = "environment" # Environment variables
    RUNTIME = "runtime"       # Runtime updates
    OVERRIDE = "override"     # Explicit overrides
```

### Configuration Validation

All configurations go through validation:

```python
from src.orchestrator.core.wrapper_config import ConfigValidationError

try:
    config = MyWrapperConfig()
    validation_result = config.validate()
    if not validation_result.valid:
        print(f"Configuration errors: {validation_result.errors}")
except ConfigValidationError as e:
    print(f"Invalid configuration: {e}")
```

## Core Wrapper Configuration

### Base Wrapper Settings

All wrapper configurations extend `BaseWrapperConfig`:

```yaml
# config.yaml - Core wrapper settings
wrapper_framework:
  # Global enablement
  enabled: true
  
  # Default timeouts and retries
  default_timeout_seconds: 30.0
  max_retry_attempts: 3
  retry_backoff_seconds: 1.0
  
  # Resource management
  connection_pool_size: 100
  max_concurrent_operations: 50
  
  # Monitoring and observability
  enable_monitoring: true
  enable_metrics_collection: true
  metrics_retention_days: 30
  
  # Caching
  enable_caching: true
  default_cache_ttl_seconds: 300
  max_cache_size_mb: 100
  
  # Security
  validate_ssl_certificates: true
  mask_sensitive_data_in_logs: true
  enable_request_signing: false
```

### Environment Variables

Core wrapper settings can be overridden with environment variables:

```bash
# Core wrapper settings
export WRAPPER_FRAMEWORK_ENABLED=true
export WRAPPER_DEFAULT_TIMEOUT=30.0
export WRAPPER_MAX_RETRIES=3

# Resource management
export WRAPPER_CONNECTION_POOL_SIZE=100
export WRAPPER_MAX_CONCURRENT_OPS=50

# Monitoring
export WRAPPER_ENABLE_MONITORING=true
export WRAPPER_METRICS_RETENTION_DAYS=30

# Caching
export WRAPPER_ENABLE_CACHING=true
export WRAPPER_DEFAULT_CACHE_TTL=300

# Security
export WRAPPER_VALIDATE_SSL=true
export WRAPPER_MASK_SENSITIVE_DATA=true
```

### Programmatic Configuration

```python
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField
from dataclasses import dataclass
from typing import Dict

@dataclass
class MyWrapperConfig(BaseWrapperConfig):
    """Example wrapper configuration."""
    
    # Service-specific settings
    api_endpoint: str = "https://api.example.com"
    api_key: str = ""
    service_version: str = "v1"
    
    # Performance settings
    timeout_seconds: float = 30.0
    max_retries: int = 3
    rate_limit_per_minute: int = 1000
    
    # Feature toggles
    enable_experimental_features: bool = False
    enable_advanced_caching: bool = True
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Define configuration fields with validation."""
        return {
            "api_endpoint": ConfigField(
                name="api_endpoint",
                field_type=str,
                default_value=self.api_endpoint,
                description="Service API endpoint URL",
                required=True,
                validator=self._validate_url
            ),
            "api_key": ConfigField(
                name="api_key",
                field_type=str,
                default_value="",
                description="API authentication key",
                required=True,
                sensitive=True,
                environment_var="MY_SERVICE_API_KEY"
            ),
            "timeout_seconds": ConfigField(
                name="timeout_seconds",
                field_type=float,
                default_value=self.timeout_seconds,
                description="Request timeout in seconds",
                min_value=1.0,
                max_value=300.0
            ),
            "rate_limit_per_minute": ConfigField(
                name="rate_limit_per_minute",
                field_type=int,
                default_value=self.rate_limit_per_minute,
                description="API rate limit per minute",
                min_value=1,
                max_value=100000
            )
        }
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format."""
        return url.startswith(("http://", "https://"))
```

## RouteLLM Configuration

### Complete RouteLLM Settings

```yaml
# config.yaml - RouteLLM configuration
routellm:
  # Core routing settings
  enabled: false  # Start disabled for safe rollout
  router_type: "mf"  # Matrix factorization router
  threshold: 0.11593  # Default routing threshold
  
  # Model configuration
  strong_model: "gpt-4-1106-preview"
  weak_model: "gpt-3.5-turbo"
  
  # Performance and reliability
  fallback_enabled: true
  max_retry_attempts: 3
  timeout_seconds: 30.0
  
  # Cost optimization
  cost_tracking_enabled: true
  cost_optimization_target: 0.5  # Target 50% cost reduction
  
  # Performance monitoring
  performance_monitoring_enabled: true
  metrics_retention_days: 30
  
  # Domain-specific routing
  domain_specific_routing: true
  domain_routing_overrides:
    medical:
      threshold: 0.05  # Lower threshold for medical domain
      strong_model: "gpt-4"
      enabled: true
    legal:
      threshold: 0.08
      strong_model: "gpt-4"
      enabled: true
    creative:
      threshold: 0.15  # Higher threshold for creative tasks
      strong_model: "gpt-4"
      enabled: true
  
  # Advanced settings
  enable_dynamic_threshold: false
  enable_quality_feedback: false
  enable_a_b_testing: false
  
  # Experimental features
  experimental:
    adaptive_routing: false
    multi_model_ensemble: false
    custom_routing_rules: false
```

### RouteLLM Environment Variables

```bash
# Core RouteLLM settings
export ROUTELLM_ENABLED=false
export ROUTELLM_ROUTER_TYPE=mf
export ROUTELLM_THRESHOLD=0.11593

# Model settings
export ROUTELLM_STRONG_MODEL="gpt-4-1106-preview"
export ROUTELLM_WEAK_MODEL="gpt-3.5-turbo"

# Performance settings
export ROUTELLM_TIMEOUT_SECONDS=30.0
export ROUTELLM_MAX_RETRIES=3
export ROUTELLM_FALLBACK_ENABLED=true

# Cost tracking
export ROUTELLM_COST_TRACKING=true
export ROUTELLM_COST_TARGET=0.5

# Monitoring
export ROUTELLM_PERFORMANCE_MONITORING=true
export ROUTELLM_METRICS_RETENTION_DAYS=30

# Domain-specific settings
export ROUTELLM_DOMAIN_ROUTING=true
export ROUTELLM_MEDICAL_THRESHOLD=0.05
export ROUTELLM_LEGAL_THRESHOLD=0.08
export ROUTELLM_CREATIVE_THRESHOLD=0.15
```

### RouteLLM Router Types Configuration

```python
from src.orchestrator.models.routellm_integration import RouterType, RouteLLMConfig

# Router type configurations
router_configs = {
    RouterType.MATRIX_FACTORIZATION: {
        "threshold": 0.11593,
        "description": "Best general-purpose router",
        "use_cases": ["general", "mixed_workload"],
        "performance_profile": "balanced"
    },
    RouterType.BERT_CLASSIFIER: {
        "threshold": 0.125,
        "description": "BERT-based classification",
        "use_cases": ["text_classification", "sentiment"],
        "performance_profile": "high_accuracy"
    },
    RouterType.CAUSAL_LLM: {
        "threshold": 0.10,
        "description": "Causal language model routing",
        "use_cases": ["text_generation", "completion"],
        "performance_profile": "generative_tasks"
    },
    RouterType.SIMILARITY_WEIGHTED: {
        "threshold": 0.15,
        "description": "Semantic similarity routing",
        "use_cases": ["semantic_search", "similarity"],
        "performance_profile": "semantic_tasks"
    }
}

# Create configuration for specific router
config = RouteLLMConfig(
    router_type=RouterType.BERT_CLASSIFIER,
    threshold=router_configs[RouterType.BERT_CLASSIFIER]["threshold"]
)
```

## POML Configuration

### Complete POML Settings

```yaml
# config.yaml - POML configuration
template_config:
  # Core POML processing
  enable_poml_processing: true
  poml_strict_mode: false  # Allow mixed content
  
  # Format detection
  auto_detect_format: true
  default_format: "hybrid"  # jinja2, poml, hybrid, plain
  
  # Fallback behavior
  fallback_to_jinja: true
  fallback_on_error: true
  
  # Advanced features
  enable_advanced_resolution: true
  cross_task_references: true
  enable_output_tracking: true
  
  # Performance settings
  enable_caching: true
  cache_ttl_seconds: 3600
  max_template_size_kb: 1024
  template_timeout_seconds: 30
  
  # Validation settings
  validate_templates: true
  strict_validation: false
  validate_poml_syntax: true
  validate_jinja_syntax: true
  
  # Migration settings
  migration:
    preserve_comments: true
    validate_output: true
    backup_originals: true
    conversion_batch_size: 10
  
  # Template resolution
  resolution:
    max_resolution_depth: 10
    enable_circular_detection: true
    cache_resolved_templates: true
    parallel_resolution: true
```

### POML Environment Variables

```bash
# Core POML settings
export TEMPLATE_ENABLE_POML=true
export TEMPLATE_STRICT_MODE=false
export TEMPLATE_AUTO_DETECT_FORMAT=true
export TEMPLATE_DEFAULT_FORMAT=hybrid

# Fallback settings
export TEMPLATE_FALLBACK_TO_JINJA=true
export TEMPLATE_FALLBACK_ON_ERROR=true

# Advanced features
export TEMPLATE_ADVANCED_RESOLUTION=true
export TEMPLATE_CROSS_TASK_REFS=true
export TEMPLATE_OUTPUT_TRACKING=true

# Performance settings
export TEMPLATE_ENABLE_CACHING=true
export TEMPLATE_CACHE_TTL=3600
export TEMPLATE_MAX_SIZE_KB=1024
export TEMPLATE_TIMEOUT_SECONDS=30

# Validation settings
export TEMPLATE_VALIDATE_TEMPLATES=true
export TEMPLATE_STRICT_VALIDATION=false
export TEMPLATE_VALIDATE_POML_SYNTAX=true

# Migration settings
export TEMPLATE_PRESERVE_COMMENTS=true
export TEMPLATE_VALIDATE_OUTPUT=true
export TEMPLATE_BACKUP_ORIGINALS=true
```

### POML Format-Specific Settings

```python
from src.orchestrator.core.template_resolver import TemplateFormat, TemplateResolver

# Format-specific configurations
format_settings = {
    TemplateFormat.JINJA2: {
        "strict_undefined": False,
        "auto_escape": True,
        "trim_blocks": True,
        "lstrip_blocks": True
    },
    TemplateFormat.POML: {
        "validate_xml_syntax": True,
        "allow_custom_elements": True,
        "enforce_closing_tags": False,
        "namespace_validation": False
    },
    TemplateFormat.HYBRID: {
        "jinja_precedence": True,
        "poml_validation": "lenient",
        "mixed_syntax_warnings": True
    }
}

# Create resolver with format-specific settings
resolver = TemplateResolver(
    enable_poml_processing=True,
    format_settings=format_settings
)
```

## Feature Flags Configuration

### Feature Flag System Settings

```yaml
# config.yaml - Feature flags configuration
feature_flags:
  # Core feature flag system
  enabled: true
  
  # Storage backend
  backend: "memory"  # memory, redis, database, file
  redis_url: "redis://localhost:6379"  # if using Redis
  file_path: "flags.json"  # if using file backend
  
  # Cache settings
  enable_caching: true
  cache_ttl_seconds: 300
  cache_size_limit: 1000
  
  # Evaluation settings
  default_strategy: "boolean"
  evaluation_timeout_ms: 100
  enable_evaluation_logging: true
  
  # Update settings
  allow_runtime_updates: true
  update_batch_size: 50
  update_notification: true
  
  # Monitoring
  track_evaluations: true
  track_flag_changes: true
  metrics_retention_days: 90
```

### Individual Feature Flags

```yaml
# config.yaml - Specific feature flags
feature_flags:
  flags:
    # RouteLLM flags
    routellm_enabled:
      description: "Enable RouteLLM routing"
      strategy: "boolean"
      default_value: false
      enabled: true
      
    routellm_medical_domain:
      description: "Enable RouteLLM for medical domain"
      strategy: "boolean"
      default_value: false
      dependencies: ["routellm_enabled"]
      
    routellm_percentage_rollout:
      description: "Percentage rollout for RouteLLM"
      strategy: "percentage"
      default_value: 0
      metadata:
        target_percentage: 100
        increment: 10
    
    # POML flags
    poml_processing_enabled:
      description: "Enable POML template processing"
      strategy: "boolean"
      default_value: true
      
    poml_advanced_features:
      description: "Enable advanced POML features"
      strategy: "whitelist"
      default_value: ["admin", "beta_user"]
      
    # Experimental flags
    experimental_caching:
      description: "Enable experimental caching"
      strategy: "percentage"
      default_value: 10
      metadata:
        experiment_name: "cache_optimization_v2"
```

### Feature Flag Environment Variables

```bash
# Feature flag system
export FEATURE_FLAGS_ENABLED=true
export FEATURE_FLAGS_BACKEND=memory
export FEATURE_FLAGS_CACHE_TTL=300

# Individual flags
export FLAG_ROUTELLM_ENABLED=false
export FLAG_ROUTELLM_MEDICAL_DOMAIN=false
export FLAG_ROUTELLM_PERCENTAGE=0
export FLAG_POML_PROCESSING=true
export FLAG_POML_ADVANCED=false
export FLAG_EXPERIMENTAL_CACHING=10
```

### Programmatic Feature Flag Configuration

```python
from src.orchestrator.core.feature_flags import (
    FeatureFlagManager, FeatureFlag, FeatureFlagStrategy
)

# Create feature flag manager
flag_manager = FeatureFlagManager()

# Define feature flags
flags = [
    FeatureFlag(
        name="advanced_routing",
        description="Enable advanced routing algorithms",
        strategy=FeatureFlagStrategy.BOOLEAN,
        default_value=False,
        enabled=True
    ),
    FeatureFlag(
        name="beta_features",
        description="Enable beta features for selected users",
        strategy=FeatureFlagStrategy.WHITELIST,
        default_value=["user123", "user456"],
        metadata={"beta_group": "early_adopters"}
    ),
    FeatureFlag(
        name="gradual_rollout", 
        description="Gradual feature rollout",
        strategy=FeatureFlagStrategy.PERCENTAGE,
        default_value=10,
        metadata={"max_percentage": 100}
    )
]

# Register flags
for flag in flags:
    flag_manager.register_flag(flag)

# Update flag values
flag_manager.update_flag("advanced_routing", True)
flag_manager.update_flag("gradual_rollout", 25)
```

## Monitoring Configuration

### Comprehensive Monitoring Settings

```yaml
# config.yaml - Monitoring configuration
monitoring:
  # Core monitoring
  enabled: true
  
  # Metrics collection
  metrics:
    enabled: true
    collection_interval_seconds: 30
    retention_days: 90
    export_interval_seconds: 60
    
    # Metric types to collect
    collect_request_metrics: true
    collect_error_metrics: true
    collect_performance_metrics: true
    collect_cache_metrics: true
    collect_resource_metrics: true
  
  # Health checking
  health_checks:
    enabled: true
    check_interval_seconds: 30
    timeout_seconds: 10
    failure_threshold: 3
    recovery_threshold: 2
  
  # Alerting
  alerts:
    enabled: true
    
    # Alert rules
    rules:
      high_error_rate:
        condition: "error_rate > 0.05"
        window_minutes: 5
        severity: "high"
        
      slow_response_time:
        condition: "avg_response_time > 2000"
        window_minutes: 10
        severity: "medium"
        
      high_cache_miss_rate:
        condition: "cache_miss_rate > 0.80"
        window_minutes: 15
        severity: "low"
  
  # Logging
  logging:
    level: "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: "structured"  # structured, plain
    include_sensitive_data: false
    
    # Log destinations
    console: true
    file: true
    file_path: "/var/log/wrapper.log"
    file_max_size_mb: 100
    file_backup_count: 5
    
    # External logging
    syslog: false
    syslog_address: "localhost:514"
    
  # Tracing
  tracing:
    enabled: false
    service_name: "wrapper-service"
    jaeger_endpoint: "http://localhost:14268"
    sampling_rate: 0.1
```

### Monitoring Environment Variables

```bash
# Core monitoring
export MONITORING_ENABLED=true

# Metrics
export METRICS_ENABLED=true
export METRICS_COLLECTION_INTERVAL=30
export METRICS_RETENTION_DAYS=90

# Health checks
export HEALTH_CHECKS_ENABLED=true
export HEALTH_CHECK_INTERVAL=30
export HEALTH_CHECK_TIMEOUT=10

# Alerting
export ALERTS_ENABLED=true
export ALERT_ERROR_RATE_THRESHOLD=0.05
export ALERT_RESPONSE_TIME_THRESHOLD=2000

# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=structured
export LOG_CONSOLE=true
export LOG_FILE=true
export LOG_FILE_PATH=/var/log/wrapper.log

# Tracing
export TRACING_ENABLED=false
export JAEGER_ENDPOINT=http://localhost:14268
```

### Custom Monitoring Configuration

```python
from src.orchestrator.core.wrapper_monitoring import (
    WrapperMonitoring, AlertRule, AlertCondition, MetricsConfig
)

# Create custom monitoring configuration
metrics_config = MetricsConfig(
    collection_interval_seconds=15,
    retention_days=60,
    enable_detailed_metrics=True
)

monitoring = WrapperMonitoring(metrics_config=metrics_config)

# Define custom alert rules
custom_alerts = [
    AlertRule(
        name="wrapper_specific_error",
        description="Alert on wrapper-specific errors",
        condition=AlertCondition.CUSTOM,
        threshold=5,  # 5 errors in window
        window_minutes=5,
        severity="high",
        custom_condition=lambda metrics: metrics.error_count > 5
    ),
    AlertRule(
        name="cache_efficiency",
        description="Alert on low cache efficiency",
        condition=AlertCondition.CACHE_HIT_RATE_THRESHOLD,
        threshold=0.70,  # Below 70% cache hit rate
        window_minutes=20,
        severity="medium"
    )
]

# Add alert rules
for alert in custom_alerts:
    monitoring.add_alert_rule(alert)
```

## Environment Management

### Environment-Specific Configurations

```yaml
# config/development.yaml
wrapper_framework:
  enabled: true
  default_timeout_seconds: 5.0  # Short timeout for development
  max_retry_attempts: 1
  enable_monitoring: false  # Disable monitoring in dev

routellm:
  enabled: false  # Disabled in development
  
template_config:
  enable_poml_processing: true
  validate_templates: false  # Skip validation in dev for speed
  
feature_flags:
  allow_runtime_updates: true  # Allow flag changes in dev

monitoring:
  enabled: false
  logging:
    level: "DEBUG"
---
# config/staging.yaml  
wrapper_framework:
  enabled: true
  default_timeout_seconds: 20.0
  max_retry_attempts: 2
  enable_monitoring: true

routellm:
  enabled: true
  threshold: 0.15  # Conservative threshold in staging
  
template_config:
  enable_poml_processing: true
  validate_templates: true
  
monitoring:
  enabled: true
  logging:
    level: "INFO"
---
# config/production.yaml
wrapper_framework:
  enabled: true
  default_timeout_seconds: 45.0
  max_retry_attempts: 5
  enable_monitoring: true
  connection_pool_size: 500  # Larger pool for production

routellm:
  enabled: true
  threshold: 0.11593  # Optimized threshold
  performance_monitoring_enabled: true
  
template_config:
  enable_poml_processing: true
  validate_templates: true
  enable_caching: true
  cache_ttl_seconds: 3600
  
monitoring:
  enabled: true
  metrics:
    retention_days: 180  # Longer retention in production
  alerts:
    enabled: true
  logging:
    level: "WARNING"  # Less verbose in production
```

### Configuration Loading

```python
# config_loader.py
import os
import yaml
from typing import Dict, Any
from pathlib import Path

class ConfigurationLoader:
    """Load and manage environment-specific configurations."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.environment = os.getenv("ENVIRONMENT", "development")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration for current environment."""
        # Load base configuration
        base_config = self._load_yaml("base.yaml")
        
        # Load environment-specific configuration
        env_config_file = f"{self.environment}.yaml"
        env_config = self._load_yaml(env_config_file)
        
        # Merge configurations (env overrides base)
        config = self._deep_merge(base_config, env_config)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        return config
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Define environment variable mappings
        env_mappings = {
            "WRAPPER_ENABLED": ("wrapper_framework", "enabled"),
            "WRAPPER_TIMEOUT": ("wrapper_framework", "default_timeout_seconds"),
            "ROUTELLM_ENABLED": ("routellm", "enabled"),
            "ROUTELLM_THRESHOLD": ("routellm", "threshold"),
            "POML_ENABLED": ("template_config", "enable_poml_processing"),
            "MONITORING_ENABLED": ("monitoring", "enabled"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in config:
                    config[section] = {}
                
                # Convert string to appropriate type
                config[section][key] = self._convert_env_value(value)
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Numeric conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value

# Usage
loader = ConfigurationLoader()
config = loader.load_config()
print(f"Loaded configuration for environment: {loader.environment}")
```

## Configuration Examples

### Complete Example Configuration

```yaml
# config/complete_example.yaml
# Complete wrapper framework configuration example

# Global wrapper framework settings
wrapper_framework:
  enabled: true
  default_timeout_seconds: 30.0
  max_retry_attempts: 3
  retry_backoff_seconds: 1.0
  connection_pool_size: 100
  max_concurrent_operations: 50
  enable_monitoring: true
  enable_metrics_collection: true
  metrics_retention_days: 30
  enable_caching: true
  default_cache_ttl_seconds: 300
  max_cache_size_mb: 100
  validate_ssl_certificates: true
  mask_sensitive_data_in_logs: true

# RouteLLM configuration
routellm:
  enabled: false
  router_type: "mf"
  threshold: 0.11593
  strong_model: "gpt-4-1106-preview"
  weak_model: "gpt-3.5-turbo"
  fallback_enabled: true
  max_retry_attempts: 3
  timeout_seconds: 30.0
  cost_tracking_enabled: true
  cost_optimization_target: 0.5
  performance_monitoring_enabled: true
  metrics_retention_days: 30
  domain_specific_routing: true
  domain_routing_overrides:
    medical:
      threshold: 0.05
      strong_model: "gpt-4"
      enabled: true
    legal:
      threshold: 0.08
      strong_model: "gpt-4"
      enabled: true
    creative:
      threshold: 0.15
      strong_model: "gpt-4"
      enabled: true

# POML template configuration
template_config:
  enable_poml_processing: true
  poml_strict_mode: false
  auto_detect_format: true
  default_format: "hybrid"
  fallback_to_jinja: true
  fallback_on_error: true
  enable_advanced_resolution: true
  cross_task_references: true
  enable_output_tracking: true
  enable_caching: true
  cache_ttl_seconds: 3600
  max_template_size_kb: 1024
  template_timeout_seconds: 30
  validate_templates: true
  strict_validation: false
  validate_poml_syntax: true
  validate_jinja_syntax: true

# Feature flags configuration
feature_flags:
  enabled: true
  backend: "memory"
  enable_caching: true
  cache_ttl_seconds: 300
  default_strategy: "boolean"
  evaluation_timeout_ms: 100
  allow_runtime_updates: true
  track_evaluations: true
  track_flag_changes: true
  
  # Specific flags
  flags:
    routellm_enabled:
      description: "Enable RouteLLM routing"
      strategy: "boolean"
      default_value: false
    poml_processing_enabled:
      description: "Enable POML template processing"
      strategy: "boolean"
      default_value: true

# Monitoring configuration
monitoring:
  enabled: true
  
  metrics:
    enabled: true
    collection_interval_seconds: 30
    retention_days: 90
    export_interval_seconds: 60
    collect_request_metrics: true
    collect_error_metrics: true
    collect_performance_metrics: true
    collect_cache_metrics: true
    collect_resource_metrics: true
  
  health_checks:
    enabled: true
    check_interval_seconds: 30
    timeout_seconds: 10
    failure_threshold: 3
    recovery_threshold: 2
  
  alerts:
    enabled: true
    rules:
      high_error_rate:
        condition: "error_rate > 0.05"
        window_minutes: 5
        severity: "high"
      slow_response_time:
        condition: "avg_response_time > 2000"
        window_minutes: 10
        severity: "medium"
  
  logging:
    level: "INFO"
    format: "structured"
    include_sensitive_data: false
    console: true
    file: true
    file_path: "/var/log/wrapper.log"
    file_max_size_mb: 100
    file_backup_count: 5

# Custom wrapper configurations
custom_wrappers:
  weather_api:
    enabled: true
    api_key: "${WEATHER_API_KEY}"
    base_url: "https://api.weatherapi.com/v1"
    timeout_seconds: 15.0
    cache_duration_minutes: 15
    rate_limit_per_minute: 1000
  
  database_wrapper:
    enabled: true
    connection_string: "${DATABASE_URL}"
    pool_size: 20
    timeout_seconds: 30.0
    retry_attempts: 3
```

### Environment-Specific Templates

```bash
# .env.development
ENVIRONMENT=development
WRAPPER_ENABLED=true
WRAPPER_TIMEOUT=5.0
WRAPPER_MAX_RETRIES=1
MONITORING_ENABLED=false
LOG_LEVEL=DEBUG
ROUTELLM_ENABLED=false
POML_ENABLED=true
FEATURE_FLAGS_ENABLED=true

# .env.staging
ENVIRONMENT=staging
WRAPPER_ENABLED=true
WRAPPER_TIMEOUT=20.0
WRAPPER_MAX_RETRIES=2
MONITORING_ENABLED=true
LOG_LEVEL=INFO
ROUTELLM_ENABLED=true
ROUTELLM_THRESHOLD=0.15
POML_ENABLED=true
FEATURE_FLAGS_ENABLED=true

# .env.production
ENVIRONMENT=production
WRAPPER_ENABLED=true
WRAPPER_TIMEOUT=45.0
WRAPPER_MAX_RETRIES=5
WRAPPER_CONNECTION_POOL_SIZE=500
MONITORING_ENABLED=true
LOG_LEVEL=WARNING
ROUTELLM_ENABLED=true
ROUTELLM_THRESHOLD=0.11593
ROUTELLM_COST_TRACKING=true
POML_ENABLED=true
FEATURE_FLAGS_ENABLED=true
METRICS_RETENTION_DAYS=180
```

## Validation and Troubleshooting

### Configuration Validation

```python
# config_validation.py
from src.orchestrator.core.wrapper_config import ConfigManager, ValidationResult
from typing import Dict, Any, List

class ConfigurationValidator:
    """Comprehensive configuration validation."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def validate_complete_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate complete configuration."""
        errors = []
        warnings = []
        
        # Validate wrapper framework config
        if "wrapper_framework" in config:
            framework_errors = self._validate_framework_config(config["wrapper_framework"])
            errors.extend(framework_errors)
        
        # Validate RouteLLM config
        if "routellm" in config:
            routellm_errors = self._validate_routellm_config(config["routellm"])
            errors.extend(routellm_errors)
        
        # Validate POML config
        if "template_config" in config:
            template_errors = self._validate_template_config(config["template_config"])
            errors.extend(template_errors)
        
        # Validate feature flags config
        if "feature_flags" in config:
            flags_errors = self._validate_feature_flags_config(config["feature_flags"])
            errors.extend(flags_errors)
        
        # Validate monitoring config
        if "monitoring" in config:
            monitoring_errors = self._validate_monitoring_config(config["monitoring"])
            errors.extend(monitoring_errors)
        
        # Check for conflicts
        conflicts = self._check_config_conflicts(config)
        warnings.extend(conflicts)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_framework_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate wrapper framework configuration."""
        errors = []
        
        if config.get("default_timeout_seconds", 0) <= 0:
            errors.append("default_timeout_seconds must be positive")
        
        if config.get("max_retry_attempts", -1) < 0:
            errors.append("max_retry_attempts cannot be negative")
        
        if config.get("connection_pool_size", 0) <= 0:
            errors.append("connection_pool_size must be positive")
        
        return errors
    
    def _validate_routellm_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate RouteLLM configuration."""
        errors = []
        
        threshold = config.get("threshold", 0)
        if threshold < 0 or threshold > 1:
            errors.append("RouteLLM threshold must be between 0 and 1")
        
        valid_router_types = ["mf", "bert", "causal_llm", "sw_ranking", "random"]
        router_type = config.get("router_type", "")
        if router_type and router_type not in valid_router_types:
            errors.append(f"Invalid router_type: {router_type}")
        
        return errors
    
    def _validate_template_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate template configuration."""
        errors = []
        
        valid_formats = ["jinja2", "poml", "hybrid", "plain"]
        default_format = config.get("default_format", "")
        if default_format and default_format not in valid_formats:
            errors.append(f"Invalid default_format: {default_format}")
        
        max_size = config.get("max_template_size_kb", 0)
        if max_size <= 0:
            errors.append("max_template_size_kb must be positive")
        
        return errors
    
    def _validate_feature_flags_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate feature flags configuration."""
        errors = []
        
        valid_backends = ["memory", "redis", "database", "file"]
        backend = config.get("backend", "")
        if backend and backend not in valid_backends:
            errors.append(f"Invalid feature flags backend: {backend}")
        
        return errors
    
    def _validate_monitoring_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate monitoring configuration."""
        errors = []
        
        if "logging" in config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            log_level = config["logging"].get("level", "")
            if log_level and log_level not in valid_levels:
                errors.append(f"Invalid log level: {log_level}")
        
        return errors
    
    def _check_config_conflicts(self, config: Dict[str, Any]) -> List[str]:
        """Check for configuration conflicts."""
        warnings = []
        
        # Check if RouteLLM is enabled but monitoring is disabled
        routellm_enabled = config.get("routellm", {}).get("enabled", False)
        monitoring_enabled = config.get("monitoring", {}).get("enabled", True)
        
        if routellm_enabled and not monitoring_enabled:
            warnings.append("RouteLLM is enabled but monitoring is disabled - consider enabling monitoring for cost tracking")
        
        # Check if POML caching is enabled but system caching is disabled
        template_caching = config.get("template_config", {}).get("enable_caching", False)
        system_caching = config.get("wrapper_framework", {}).get("enable_caching", True)
        
        if template_caching and not system_caching:
            warnings.append("Template caching is enabled but system caching is disabled")
        
        return warnings

# Usage
validator = ConfigurationValidator()
config = load_config_from_file("config.yaml")
result = validator.validate_complete_config(config)

if result.valid:
    print("✓ Configuration is valid")
else:
    print("❌ Configuration errors:")
    for error in result.errors:
        print(f"  - {error}")

if result.warnings:
    print("⚠️  Configuration warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
```

### Common Configuration Issues

```python
# troubleshooting.py
from typing import Dict, Any, List

class ConfigurationTroubleshooter:
    """Troubleshoot common configuration issues."""
    
    def diagnose_issues(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Diagnose common configuration issues."""
        issues = {
            "critical": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check for missing required configurations
        if not config.get("wrapper_framework", {}).get("enabled", False):
            issues["critical"].append("Wrapper framework is not enabled")
        
        # Check for performance issues
        timeout = config.get("wrapper_framework", {}).get("default_timeout_seconds", 30)
        if timeout < 5:
            issues["warnings"].append("Very low timeout may cause frequent failures")
        elif timeout > 120:
            issues["warnings"].append("Very high timeout may impact user experience")
        
        # Check for security issues
        ssl_validation = config.get("wrapper_framework", {}).get("validate_ssl_certificates", True)
        if not ssl_validation:
            issues["critical"].append("SSL certificate validation is disabled - security risk")
        
        sensitive_masking = config.get("wrapper_framework", {}).get("mask_sensitive_data_in_logs", True)
        if not sensitive_masking:
            issues["warnings"].append("Sensitive data masking is disabled - potential security issue")
        
        # Check RouteLLM configuration
        if config.get("routellm", {}).get("enabled", False):
            if not config.get("routellm", {}).get("cost_tracking_enabled", False):
                issues["suggestions"].append("Consider enabling cost tracking for RouteLLM")
            
            threshold = config.get("routellm", {}).get("threshold", 0.11593)
            if threshold > 0.2:
                issues["warnings"].append("High RouteLLM threshold may reduce cost savings")
        
        # Check monitoring configuration
        if not config.get("monitoring", {}).get("enabled", False):
            issues["warnings"].append("Monitoring is disabled - consider enabling for production")
        
        # Check cache configuration
        cache_size = config.get("wrapper_framework", {}).get("max_cache_size_mb", 100)
        if cache_size > 1000:
            issues["warnings"].append("Large cache size may impact memory usage")
        
        return issues
    
    def suggest_optimizations(self, config: Dict[str, Any]) -> List[str]:
        """Suggest configuration optimizations."""
        suggestions = []
        
        # Performance optimizations
        pool_size = config.get("wrapper_framework", {}).get("connection_pool_size", 100)
        concurrent_ops = config.get("wrapper_framework", {}).get("max_concurrent_operations", 50)
        
        if pool_size < concurrent_ops * 2:
            suggestions.append("Consider increasing connection pool size for better performance")
        
        # Cost optimization suggestions
        if config.get("routellm", {}).get("enabled", False):
            if not config.get("routellm", {}).get("domain_specific_routing", False):
                suggestions.append("Enable domain-specific routing for better RouteLLM optimization")
        
        # Monitoring optimizations
        if config.get("monitoring", {}).get("enabled", False):
            retention = config.get("monitoring", {}).get("metrics", {}).get("retention_days", 90)
            if retention < 30:
                suggestions.append("Consider longer metrics retention for better trend analysis")
        
        return suggestions

# Usage
troubleshooter = ConfigurationTroubleshooter()
issues = troubleshooter.diagnose_issues(config)
optimizations = troubleshooter.suggest_optimizations(config)

print("Configuration Diagnosis:")
for category, issue_list in issues.items():
    if issue_list:
        print(f"\n{category.upper()}:")
        for issue in issue_list:
            print(f"  - {issue}")

print("\nOptimization Suggestions:")
for suggestion in optimizations:
    print(f"  - {suggestion}")
```

### Configuration Testing

```python
# config_testing.py
import pytest
from unittest.mock import patch
import os

class TestConfigurationManagement:
    """Test configuration management functionality."""
    
    def test_environment_variable_overrides(self):
        """Test environment variable configuration overrides."""
        with patch.dict(os.environ, {
            "WRAPPER_ENABLED": "true",
            "WRAPPER_TIMEOUT": "45.0",
            "ROUTELLM_THRESHOLD": "0.12"
        }):
            loader = ConfigurationLoader()
            config = loader.load_config()
            
            assert config["wrapper_framework"]["enabled"] is True
            assert config["wrapper_framework"]["default_timeout_seconds"] == 45.0
            assert config["routellm"]["threshold"] == 0.12
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        validator = ConfigurationValidator()
        
        # Valid configuration
        valid_config = {
            "wrapper_framework": {
                "enabled": True,
                "default_timeout_seconds": 30.0,
                "max_retry_attempts": 3
            }
        }
        
        result = validator.validate_complete_config(valid_config)
        assert result.valid is True
        assert len(result.errors) == 0
        
        # Invalid configuration
        invalid_config = {
            "wrapper_framework": {
                "enabled": True,
                "default_timeout_seconds": -1,  # Invalid
                "max_retry_attempts": -5  # Invalid
            }
        }
        
        result = validator.validate_complete_config(invalid_config)
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_configuration_conflicts(self):
        """Test configuration conflict detection."""
        validator = ConfigurationValidator()
        
        conflicting_config = {
            "routellm": {"enabled": True},
            "monitoring": {"enabled": False}
        }
        
        result = validator.validate_complete_config(conflicting_config)
        assert len(result.warnings) > 0
        assert any("monitoring" in warning.lower() for warning in result.warnings)
```

This comprehensive configuration guide provides all the tools and knowledge needed to properly configure, validate, and manage wrapper framework settings across all environments and use cases.