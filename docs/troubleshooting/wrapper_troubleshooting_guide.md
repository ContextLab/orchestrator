# Wrapper Framework Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, debugging techniques, and resolution procedures for the wrapper framework, including RouteLLM integration, POML template processing, feature flags, monitoring, and performance issues.

## Table of Contents

- [Quick Diagnostic Tools](#quick-diagnostic-tools)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [RouteLLM Troubleshooting](#routellm-troubleshooting)
- [POML Template Troubleshooting](#poml-template-troubleshooting)
- [Feature Flag Issues](#feature-flag-issues)
- [Performance Problems](#performance-problems)
- [Configuration Issues](#configuration-issues)
- [Monitoring and Logging](#monitoring-and-logging)
- [Integration Problems](#integration-problems)
- [Advanced Debugging](#advanced-debugging)

## Quick Diagnostic Tools

### System Health Check

Use this script to quickly diagnose wrapper framework health:

```python
#!/usr/bin/env python3
# quick_diagnosis.py - Quick system health check

import asyncio
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Any

async def run_system_diagnosis():
    """Run comprehensive system diagnosis."""
    
    print("Wrapper Framework System Diagnosis")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    results = {
        "core_framework": await diagnose_core_framework(),
        "routellm": await diagnose_routellm(),
        "poml": await diagnose_poml(),
        "feature_flags": await diagnose_feature_flags(),
        "monitoring": await diagnose_monitoring(),
        "configuration": diagnose_configuration()
    }
    
    # Print summary
    print("\nDiagnosis Summary:")
    print("-" * 20)
    
    overall_status = "HEALTHY"
    for component, status in results.items():
        status_icon = "✅" if status["healthy"] else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}: {status['status']}")
        
        if not status["healthy"]:
            overall_status = "ISSUES FOUND"
        
        if status.get("warnings"):
            for warning in status["warnings"]:
                print(f"   ⚠️  {warning}")
        
        if status.get("errors"):
            for error in status["errors"]:
                print(f"   ❌ {error}")
    
    print(f"\nOverall System Status: {overall_status}")
    
    return results

async def diagnose_core_framework() -> Dict[str, Any]:
    """Diagnose core wrapper framework."""
    try:
        from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult
        from src.orchestrator.core.wrapper_config import BaseWrapperConfig
        
        result = {
            "healthy": True,
            "status": "Core framework components available",
            "warnings": [],
            "errors": []
        }
        
        # Check if core classes can be imported and instantiated
        assert BaseWrapper is not None
        assert WrapperResult is not None
        assert BaseWrapperConfig is not None
        
        return result
        
    except ImportError as e:
        return {
            "healthy": False,
            "status": "Import error",
            "errors": [f"Cannot import core framework: {e}"]
        }
    except Exception as e:
        return {
            "healthy": False,
            "status": "Unexpected error",
            "errors": [f"Core framework error: {e}"]
        }

async def diagnose_routellm() -> Dict[str, Any]:
    """Diagnose RouteLLM integration."""
    try:
        from src.orchestrator.models.routellm_integration import (
            RouteLLMConfig, FeatureFlags, CostTracker
        )
        
        result = {
            "healthy": True,
            "status": "RouteLLM components available",
            "warnings": [],
            "errors": []
        }
        
        # Test configuration
        config = RouteLLMConfig()
        if not config.api_key and config.enabled:
            result["warnings"].append("RouteLLM enabled but no API key configured")
        
        # Test feature flags
        flags = FeatureFlags()
        if not flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED):
            result["status"] = "RouteLLM disabled via feature flags"
        
        # Test cost tracker
        tracker = CostTracker()
        summary = tracker.get_metrics_summary()
        if summary["total_requests"] == 0:
            result["warnings"].append("No RouteLLM usage recorded")
        
        return result
        
    except ImportError as e:
        return {
            "healthy": False,
            "status": "RouteLLM not available",
            "errors": [f"Cannot import RouteLLM: {e}"]
        }
    except Exception as e:
        return {
            "healthy": False,
            "status": "RouteLLM error",
            "errors": [f"RouteLLM error: {e}"]
        }

async def diagnose_poml() -> Dict[str, Any]:
    """Diagnose POML template processing."""
    try:
        from src.orchestrator.core.template_resolver import TemplateResolver, POML_AVAILABLE
        
        result = {
            "healthy": True,
            "status": "POML components available",
            "warnings": [],
            "errors": []
        }
        
        if not POML_AVAILABLE:
            result["warnings"].append("POML library not installed - using built-in parser")
        
        # Test template resolver
        resolver = TemplateResolver(enable_poml_processing=True)
        
        # Test basic template resolution
        test_template = "<role>assistant</role><task>{{ task }}</task>"
        test_result = await resolver.resolve_template_content(
            test_template, {"task": "test"}
        )
        
        if "assistant" not in test_result or "test" not in test_result:
            result["errors"].append("Template resolution failed")
            result["healthy"] = False
        
        return result
        
    except ImportError as e:
        return {
            "healthy": False,
            "status": "POML not available",
            "errors": [f"Cannot import POML components: {e}"]
        }
    except Exception as e:
        return {
            "healthy": False,
            "status": "POML error",
            "errors": [f"POML error: {e}"]
        }

async def diagnose_feature_flags() -> Dict[str, Any]:
    """Diagnose feature flag system."""
    try:
        from src.orchestrator.core.feature_flags import FeatureFlagManager
        
        result = {
            "healthy": True,
            "status": "Feature flags system available",
            "warnings": [],
            "errors": []
        }
        
        # Test feature flag manager
        manager = FeatureFlagManager()
        
        # Test creating and evaluating a flag
        test_flag = "test_diagnosis_flag"
        manager.create_flag(test_flag, default_value=True)
        
        if not manager.is_enabled(test_flag):
            result["errors"].append("Feature flag evaluation failed")
            result["healthy"] = False
        
        # Check for common flags
        common_flags = ["routellm_enabled", "poml_processing_enabled"]
        missing_flags = []
        
        all_flags = manager.get_all_flags()
        for flag in common_flags:
            if flag not in all_flags:
                missing_flags.append(flag)
        
        if missing_flags:
            result["warnings"].append(f"Missing common flags: {missing_flags}")
        
        return result
        
    except ImportError as e:
        return {
            "healthy": False,
            "status": "Feature flags not available",
            "errors": [f"Cannot import feature flags: {e}"]
        }
    except Exception as e:
        return {
            "healthy": False,
            "status": "Feature flags error",
            "errors": [f"Feature flags error: {e}"]
        }

async def diagnose_monitoring() -> Dict[str, Any]:
    """Diagnose monitoring system."""
    try:
        from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring
        
        result = {
            "healthy": True,
            "status": "Monitoring system available",
            "warnings": [],
            "errors": []
        }
        
        # Test monitoring system
        monitoring = WrapperMonitoring()
        
        # Test operation tracking
        op_id = monitoring.start_operation("test_wrapper", "diagnosis")
        monitoring.record_success(op_id, {"test": "data"})
        monitoring.end_operation(op_id)
        
        # Check system health
        system_health = monitoring.get_system_health()
        if not system_health:
            result["errors"].append("Cannot retrieve system health")
            result["healthy"] = False
        
        return result
        
    except ImportError as e:
        return {
            "healthy": False,
            "status": "Monitoring not available",
            "errors": [f"Cannot import monitoring: {e}"]
        }
    except Exception as e:
        return {
            "healthy": False,
            "status": "Monitoring error",
            "errors": [f"Monitoring error: {e}"]
        }

def diagnose_configuration() -> Dict[str, Any]:
    """Diagnose configuration system."""
    import os
    
    result = {
        "healthy": True,
        "status": "Configuration system healthy",
        "warnings": [],
        "errors": []
    }
    
    # Check environment variables
    important_vars = [
        "WRAPPER_FRAMEWORK_ENABLED",
        "ROUTELLM_ENABLED", 
        "TEMPLATE_ENABLE_POML",
        "FEATURE_FLAGS_ENABLED"
    ]
    
    missing_vars = []
    for var in important_vars:
        if var not in os.environ:
            missing_vars.append(var)
    
    if missing_vars:
        result["warnings"].append(f"Missing environment variables: {missing_vars}")
    
    # Check for configuration files
    config_files = ["config.yaml", "config.json", ".env"]
    found_configs = []
    
    for config_file in config_files:
        if os.path.exists(config_file):
            found_configs.append(config_file)
    
    if not found_configs:
        result["warnings"].append("No configuration files found")
    else:
        result["status"] = f"Configuration files found: {found_configs}"
    
    return result

if __name__ == "__main__":
    try:
        asyncio.run(run_system_diagnosis())
    except Exception as e:
        print(f"Diagnosis failed: {e}")
        traceback.print_exc()
        sys.exit(1)
```

### Configuration Validator

```python
#!/usr/bin/env python3
# config_validator.py - Configuration validation tool

import yaml
import json
import os
from typing import Dict, Any, List, Optional

class ConfigurationValidator:
    """Validate wrapper framework configuration."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.suggestions = []
    
    def validate_config_file(self, config_path: str) -> Dict[str, Any]:
        """Validate configuration file."""
        print(f"Validating configuration: {config_path}")
        
        # Load configuration
        try:
            config = self._load_config(config_path)
        except Exception as e:
            self.errors.append(f"Cannot load config file: {e}")
            return self._create_result()
        
        # Validate sections
        self._validate_wrapper_framework(config.get("wrapper_framework", {}))
        self._validate_routellm(config.get("routellm", {}))
        self._validate_template_config(config.get("template_config", {}))
        self._validate_feature_flags(config.get("feature_flags", {}))
        self._validate_monitoring(config.get("monitoring", {}))
        
        # Check for conflicts
        self._check_conflicts(config)
        
        return self._create_result()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f) or {}
            elif config_path.endswith('.json'):
                return json.load(f) or {}
            else:
                raise ValueError("Unsupported configuration file format")
    
    def _validate_wrapper_framework(self, config: Dict[str, Any]):
        """Validate wrapper framework configuration."""
        if not config:
            self.warnings.append("No wrapper framework configuration found")
            return
        
        # Required fields
        if not config.get("enabled", True):
            self.warnings.append("Wrapper framework is disabled")
        
        # Timeout validation
        timeout = config.get("default_timeout_seconds", 30)
        if timeout <= 0:
            self.errors.append("default_timeout_seconds must be positive")
        elif timeout < 5:
            self.warnings.append("Very low timeout may cause frequent failures")
        elif timeout > 120:
            self.warnings.append("Very high timeout may impact user experience")
        
        # Connection pool validation
        pool_size = config.get("connection_pool_size", 100)
        if pool_size <= 0:
            self.errors.append("connection_pool_size must be positive")
        elif pool_size > 1000:
            self.warnings.append("Large connection pool may consume excessive resources")
        
        # Retry validation
        retries = config.get("max_retry_attempts", 3)
        if retries < 0:
            self.errors.append("max_retry_attempts cannot be negative")
        elif retries > 10:
            self.warnings.append("High retry attempts may cause delays")
    
    def _validate_routellm(self, config: Dict[str, Any]):
        """Validate RouteLLM configuration."""
        if not config:
            self.suggestions.append("Consider adding RouteLLM configuration for cost optimization")
            return
        
        if not config.get("enabled", False):
            self.suggestions.append("RouteLLM is disabled - consider enabling for cost savings")
            return
        
        # Threshold validation
        threshold = config.get("threshold", 0.11593)
        if threshold < 0 or threshold > 1:
            self.errors.append("RouteLLM threshold must be between 0 and 1")
        elif threshold > 0.5:
            self.warnings.append("High RouteLLM threshold may reduce cost savings")
        
        # Router type validation
        valid_routers = ["mf", "bert", "causal_llm", "sw_ranking", "random"]
        router_type = config.get("router_type", "mf")
        if router_type not in valid_routers:
            self.errors.append(f"Invalid router_type: {router_type}")
        
        # Model validation
        if not config.get("strong_model"):
            self.errors.append("strong_model is required when RouteLLM is enabled")
        if not config.get("weak_model"):
            self.errors.append("weak_model is required when RouteLLM is enabled")
        
        # Cost tracking validation
        if not config.get("cost_tracking_enabled", False):
            self.suggestions.append("Consider enabling cost tracking for RouteLLM")
    
    def _validate_template_config(self, config: Dict[str, Any]):
        """Validate template configuration."""
        if not config:
            self.suggestions.append("Consider adding template configuration")
            return
        
        # Format validation
        valid_formats = ["jinja2", "poml", "hybrid", "plain"]
        default_format = config.get("default_format", "hybrid")
        if default_format not in valid_formats:
            self.errors.append(f"Invalid default_format: {default_format}")
        
        # Size validation
        max_size = config.get("max_template_size_kb", 1024)
        if max_size <= 0:
            self.errors.append("max_template_size_kb must be positive")
        elif max_size > 10240:  # 10MB
            self.warnings.append("Large template size limit may impact performance")
        
        # Cache validation
        if config.get("enable_caching", True):
            cache_ttl = config.get("cache_ttl_seconds", 3600)
            if cache_ttl <= 0:
                self.errors.append("cache_ttl_seconds must be positive")
    
    def _validate_feature_flags(self, config: Dict[str, Any]):
        """Validate feature flags configuration."""
        if not config:
            self.suggestions.append("Consider adding feature flags configuration")
            return
        
        if not config.get("enabled", True):
            self.warnings.append("Feature flags system is disabled")
        
        # Backend validation
        valid_backends = ["memory", "redis", "database", "file"]
        backend = config.get("backend", "memory")
        if backend not in valid_backends:
            self.errors.append(f"Invalid feature flags backend: {backend}")
        
        # Strategy validation
        valid_strategies = ["boolean", "percentage", "whitelist", "blacklist", "custom"]
        default_strategy = config.get("default_strategy", "boolean")
        if default_strategy not in valid_strategies:
            self.errors.append(f"Invalid default_strategy: {default_strategy}")
    
    def _validate_monitoring(self, config: Dict[str, Any]):
        """Validate monitoring configuration."""
        if not config:
            self.warnings.append("No monitoring configuration - consider enabling for production")
            return
        
        if not config.get("enabled", True):
            self.warnings.append("Monitoring is disabled")
            return
        
        # Logging validation
        logging_config = config.get("logging", {})
        if logging_config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            log_level = logging_config.get("level", "INFO")
            if log_level not in valid_levels:
                self.errors.append(f"Invalid log level: {log_level}")
        
        # Metrics validation
        metrics_config = config.get("metrics", {})
        if metrics_config:
            retention = metrics_config.get("retention_days", 90)
            if retention <= 0:
                self.errors.append("metrics retention_days must be positive")
            elif retention < 7:
                self.warnings.append("Short metrics retention may limit analysis")
    
    def _check_conflicts(self, config: Dict[str, Any]):
        """Check for configuration conflicts."""
        # RouteLLM enabled but monitoring disabled
        routellm_enabled = config.get("routellm", {}).get("enabled", False)
        monitoring_enabled = config.get("monitoring", {}).get("enabled", True)
        
        if routellm_enabled and not monitoring_enabled:
            self.warnings.append("RouteLLM enabled but monitoring disabled - cost tracking unavailable")
        
        # POML caching enabled but system caching disabled
        template_caching = config.get("template_config", {}).get("enable_caching", False)
        system_caching = config.get("wrapper_framework", {}).get("enable_caching", True)
        
        if template_caching and not system_caching:
            self.warnings.append("Template caching enabled but system caching disabled")
        
        # High connection pool but low concurrent operations
        pool_size = config.get("wrapper_framework", {}).get("connection_pool_size", 100)
        concurrent_ops = config.get("wrapper_framework", {}).get("max_concurrent_operations", 50)
        
        if pool_size > concurrent_ops * 4:
            self.suggestions.append("Connection pool may be oversized for concurrent operations limit")
    
    def _create_result(self) -> Dict[str, Any]:
        """Create validation result."""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "summary": {
                "total_issues": len(self.errors) + len(self.warnings),
                "critical_issues": len(self.errors),
                "warnings": len(self.warnings),
                "suggestions": len(self.suggestions)
            }
        }

def main():
    """Main configuration validation function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    validator = ConfigurationValidator()
    result = validator.validate_config_file(config_file)
    
    print("\nValidation Results:")
    print("=" * 20)
    
    if result["valid"]:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors")
    
    if result["errors"]:
        print("\n🚨 ERRORS:")
        for error in result["errors"]:
            print(f"  ❌ {error}")
    
    if result["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in result["warnings"]:
            print(f"  ⚠️  {warning}")
    
    if result["suggestions"]:
        print("\n💡 SUGGESTIONS:")
        for suggestion in result["suggestions"]:
            print(f"  💡 {suggestion}")
    
    print(f"\nSummary: {result['summary']['critical_issues']} errors, {result['summary']['warnings']} warnings, {result['summary']['suggestions']} suggestions")

if __name__ == "__main__":
    main()
```

## Common Issues and Solutions

### Issue: Wrapper Framework Not Loading

**Symptoms:**
- Import errors when trying to use wrapper components
- "Module not found" errors
- Framework components not available

**Diagnosis:**
```python
# Check if framework is properly installed
try:
    from src.orchestrator.core.wrapper_base import BaseWrapper
    print("✅ Wrapper framework available")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

**Solutions:**

1. **Check Installation:**
```bash
pip list | grep orchestrator
pip show orchestrator-framework
```

2. **Reinstall Framework:**
```bash
pip uninstall orchestrator-framework
pip install orchestrator-framework
```

3. **Check Python Path:**
```python
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")
```

4. **Verify Environment:**
```bash
which python
python --version
pip --version
```

### Issue: Configuration Loading Failures

**Symptoms:**
- Configuration files not found
- Invalid configuration values
- Environment variables not applied

**Diagnosis:**
```python
import os
import yaml

# Check configuration file
config_file = "config.yaml"
if os.path.exists(config_file):
    print(f"✅ Config file exists: {config_file}")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Config file is valid YAML")
    except Exception as e:
        print(f"❌ Config file error: {e}")
else:
    print(f"❌ Config file not found: {config_file}")

# Check environment variables
env_vars = ["WRAPPER_ENABLED", "ROUTELLM_ENABLED", "POML_ENABLED"]
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value}")
    else:
        print(f"⚠️  {var}: Not set")
```

**Solutions:**

1. **Create Default Configuration:**
```yaml
# config.yaml
wrapper_framework:
  enabled: true
  default_timeout_seconds: 30.0
  max_retry_attempts: 3

routellm:
  enabled: false

template_config:
  enable_poml_processing: true

feature_flags:
  enabled: true

monitoring:
  enabled: true
```

2. **Set Environment Variables:**
```bash
export WRAPPER_ENABLED=true
export ROUTELLM_ENABLED=false
export POML_ENABLED=true
```

3. **Validate Configuration:**
```python
from src.orchestrator.core.wrapper_config import ConfigManager

manager = ConfigManager()
try:
    manager.validate_config(config)
    print("✅ Configuration is valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### Issue: Permission and Access Errors

**Symptoms:**
- "Permission denied" errors
- Cannot write to log files
- Cannot access configuration files

**Diagnosis:**
```bash
# Check file permissions
ls -la config.yaml
ls -la /var/log/

# Check user permissions
whoami
id

# Check directory permissions
ls -ld /var/log/
```

**Solutions:**

1. **Fix File Permissions:**
```bash
chmod 644 config.yaml
chmod 755 /var/log/wrapper/
```

2. **Create Log Directory:**
```bash
sudo mkdir -p /var/log/wrapper
sudo chown $USER:$USER /var/log/wrapper
```

3. **Use User Directory:**
```python
import os
log_dir = os.path.expanduser("~/.orchestrator/logs")
os.makedirs(log_dir, exist_ok=True)
```

## RouteLLM Troubleshooting

### Issue: RouteLLM Not Routing Correctly

**Symptoms:**
- All requests go to the same model
- Routing decisions seem incorrect
- Cost savings not realized

**Diagnosis:**
```python
from src.orchestrator.models.routellm_integration import (
    RouteLLMConfig, FeatureFlags, CostTracker
)

# Check configuration
config = RouteLLMConfig()
print(f"RouteLLM enabled: {config.enabled}")
print(f"Router type: {config.router_type}")
print(f"Threshold: {config.threshold}")
print(f"Strong model: {config.strong_model}")
print(f"Weak model: {config.weak_model}")

# Check feature flags
flags = FeatureFlags()
print(f"Global flag: {flags.is_enabled(flags.ROUTELLM_ENABLED)}")
print(f"Cost tracking: {flags.is_enabled(flags.ROUTELLM_COST_TRACKING)}")

# Check cost tracker
tracker = CostTracker()
summary = tracker.get_metrics_summary()
print(f"Total requests: {summary['total_requests']}")
print(f"RouteLLM usage rate: {summary['routellm_usage_rate']:.1%}")
```

**Solutions:**

1. **Enable RouteLLM:**
```python
flags = FeatureFlags()
flags.enable(FeatureFlags.ROUTELLM_ENABLED)
```

2. **Adjust Threshold:**
```python
config = RouteLLMConfig(
    enabled=True,
    threshold=0.15  # Higher threshold = more weak model usage
)
```

3. **Check Domain Configuration:**
```python
config = RouteLLMConfig(
    domain_specific_routing=True,
    domain_routing_overrides={
        "technical": {"enabled": True, "threshold": 0.12}
    }
)
```

### Issue: High RouteLLM Costs

**Symptoms:**
- Cost tracking shows increased costs
- Frequent strong model usage
- Poor routing efficiency

**Diagnosis:**
```python
# Analyze cost data
tracker = CostTracker()
report = tracker.get_cost_savings_report(period_days=7)

print(f"Estimated savings: ${report.estimated_savings:.2f}")
print(f"Savings percentage: {report.savings_percentage:.1f}%")
print(f"Strong model usage: {report.traditional_requests}/{report.total_requests}")

if report.estimated_savings < 0:
    print("⚠️  RouteLLM is increasing costs!")
```

**Solutions:**

1. **Increase Threshold:**
```python
config = RouteLLMConfig(threshold=0.20)  # More aggressive weak model usage
```

2. **Review Domain Settings:**
```python
# Check if domain overrides are too conservative
overrides = config.domain_routing_overrides
for domain, settings in overrides.items():
    if settings.get("threshold", 0) < 0.10:
        print(f"Domain {domain} has very conservative threshold")
```

3. **Analyze Routing Patterns:**
```python
# Look at recent routing decisions
metrics = tracker.metrics[-100:]  # Last 100 requests
strong_model_count = sum(1 for m in metrics if m.selected_model == config.strong_model)
print(f"Strong model usage: {strong_model_count/len(metrics):.1%}")
```

### Issue: RouteLLM Timeout Errors

**Symptoms:**
- Frequent timeout errors in routing decisions
- Fallback usage due to timeouts
- Slow routing performance

**Diagnosis:**
```python
# Check routing performance
tracker = CostTracker()
recent_metrics = tracker.metrics[-50:]
avg_latency = sum(m.routing_latency_ms for m in recent_metrics) / len(recent_metrics)
print(f"Average routing latency: {avg_latency:.1f}ms")

timeout_errors = [m for m in recent_metrics if "timeout" in (m.error_message or "").lower()]
print(f"Timeout error rate: {len(timeout_errors)/len(recent_metrics):.1%}")
```

**Solutions:**

1. **Increase Timeout:**
```python
config = RouteLLMConfig(timeout_seconds=60.0)  # Increase from default 30s
```

2. **Optimize Router Type:**
```python
# Try faster router
config = RouteLLMConfig(router_type=RouterType.RANDOM)  # For testing
```

3. **Enable Caching:**
```python
# Cache routing decisions for similar queries
config = RouteLLMConfig(enable_caching=True)
```

## POML Template Troubleshooting

### Issue: POML Templates Not Rendering

**Symptoms:**
- POML syntax appears in output
- Templates not processed
- Jinja2 fallback not working

**Diagnosis:**
```python
from src.orchestrator.core.template_resolver import (
    TemplateResolver, TemplateFormatDetector, POML_AVAILABLE
)

print(f"POML library available: {POML_AVAILABLE}")

# Test template resolution
resolver = TemplateResolver(enable_poml_processing=True)
test_template = "<role>assistant</role><task>{{ task }}</task>"
try:
    result = await resolver.resolve_template_content(
        test_template, {"task": "test"}
    )
    print(f"Template result: {result}")
except Exception as e:
    print(f"Template error: {e}")

# Test format detection
detector = TemplateFormatDetector()
format_detected = detector.detect_format(test_template)
print(f"Detected format: {format_detected}")
```

**Solutions:**

1. **Enable POML Processing:**
```python
resolver = TemplateResolver(
    enable_poml_processing=True,
    auto_detect_format=True,
    fallback_to_jinja=True
)
```

2. **Install POML Library:**
```bash
pip install poml
```

3. **Check Template Format:**
```python
# Ensure templates are valid
template = """
<role>assistant</role>
<task>{{ task_description }}</task>
<examples>
  <example>{{ example_text }}</example>
</examples>
"""
```

### Issue: Template Format Detection Errors

**Symptoms:**
- Wrong format detected
- Mixed templates not handled correctly
- Format detection inconsistent

**Diagnosis:**
```python
detector = TemplateFormatDetector()

test_templates = {
    "jinja_only": "Hello {{ name }}!",
    "poml_only": "<role>assistant</role>",
    "hybrid": "<role>{{ role }}</role>",
    "plain": "Just plain text"
}

for name, template in test_templates.items():
    detected = detector.detect_format(template)
    confidence = detector.get_format_confidence(template, detected)
    print(f"{name}: {detected.value} (confidence: {confidence:.2f})")
```

**Solutions:**

1. **Explicit Format Specification:**
```python
resolver = TemplateResolver()
result = await resolver.resolve_template_content(
    template_content,
    context,
    template_format=TemplateFormat.HYBRID
)
```

2. **Improve Template Structure:**
```python
# Make format clearer
template = """
<poml>
<role>{{ role_type }}</role>
<task>{{ task_description }}</task>
</poml>
"""
```

3. **Use Format Hints:**
```python
# Add format detection hints
if template.strip().startswith("<"):
    format_hint = TemplateFormat.POML
elif "{{" in template:
    format_hint = TemplateFormat.JINJA2
```

### Issue: Cross-Task References Not Working

**Symptoms:**
- `output_refs` variables not resolved
- Template references show as literal text
- Missing data from previous tasks

**Diagnosis:**
```python
from src.orchestrator.core.output_tracker import OutputTracker
from src.orchestrator.core.template_resolver import TemplateResolver

# Check output tracker
tracker = OutputTracker()
available_outputs = tracker.list_available_outputs()
print(f"Available outputs: {available_outputs}")

# Test cross-task resolution
resolver = TemplateResolver(cross_task_references=True)
template = "Data: {{ output_refs.previous_task.result }}"

try:
    result = await resolver.resolve_with_output_references(
        template, {}, tracker
    )
    print(f"Resolved: {result}")
except Exception as e:
    print(f"Resolution error: {e}")
```

**Solutions:**

1. **Enable Cross-Task References:**
```python
resolver = TemplateResolver(
    enable_advanced_resolution=True,
    cross_task_references=True
)
```

2. **Check Output Availability:**
```python
# Ensure previous task outputs are saved
tracker = OutputTracker()
tracker.save_output("task_name", {"result": "data"})
```

3. **Use Correct Reference Syntax:**
```python
template = """
<role>analyst</role>
<task>
Analyze: {{ output_refs.data_processing.cleaned_data }}
Based on: {{ output_refs.initial_analysis.summary }}
</task>
"""
```

## Feature Flag Issues

### Issue: Feature Flags Not Evaluating Correctly

**Symptoms:**
- Flags always return default values
- Runtime updates not taking effect
- Inconsistent flag behavior

**Diagnosis:**
```python
from src.orchestrator.core.feature_flags import FeatureFlagManager

manager = FeatureFlagManager()

# Check flag registration
all_flags = manager.get_all_flags()
print(f"Registered flags: {list(all_flags.keys())}")

# Test flag evaluation
test_flag = "test_flag"
manager.create_flag(test_flag, default_value=True)
is_enabled = manager.is_enabled(test_flag)
print(f"Flag '{test_flag}' enabled: {is_enabled}")

# Check flag configuration
flag_info = manager.get_flag_info(test_flag)
print(f"Flag info: {flag_info}")
```

**Solutions:**

1. **Check Flag Registration:**
```python
# Ensure flags are properly registered
flag_manager = FeatureFlagManager()
flag_manager.create_flag("my_feature", default_value=False)
```

2. **Verify Flag Updates:**
```python
# Check if updates are applied
flag_manager.update_flag("my_feature", True)
time.sleep(0.1)  # Allow for update propagation
assert flag_manager.is_enabled("my_feature")
```

3. **Check Context Evaluation:**
```python
# Provide context for complex flags
context = {"user_id": "123", "region": "us-west"}
is_enabled = flag_manager.is_enabled("feature_flag", context)
```

### Issue: Percentage-Based Rollouts Not Working

**Symptoms:**
- Percentage flags not distributing correctly
- All users get same result
- Inconsistent rollout behavior

**Diagnosis:**
```python
from src.orchestrator.core.feature_flags import FeatureFlag, FeatureFlagStrategy

# Create percentage flag
flag = FeatureFlag(
    name="rollout_test",
    strategy=FeatureFlagStrategy.PERCENTAGE,
    default_value=50  # 50% rollout
)

manager = FeatureFlagManager()
manager.register_flag(flag)

# Test with different contexts
results = []
for i in range(100):
    context = {"user_id": str(i)}
    enabled = manager.is_enabled("rollout_test", context)
    results.append(enabled)

enabled_percentage = sum(results) / len(results) * 100
print(f"Actual rollout percentage: {enabled_percentage:.1f}%")
```

**Solutions:**

1. **Provide Consistent Context:**
```python
# Use stable user identifier
context = {"user_id": user.id}  # Consistent across requests
is_enabled = flag_manager.is_enabled("feature", context)
```

2. **Check Hash Function:**
```python
# Ensure hash function is working
import hashlib
user_id = "test_user"
hash_value = hashlib.md5(f"rollout_test:{user_id}".encode()).hexdigest()
hash_int = int(hash_value[:8], 16)
percentage = (hash_int % 100)
print(f"Hash percentage for {user_id}: {percentage}%")
```

3. **Adjust Rollout Configuration:**
```python
flag = FeatureFlag(
    name="feature_rollout",
    strategy=FeatureFlagStrategy.PERCENTAGE,
    default_value=25,  # Start with 25%
    metadata={"increment": 5}  # Increase by 5% steps
)
```

## Performance Problems

### Issue: Slow Wrapper Response Times

**Symptoms:**
- High latency in wrapper operations
- Timeouts occurring frequently
- Poor user experience

**Diagnosis:**
```python
import time
import asyncio
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

# Monitor wrapper performance
monitoring = WrapperMonitoring()

async def diagnose_performance():
    # Test wrapper performance
    start_time = time.time()
    
    # Simulate wrapper operation
    op_id = monitoring.start_operation("test_wrapper", "performance_test")
    
    # Add artificial delay to simulate slow operation
    await asyncio.sleep(0.1)
    
    monitoring.record_success(op_id, {"test": "data"})
    monitoring.end_operation(op_id)
    
    duration = time.time() - start_time
    print(f"Operation took {duration*1000:.1f}ms")
    
    # Check wrapper health
    health = monitoring.get_wrapper_health("test_wrapper")
    print(f"Average response time: {health.avg_response_time_ms:.1f}ms")

asyncio.run(diagnose_performance())
```

**Solutions:**

1. **Optimize Connection Pooling:**
```python
import aiohttp

# Use connection pooling
connector = aiohttp.TCPConnector(
    limit=100,  # Total connection pool size
    limit_per_host=20,  # Per-host connections
    ttl_dns_cache=300,  # DNS cache TTL
    use_dns_cache=True
)

session = aiohttp.ClientSession(connector=connector)
```

2. **Enable Caching:**
```python
from src.orchestrator.core.wrapper_base import BaseWrapper

class CachedWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.cache = {}  # Simple in-memory cache
    
    async def _execute_wrapper_operation(self, context, **kwargs):
        cache_key = self._generate_cache_key(**kwargs)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await self._actual_operation(**kwargs)
        self.cache[cache_key] = result
        return result
```

3. **Implement Async Patterns:**
```python
import asyncio

# Use asyncio.gather for concurrent operations
async def parallel_operations():
    tasks = [
        wrapper1.execute("operation1"),
        wrapper2.execute("operation2"),
        wrapper3.execute("operation3")
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### Issue: Memory Usage Issues

**Symptoms:**
- High memory consumption
- Memory leaks over time
- Out of memory errors

**Diagnosis:**
```python
import psutil
import gc

def diagnose_memory():
    """Diagnose memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"RSS Memory: {memory_info.rss / 1024 / 1024:.1f}MB")
    print(f"VMS Memory: {memory_info.vms / 1024 / 1024:.1f}MB")
    
    # Check garbage collection
    gc.collect()
    print(f"Garbage collection stats: {gc.get_stats()}")
    
    # Check object counts
    import sys
    print(f"Reference count: {sys.getrefcount}")

diagnose_memory()
```

**Solutions:**

1. **Implement Proper Cleanup:**
```python
class MemoryEfficientWrapper(BaseWrapper):
    async def cleanup(self):
        """Clean up resources properly."""
        if hasattr(self, '_session') and self._session:
            await self._session.close()
        
        if hasattr(self, '_cache'):
            self._cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
```

2. **Use Bounded Collections:**
```python
from collections import deque

class BoundedCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.access_order = deque(maxlen=max_size)
        self.max_size = max_size
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove oldest item
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]
        
        self.cache[key] = value
        self.access_order.append(key)
```

3. **Monitor Memory Usage:**
```python
import psutil
import logging

class MemoryMonitor:
    def __init__(self, threshold_mb=500):
        self.threshold_mb = threshold_mb
        self.process = psutil.Process()
    
    def check_memory(self):
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.threshold_mb:
            logging.warning(f"High memory usage: {memory_mb:.1f}MB")
            
            # Trigger cleanup
            gc.collect()
            
        return memory_mb
```

## Configuration Issues

### Issue: Environment Variables Not Applied

**Symptoms:**
- Configuration values not updated from environment
- Default values always used
- Environment-specific settings ignored

**Diagnosis:**
```bash
# Check environment variables
env | grep WRAPPER
env | grep ROUTELLM
env | grep TEMPLATE
env | grep FEATURE_FLAGS

# Check in Python
python -c "import os; print({k:v for k,v in os.environ.items() if 'WRAPPER' in k})"
```

**Solutions:**

1. **Verify Environment Variable Names:**
```python
import os

# Check expected variable names
expected_vars = [
    "WRAPPER_FRAMEWORK_ENABLED",
    "ROUTELLM_ENABLED",
    "TEMPLATE_ENABLE_POML",
    "MONITORING_ENABLED"
]

for var in expected_vars:
    value = os.getenv(var)
    print(f"{var}: {value if value is not None else 'NOT SET'}")
```

2. **Check Variable Loading:**
```python
from src.orchestrator.core.wrapper_config import ConfigManager

manager = ConfigManager()
config = manager.load_from_environment()
print("Environment config:", config)
```

3. **Use dotenv for Development:**
```python
from dotenv import load_dotenv
load_dotenv()  # Load .env file

import os
print("WRAPPER_ENABLED:", os.getenv("WRAPPER_ENABLED"))
```

### Issue: Configuration File Format Errors

**Symptoms:**
- YAML parsing errors
- Invalid JSON format
- Configuration not loading

**Diagnosis:**
```python
import yaml
import json

def validate_config_file(filename):
    """Validate configuration file format."""
    try:
        with open(filename, 'r') as f:
            if filename.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
                print(f"✅ Valid YAML: {filename}")
            elif filename.endswith('.json'):
                config = json.load(f)
                print(f"✅ Valid JSON: {filename}")
            else:
                print(f"❓ Unknown format: {filename}")
            
            return config
    except Exception as e:
        print(f"❌ Invalid format: {e}")
        return None

# Test configuration files
validate_config_file("config.yaml")
```

**Solutions:**

1. **Fix YAML Syntax:**
```yaml
# Correct YAML format
wrapper_framework:
  enabled: true  # Boolean, not "true"
  timeout_seconds: 30.0  # Float, not "30.0"
  
routellm:
  enabled: false
  threshold: 0.11593
```

2. **Validate JSON:**
```json
{
  "wrapper_framework": {
    "enabled": true,
    "timeout_seconds": 30.0
  },
  "routellm": {
    "enabled": false,
    "threshold": 0.11593
  }
}
```

3. **Use Configuration Validation:**
```python
import yaml

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Validate structure
    required_sections = ['wrapper_framework', 'routellm', 'monitoring']
    for section in required_sections:
        if section not in config:
            print(f"⚠️  Missing section: {section}")
        
except yaml.YAMLError as e:
    print(f"YAML Error: {e}")
```

## Monitoring and Logging

### Issue: Logs Not Appearing

**Symptoms:**
- No log output
- Log files not created
- Missing log entries

**Diagnosis:**
```python
import logging
import os

# Check logging configuration
logger = logging.getLogger("orchestrator")
print(f"Logger level: {logger.level}")
print(f"Logger handlers: {logger.handlers}")

# Check log file permissions
log_file = "/var/log/wrapper.log"
if os.path.exists(log_file):
    print(f"✅ Log file exists: {log_file}")
    print(f"Permissions: {oct(os.stat(log_file).st_mode)[-3:]}")
else:
    print(f"❌ Log file not found: {log_file}")

# Test logging
logger.info("Test log message")
logger.error("Test error message")
```

**Solutions:**

1. **Configure Logging Properly:**
```python
import logging

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wrapper.log'),
        logging.StreamHandler()  # Also log to console
    ]
)

logger = logging.getLogger("wrapper")
logger.info("Logging configured")
```

2. **Check File Permissions:**
```bash
# Create log directory with proper permissions
sudo mkdir -p /var/log/wrapper
sudo chown $USER:$USER /var/log/wrapper
sudo chmod 755 /var/log/wrapper
```

3. **Use Alternative Log Location:**
```python
import os
import logging

# Use user home directory for logs
log_dir = os.path.expanduser("~/.orchestrator/logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "wrapper.log")
logging.basicConfig(filename=log_file, level=logging.INFO)
```

### Issue: Monitoring Metrics Not Collected

**Symptoms:**
- No metrics data available
- Health checks failing
- Monitoring dashboard empty

**Diagnosis:**
```python
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

monitoring = WrapperMonitoring()

# Check if monitoring is enabled
print(f"Monitoring enabled: {monitoring.enabled}")

# Test metric collection
op_id = monitoring.start_operation("test_wrapper", "diagnosis")
monitoring.record_success(op_id, {"test": "data"})
monitoring.end_operation(op_id)

# Check collected metrics
health = monitoring.get_wrapper_health("test_wrapper")
print(f"Wrapper health: {health}")

system_health = monitoring.get_system_health()
print(f"System health: {system_health.overall_status}")
```

**Solutions:**

1. **Enable Monitoring:**
```python
monitoring = WrapperMonitoring(enabled=True)
```

2. **Check Monitoring Configuration:**
```yaml
monitoring:
  enabled: true
  metrics:
    enabled: true
    collection_interval_seconds: 30
  health_checks:
    enabled: true
```

3. **Verify Metrics Storage:**
```python
# Check if metrics are being stored
monitoring = WrapperMonitoring()
metrics_count = len(monitoring.get_all_metrics())
print(f"Stored metrics: {metrics_count}")

if metrics_count == 0:
    print("⚠️  No metrics collected - check wrapper operations")
```

## Integration Problems

### Issue: External API Connection Failures

**Symptoms:**
- Connection refused errors
- SSL certificate errors
- Authentication failures

**Diagnosis:**
```python
import aiohttp
import asyncio

async def test_api_connection():
    """Test external API connectivity."""
    test_url = "https://api.example.com/health"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                print(f"✅ API accessible: {response.status}")
                return True
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Connection error: {e}")
    except aiohttp.ClientResponseError as e:
        print(f"❌ HTTP error: {e.status}")
    except asyncio.TimeoutError:
        print("❌ Timeout error")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return False

asyncio.run(test_api_connection())
```

**Solutions:**

1. **Check Network Connectivity:**
```bash
# Test basic connectivity
curl -I https://api.example.com/health
ping api.example.com
nslookup api.example.com
```

2. **Verify SSL Configuration:**
```python
import ssl
import aiohttp

# Create SSL context
ssl_context = ssl.create_default_context()
# For development only - don't use in production
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE

session = aiohttp.ClientSession(
    connector=aiohttp.TCPConnector(ssl=ssl_context)
)
```

3. **Check Authentication:**
```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

async with session.get(url, headers=headers) as response:
    if response.status == 401:
        print("❌ Authentication failed - check API key")
    elif response.status == 403:
        print("❌ Authorization failed - check permissions")
```

### Issue: Database Connection Problems

**Symptoms:**
- Cannot connect to database
- Connection pool exhausted
- Timeout errors

**Diagnosis:**
```python
import asyncio
import asyncpg  # Example with PostgreSQL

async def test_database_connection():
    """Test database connectivity."""
    connection_string = "postgresql://user:pass@localhost:5432/db"
    
    try:
        conn = await asyncpg.connect(connection_string)
        result = await conn.fetchval("SELECT 1")
        await conn.close()
        
        print(f"✅ Database accessible: {result}")
        return True
    except asyncpg.PostgresConnectionError as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    return False

asyncio.run(test_database_connection())
```

**Solutions:**

1. **Verify Connection String:**
```python
import os

# Check connection parameters
db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "database": os.getenv("DB_NAME", "orchestrator"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

print("Database configuration:", {k: v for k, v in db_config.items() if k != "password"})
```

2. **Configure Connection Pool:**
```python
import asyncpg

# Create connection pool
pool = await asyncpg.create_pool(
    connection_string,
    min_size=5,
    max_size=20,
    command_timeout=30
)
```

3. **Test Database Access:**
```bash
# Test with command line tools
psql -h localhost -U postgres -d orchestrator -c "SELECT 1;"
```

## Advanced Debugging

### Enabling Debug Mode

```python
# debug_mode.py - Enable comprehensive debugging

import logging
import os
import sys
from typing import Dict, Any

def enable_debug_mode():
    """Enable comprehensive debug mode for wrapper framework."""
    
    # Set debug environment variables
    debug_vars = {
        "WRAPPER_DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "PYTHONPATH": os.getcwd()
    }
    
    for key, value in debug_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    # Configure debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log', mode='w')
        ]
    )
    
    # Enable asyncio debug mode
    if hasattr(asyncio, 'get_event_loop'):
        loop = asyncio.get_event_loop()
        loop.set_debug(True)
    
    print("Debug mode enabled")

def create_debug_wrapper(wrapper_class):
    """Create debug version of wrapper with extensive logging."""
    
    class DebugWrapper(wrapper_class):
        async def execute(self, operation_type: str, **kwargs):
            logger = logging.getLogger(f"{self.__class__.__name__}.debug")
            logger.debug(f"Executing {operation_type} with args: {kwargs}")
            
            try:
                result = await super().execute(operation_type, **kwargs)
                logger.debug(f"Execution successful: {result.success}")
                if not result.success:
                    logger.debug(f"Error: {result.error}")
                    logger.debug(f"Fallback used: {result.fallback_used}")
                return result
            except Exception as e:
                logger.debug(f"Execution failed: {e}", exc_info=True)
                raise
    
    return DebugWrapper

# Usage
if __name__ == "__main__":
    enable_debug_mode()
```

### Performance Profiling

```python
# profiling.py - Performance profiling tools

import asyncio
import cProfile
import pstats
import time
from typing import Dict, List, Any

class PerformanceProfiler:
    """Performance profiler for wrapper operations."""
    
    def __init__(self):
        self.profile_data = []
    
    async def profile_wrapper_operation(self, wrapper, operation_type: str, **kwargs):
        """Profile a single wrapper operation."""
        
        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        
        try:
            result = await wrapper.execute(operation_type, **kwargs)
            success = result.success
            error = result.error
        except Exception as e:
            success = False
            error = str(e)
            result = None
        
        end_time = time.time()
        profiler.disable()
        
        # Collect profile data
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        profile_info = {
            "operation_type": operation_type,
            "duration_ms": (end_time - start_time) * 1000,
            "success": success,
            "error": error,
            "profile_stats": stats
        }
        
        self.profile_data.append(profile_info)
        return profile_info
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        if not self.profile_data:
            return {"error": "No profile data available"}
        
        # Calculate statistics
        durations = [p["duration_ms"] for p in self.profile_data]
        success_rate = sum(1 for p in self.profile_data if p["success"]) / len(self.profile_data)
        
        report = {
            "total_operations": len(self.profile_data),
            "success_rate": success_rate,
            "average_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "slow_operations": [
                p for p in self.profile_data 
                if p["duration_ms"] > sum(durations) / len(durations) * 2
            ]
        }
        
        return report

# Memory profiling
import tracemalloc

class MemoryProfiler:
    """Memory usage profiler."""
    
    def __init__(self):
        tracemalloc.start()
        self.snapshots = []
    
    def take_snapshot(self, label: str = ""):
        """Take memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            "label": label,
            "timestamp": time.time(),
            "snapshot": snapshot
        })
    
    def compare_snapshots(self, idx1: int = 0, idx2: int = -1):
        """Compare two memory snapshots."""
        if len(self.snapshots) < 2:
            return "Need at least 2 snapshots to compare"
        
        snap1 = self.snapshots[idx1]["snapshot"]
        snap2 = self.snapshots[idx2]["snapshot"]
        
        top_stats = snap2.compare_to(snap1, 'lineno')
        
        print("Top 10 memory differences:")
        for stat in top_stats[:10]:
            print(stat)

# Usage example
async def profile_wrapper_performance():
    """Example of profiling wrapper performance."""
    from your_wrapper import YourWrapper, YourWrapperConfig
    
    # Setup profiler
    profiler = PerformanceProfiler()
    memory_profiler = MemoryProfiler()
    
    # Create wrapper
    config = YourWrapperConfig()
    wrapper = YourWrapper(config)
    
    try:
        memory_profiler.take_snapshot("before_operations")
        
        # Profile multiple operations
        operations = [
            ("test_op1", {"param1": "value1"}),
            ("test_op2", {"param2": "value2"}),
            ("test_op3", {"param3": "value3"})
        ]
        
        for op_type, kwargs in operations:
            await profiler.profile_wrapper_operation(wrapper, op_type, **kwargs)
        
        memory_profiler.take_snapshot("after_operations")
        
        # Generate reports
        performance_report = profiler.generate_performance_report()
        print("Performance Report:", performance_report)
        
        # Memory comparison
        memory_profiler.compare_snapshots(0, 1)
        
    finally:
        await wrapper.cleanup()

if __name__ == "__main__":
    asyncio.run(profile_wrapper_performance())
```

This comprehensive troubleshooting guide provides systematic approaches to diagnosing and resolving common issues with the wrapper framework, ensuring reliable operation across all components and environments.