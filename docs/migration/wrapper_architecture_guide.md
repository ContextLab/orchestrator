# Wrapper Architecture Migration Guide

## Overview

This guide provides comprehensive instructions for migrating to the unified wrapper architecture in the orchestrator framework. The wrapper architecture provides a standardized approach to integrating external tools while maintaining backward compatibility, comprehensive error handling, and robust monitoring capabilities.

## What is the Wrapper Architecture?

The unified wrapper architecture is a comprehensive framework that provides:

- **Standardized Integration**: Consistent patterns for integrating external tools
- **Feature Flag Management**: Safe rollout and A/B testing capabilities  
- **Configuration System**: Unified configuration with validation and overrides
- **Monitoring and Metrics**: Built-in health checking and performance tracking
- **Error Handling**: Comprehensive fallback mechanisms
- **Type Safety**: Full generic type support with IDE assistance

## Pre-Migration Assessment

### System Requirements

Before starting the migration, ensure your system meets these requirements:

```bash
# Check Python version (3.8+ required)
python --version

# Check orchestrator version
pip show orchestrator-framework

# Verify wrapper framework components are available
python -c "from src.orchestrator.core.wrapper_base import BaseWrapper; print('Wrapper framework available')"
```

### Integration Assessment

Assess your current external tool integrations:

```python
# integration_assessment.py
import inspect
import importlib
from typing import List, Dict, Any

def assess_current_integrations():
    """Assess current external tool integrations for wrapper migration."""
    
    assessment = {
        "external_apis": [],
        "model_integrations": [],
        "tool_integrations": [],
        "custom_handlers": [],
        "total_integrations": 0
    }
    
    # Common integration patterns to look for
    integration_patterns = [
        ("External API calls", ["requests", "httpx", "aiohttp"]),
        ("Model integrations", ["openai", "anthropic", "transformers"]),
        ("Database connections", ["sqlite3", "psycopg2", "pymongo"]),
        ("Cloud services", ["boto3", "azure", "google-cloud"]),
        ("Custom tools", ["custom_tool", "external_tool"])
    ]
    
    print("Current Integration Assessment")
    print("=" * 40)
    
    for pattern_name, modules in integration_patterns:
        found_modules = []
        for module in modules:
            try:
                importlib.import_module(module)
                found_modules.append(module)
            except ImportError:
                pass
        
        if found_modules:
            assessment[pattern_name.lower().replace(" ", "_")] = found_modules
            assessment["total_integrations"] += len(found_modules)
            print(f"✓ {pattern_name}: {', '.join(found_modules)}")
        else:
            print(f"- {pattern_name}: Not found")
    
    return assessment

def identify_wrapper_candidates():
    """Identify code that would benefit from wrapper architecture."""
    
    candidates = []
    
    # Look for common patterns that suggest external integrations
    patterns = [
        "def call_external_api",
        "async def fetch_data", 
        "class.*Client",
        "def process_with_",
        "async def query_"
    ]
    
    print(f"\nWrapper Migration Candidates:")
    print("=" * 40)
    print("Look for these patterns in your codebase:")
    for pattern in patterns:
        print(f"- {pattern}")
    
    print("\nManual review recommended for:")
    print("- Functions that call external APIs")
    print("- Classes that wrap external services")  
    print("- Async operations with external dependencies")
    print("- Error-prone external integrations")
    
    return candidates

if __name__ == "__main__":
    assessment = assess_current_integrations()
    identify_wrapper_candidates()
    
    print(f"\nSummary: {assessment['total_integrations']} integrations found")
    print("Next: Review code for wrapper migration candidates")
```

### Compatibility Checklist

- [ ] **Python 3.8+**: Wrapper framework requires Python 3.8 or higher
- [ ] **Async Support**: Code uses async/await patterns (or can be converted)
- [ ] **Type Hints**: Code uses type hints (recommended for full benefits)
- [ ] **Configuration**: Current configuration system is compatible
- [ ] **Testing**: Test framework can accommodate wrapper testing patterns
- [ ] **Monitoring**: Optional monitoring system integration available

## Migration Strategies

### Strategy 1: Incremental Migration (Recommended)

Migrate integrations one by one while maintaining existing functionality.

**Benefits:**
- Zero downtime migration  
- Gradual learning curve
- Easy rollback for specific integrations
- Continuous validation and monitoring

**Process:**
1. Identify highest-priority integration for migration
2. Create wrapper implementation alongside existing code
3. Add feature flags for controlled rollout
4. Gradually shift traffic to wrapper implementation
5. Remove old implementation after validation

### Strategy 2: New Integrations Only

Apply wrapper architecture to new integrations while leaving existing ones unchanged.

**Benefits:**
- No risk to existing functionality
- Immediate benefits for new features
- Gradual architecture evolution

**Considerations:**
- Mixed architecture patterns in codebase
- Requires maintaining both approaches
- Benefits realized slowly

### Strategy 3: Comprehensive Migration

Migrate all integrations to wrapper architecture at once.

**Benefits:**
- Consistent architecture across entire system
- Immediate access to all wrapper benefits
- Single migration effort

**Considerations:**
- Higher risk and complexity
- Requires comprehensive testing
- More complex rollback procedures

## Migration Steps

### Step 1: Set Up Wrapper Framework

Install and configure the wrapper framework components:

```python
# wrapper_setup.py
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField
from src.orchestrator.core.feature_flags import FeatureFlagManager
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

def setup_wrapper_framework():
    """Initialize wrapper framework components."""
    
    print("Setting up wrapper framework...")
    
    # Initialize feature flag manager
    flag_manager = FeatureFlagManager()
    print("✓ Feature flag manager initialized")
    
    # Initialize monitoring
    monitoring = WrapperMonitoring()
    print("✓ Wrapper monitoring initialized")
    
    # Verify core components
    assert BaseWrapper is not None
    assert WrapperResult is not None
    assert BaseWrapperConfig is not None
    print("✓ Core wrapper components available")
    
    return {
        "flag_manager": flag_manager,
        "monitoring": monitoring
    }

if __name__ == "__main__":
    setup_wrapper_framework()
```

### Step 2: Create Your First Wrapper

Start with a simple integration to learn the patterns:

```python
# example_wrapper.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import asyncio
import aiohttp
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult, WrapperContext
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField

@dataclass
class ExampleAPIConfig(BaseWrapperConfig):
    """Configuration for example API wrapper."""
    
    api_endpoint: str = "https://api.example.com"
    api_key: str = ""
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Define configuration fields with validation."""
        return {
            "api_endpoint": ConfigField(
                name="api_endpoint",
                field_type=str, 
                default_value=self.api_endpoint,
                description="API endpoint URL",
                required=True
            ),
            "api_key": ConfigField(
                name="api_key",
                field_type=str,
                default_value="",
                description="API authentication key",
                required=True,
                sensitive=True,
                environment_var="EXAMPLE_API_KEY"
            ),
            "timeout_seconds": ConfigField(
                name="timeout_seconds",
                field_type=float,
                default_value=self.timeout_seconds,
                description="Request timeout in seconds",
                min_value=1.0,
                max_value=300.0
            )
        }

class ExampleAPIWrapper(BaseWrapper[Dict[str, Any], ExampleAPIConfig]):
    """Example wrapper for external API integration."""
    
    def __init__(self, config: Optional[ExampleAPIConfig] = None):
        """Initialize wrapper with configuration."""
        config = config or ExampleAPIConfig()
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _execute_wrapper_operation(
        self, 
        context: WrapperContext, 
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute external API call through wrapper."""
        
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            **kwargs
        }
        
        async with self._session.post(
            f"{self.config.api_endpoint}/query",
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            return {
                "success": True,
                "data": result,
                "status_code": response.status
            }
    
    async def _execute_fallback_operation(
        self,
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        query: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute fallback when wrapper operation fails."""
        
        # Simple fallback - could be more sophisticated
        return {
            "success": False,
            "error": str(original_error) if original_error else "Unknown error",
            "fallback": True,
            "data": {"message": f"Fallback response for query: {query}"}
        }
    
    def _validate_config(self) -> bool:
        """Validate wrapper configuration."""
        return bool(
            self.config.api_endpoint and 
            self.config.api_key and
            self.config.timeout_seconds > 0
        )
    
    def get_capabilities(self) -> list:
        """Return wrapper capabilities."""
        from src.orchestrator.core.wrapper_base import WrapperCapability
        return [
            WrapperCapability.MONITORING,
            WrapperCapability.FALLBACK,
            WrapperCapability.CONFIGURATION_MANAGEMENT
        ]
    
    async def cleanup(self):
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None

# Example usage
async def test_example_wrapper():
    """Test the example wrapper implementation."""
    
    config = ExampleAPIConfig(
        api_endpoint="https://httpbin.org",  # Test endpoint
        api_key="test_key_123",
        timeout_seconds=10.0
    )
    
    wrapper = ExampleAPIWrapper(config)
    
    try:
        # Test wrapper operation
        result = await wrapper.execute(
            operation_type="query",
            query="test query",
            additional_param="test_value"
        )
        
        print(f"Wrapper result: {result}")
        
        # Check result structure
        assert isinstance(result, WrapperResult)
        print(f"Success: {result.success}")
        print(f"Data: {result.data}")
        if result.fallback_used:
            print(f"Fallback reason: {result.fallback_reason}")
        
    finally:
        await wrapper.cleanup()

if __name__ == "__main__":
    asyncio.run(test_example_wrapper())
```

### Step 3: Add Feature Flag Control

Add feature flags for safe rollout:

```python
# feature_flag_setup.py
from src.orchestrator.core.feature_flags import FeatureFlagManager, FeatureFlag, FeatureFlagStrategy

def setup_wrapper_feature_flags():
    """Set up feature flags for wrapper rollout."""
    
    flag_manager = FeatureFlagManager()
    
    # Core wrapper flags
    wrapper_flags = [
        FeatureFlag(
            name="example_wrapper_enabled",
            description="Enable example API wrapper",
            strategy=FeatureFlagStrategy.BOOLEAN,
            default_value=False,
            metadata={"rollout_phase": "testing"}
        ),
        FeatureFlag(
            name="example_wrapper_monitoring",
            description="Enable monitoring for example wrapper", 
            strategy=FeatureFlagStrategy.BOOLEAN,
            default_value=True,
            dependencies=["example_wrapper_enabled"]
        ),
        FeatureFlag(
            name="example_wrapper_percentage",
            description="Percentage rollout for example wrapper",
            strategy=FeatureFlagStrategy.PERCENTAGE,
            default_value=0,  # Start with 0%
            metadata={"target_percentage": 100}
        )
    ]
    
    # Register flags
    for flag in wrapper_flags:
        flag_manager.register_flag(flag)
    
    print("Feature flags registered:")
    for flag in wrapper_flags:
        print(f"  - {flag.name}: {flag.description}")
    
    return flag_manager

async def gradual_rollout_example():
    """Example of gradual wrapper rollout using feature flags."""
    
    flag_manager = setup_wrapper_feature_flags()
    
    # Phase 1: Enable for monitoring only
    print("\n=== Phase 1: Monitoring Only ===")
    flag_manager.update_flag("example_wrapper_monitoring", True)
    flag_manager.update_flag("example_wrapper_enabled", False)
    
    # Phase 2: 10% rollout  
    print("\n=== Phase 2: 10% Rollout ===")
    flag_manager.update_flag("example_wrapper_enabled", True)
    flag_manager.update_flag("example_wrapper_percentage", 10)
    
    # Phase 3: 50% rollout
    print("\n=== Phase 3: 50% Rollout ===")
    flag_manager.update_flag("example_wrapper_percentage", 50)
    
    # Phase 4: Full rollout
    print("\n=== Phase 4: Full Rollout ===") 
    flag_manager.update_flag("example_wrapper_percentage", 100)
    
    print("\nRollout complete!")

if __name__ == "__main__":
    asyncio.run(gradual_rollout_example())
```

### Step 4: Implement Monitoring

Add comprehensive monitoring to your wrapper:

```python
# wrapper_monitoring_setup.py
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring, AlertRule, AlertCondition
import asyncio

def setup_wrapper_monitoring():
    """Set up monitoring for wrapper integrations."""
    
    monitoring = WrapperMonitoring()
    
    # Define alert rules
    alert_rules = [
        AlertRule(
            name="high_error_rate",
            description="Alert when wrapper error rate exceeds threshold",
            condition=AlertCondition.ERROR_RATE_THRESHOLD,
            threshold=0.05,  # 5% error rate
            window_minutes=5,
            severity="high"
        ),
        AlertRule(
            name="slow_response_time",
            description="Alert when response time is too slow",
            condition=AlertCondition.RESPONSE_TIME_THRESHOLD,
            threshold=5000,  # 5 seconds
            window_minutes=10,
            severity="medium"
        ),
        AlertRule(
            name="high_fallback_usage",
            description="Alert when fallback usage is too high",
            condition=AlertCondition.FALLBACK_RATE_THRESHOLD,
            threshold=0.10,  # 10% fallback rate
            window_minutes=15,
            severity="medium"
        )
    ]
    
    # Register alert rules
    for rule in alert_rules:
        monitoring.add_alert_rule(rule)
    
    print("Monitoring configured with alert rules:")
    for rule in alert_rules:
        print(f"  - {rule.name}: {rule.description}")
    
    return monitoring

async def monitor_wrapper_health():
    """Monitor wrapper health and generate reports."""
    
    monitoring = setup_wrapper_monitoring()
    
    # Get health status for all wrappers
    health_status = monitoring.get_system_health()
    
    print("System Health Report")
    print("=" * 30)
    print(f"Overall health: {health_status.overall_status}")
    print(f"Active wrappers: {len(health_status.wrapper_healths)}")
    
    for wrapper_name, health in health_status.wrapper_healths.items():
        print(f"\n{wrapper_name}:")
        print(f"  Status: {health.status}")
        print(f"  Success rate: {health.success_rate:.2%}")
        print(f"  Avg response time: {health.avg_response_time_ms:.1f}ms")
        print(f"  Error rate: {health.error_rate:.2%}")
        print(f"  Fallback rate: {health.fallback_rate:.2%}")
    
    # Check for active alerts
    active_alerts = monitoring.get_active_alerts()
    if active_alerts:
        print("\nActive Alerts:")
        for alert in active_alerts:
            print(f"  ⚠️  {alert.rule_name}: {alert.message}")
    else:
        print("\n✓ No active alerts")

if __name__ == "__main__":
    asyncio.run(monitor_wrapper_health())
```

### Step 5: Configuration Management

Implement comprehensive configuration management:

```python
# config_management.py
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField, ConfigManager
from src.orchestrator.core.feature_flags import FeatureFlagManager
from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass  
class SystemWrapperConfig(BaseWrapperConfig):
    """System-wide wrapper configuration."""
    
    # Global wrapper settings
    enable_wrapper_framework: bool = True
    default_timeout_seconds: float = 30.0
    max_retry_attempts: int = 3
    enable_monitoring: bool = True
    enable_feature_flags: bool = True
    
    # Performance settings
    connection_pool_size: int = 100
    max_concurrent_operations: int = 50
    cache_ttl_seconds: int = 300
    
    # Security settings
    validate_ssl_certificates: bool = True
    mask_sensitive_data_in_logs: bool = True
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Define system configuration fields."""
        return {
            "enable_wrapper_framework": ConfigField(
                name="enable_wrapper_framework",
                field_type=bool,
                default_value=self.enable_wrapper_framework,
                description="Enable wrapper framework globally",
                environment_var="WRAPPER_FRAMEWORK_ENABLED"
            ),
            "default_timeout_seconds": ConfigField(
                name="default_timeout_seconds", 
                field_type=float,
                default_value=self.default_timeout_seconds,
                description="Default timeout for wrapper operations",
                min_value=1.0,
                max_value=300.0,
                environment_var="WRAPPER_DEFAULT_TIMEOUT"
            ),
            "max_retry_attempts": ConfigField(
                name="max_retry_attempts",
                field_type=int, 
                default_value=self.max_retry_attempts,
                description="Maximum retry attempts for failed operations",
                min_value=0,
                max_value=10,
                environment_var="WRAPPER_MAX_RETRIES"
            )
        }

def setup_configuration_management():
    """Set up comprehensive configuration management."""
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load system configuration
    system_config = SystemWrapperConfig()
    
    # Override with environment variables
    env_overrides = {
        "enable_wrapper_framework": os.getenv("WRAPPER_FRAMEWORK_ENABLED", "true").lower() == "true",
        "default_timeout_seconds": float(os.getenv("WRAPPER_DEFAULT_TIMEOUT", "30.0")),
        "max_retry_attempts": int(os.getenv("WRAPPER_MAX_RETRIES", "3"))
    }
    
    # Apply overrides
    for key, value in env_overrides.items():
        if hasattr(system_config, key):
            setattr(system_config, key, value)
    
    # Validate configuration
    try:
        config_manager.validate_config(system_config)
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return None
    
    print("Configuration Management Setup:")
    print(f"  Wrapper framework: {'Enabled' if system_config.enable_wrapper_framework else 'Disabled'}")
    print(f"  Default timeout: {system_config.default_timeout_seconds}s")
    print(f"  Max retries: {system_config.max_retry_attempts}")
    print(f"  Monitoring: {'Enabled' if system_config.enable_monitoring else 'Disabled'}")
    
    return config_manager, system_config

# Environment configuration template
def create_environment_config_template():
    """Create environment configuration template."""
    
    env_template = """
# Wrapper Framework Configuration
WRAPPER_FRAMEWORK_ENABLED=true
WRAPPER_DEFAULT_TIMEOUT=30.0
WRAPPER_MAX_RETRIES=3

# Feature Flags
FEATURE_FLAGS_ENABLED=true
FEATURE_FLAGS_CACHE_TTL=300

# Monitoring
WRAPPER_MONITORING_ENABLED=true
WRAPPER_METRICS_RETENTION_DAYS=30

# Security
WRAPPER_VALIDATE_SSL=true
WRAPPER_MASK_SENSITIVE_DATA=true

# Performance  
WRAPPER_CONNECTION_POOL_SIZE=100
WRAPPER_MAX_CONCURRENT_OPS=50
WRAPPER_CACHE_TTL=300

# Specific wrapper configurations
EXAMPLE_API_KEY=your_api_key_here
EXAMPLE_API_ENDPOINT=https://api.example.com
"""
    
    with open(".env.wrapper_template", "w") as f:
        f.write(env_template.strip())
    
    print("Environment configuration template created: .env.wrapper_template")

if __name__ == "__main__":
    config_manager, system_config = setup_configuration_management()
    create_environment_config_template()
```

### Step 6: Testing and Validation

Implement comprehensive testing for wrapper integrations:

```python
# wrapper_testing.py
import asyncio
import time
from typing import List, Dict, Any
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult
from src.orchestrator.core.wrapper_testing import WrapperTestHarness, TestScenario

async def comprehensive_wrapper_testing():
    """Comprehensive testing of wrapper integrations."""
    
    # Initialize test harness
    test_harness = WrapperTestHarness()
    
    # Define test scenarios
    test_scenarios = [
        TestScenario(
            name="successful_operation",
            description="Test successful wrapper operation",
            setup_data={"query": "test query"},
            expected_success=True,
            timeout_seconds=10.0
        ),
        TestScenario(
            name="timeout_handling", 
            description="Test timeout handling",
            setup_data={"query": "slow query", "delay": 5.0},
            expected_success=False,
            timeout_seconds=2.0,
            expected_fallback=True
        ),
        TestScenario(
            name="error_recovery",
            description="Test error recovery and fallback",
            setup_data={"query": "error query"},
            expected_success=False,
            expected_fallback=True
        ),
        TestScenario(
            name="configuration_validation",
            description="Test configuration validation",
            setup_data={"invalid_config": True},
            expected_success=False,
            validate_config=True
        )
    ]
    
    print("Running Comprehensive Wrapper Tests")
    print("=" * 40)
    
    test_results = []
    
    for scenario in test_scenarios:
        print(f"\nRunning: {scenario.name}")
        print(f"Description: {scenario.description}")
        
        start_time = time.time()
        
        try:
            # This would integrate with your actual wrapper implementation
            # For demo purposes, we'll simulate test results
            result = await simulate_test_scenario(scenario)
            duration = time.time() - start_time
            
            test_results.append({
                "scenario": scenario.name,
                "success": result.success == scenario.expected_success,
                "duration_ms": duration * 1000,
                "fallback_used": result.fallback_used,
                "error": result.error
            })
            
            status = "✓ PASS" if result.success == scenario.expected_success else "❌ FAIL"
            print(f"Result: {status} ({duration*1000:.1f}ms)")
            
        except Exception as e:
            test_results.append({
                "scenario": scenario.name,
                "success": False,
                "duration_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            })
            print(f"Result: ❌ ERROR - {e}")
    
    # Generate test report
    generate_test_report(test_results)

async def simulate_test_scenario(scenario: TestScenario) -> WrapperResult:
    """Simulate test scenario execution."""
    
    # Simulate different scenarios
    if scenario.name == "successful_operation":
        return WrapperResult(
            success=True,
            data={"result": "success", "query": scenario.setup_data.get("query")},
            execution_time_ms=50.0
        )
    elif scenario.name == "timeout_handling":
        return WrapperResult(
            success=False,
            error="Timeout exceeded",
            fallback_used=True,
            fallback_reason="timeout",
            execution_time_ms=2000.0
        )
    elif scenario.name == "error_recovery":
        return WrapperResult(
            success=False,
            error="External service error",
            fallback_used=True,
            fallback_reason="external_error",
            data={"fallback_response": True}
        )
    else:
        return WrapperResult(
            success=False,
            error="Configuration invalid"
        )

def generate_test_report(test_results: List[Dict[str, Any]]):
    """Generate comprehensive test report."""
    
    print("\n" + "="*50)
    print("WRAPPER TESTING REPORT") 
    print("="*50)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result["success"])
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests:.1%}")
    
    avg_duration = sum(result["duration_ms"] for result in test_results) / total_tests
    print(f"Average duration: {avg_duration:.1f}ms")
    
    print("\nDetailed Results:")
    for result in test_results:
        status = "✓" if result["success"] else "❌"
        print(f"  {status} {result['scenario']} ({result['duration_ms']:.1f}ms)")
        if not result["success"] and "error" in result:
            print(f"    Error: {result['error']}")
    
    # Performance analysis
    slow_tests = [r for r in test_results if r["duration_ms"] > 1000]
    if slow_tests:
        print(f"\n⚠️  Slow tests (>1s): {len(slow_tests)}")
        for test in slow_tests:
            print(f"    - {test['scenario']}: {test['duration_ms']:.1f}ms")
    
    print(f"\nOverall: {'✅ PASSED' if failed_tests == 0 else '❌ FAILED'}")

if __name__ == "__main__":
    asyncio.run(comprehensive_wrapper_testing())
```

## Migration Validation

### Validation Checklist

After migration, verify these items:

- [ ] **Framework Installation**: Wrapper framework components are available
- [ ] **Configuration**: Configuration system works with validation
- [ ] **Feature Flags**: Feature flag system operates correctly
- [ ] **Monitoring**: Monitoring and metrics collection function
- [ ] **Error Handling**: Fallback mechanisms work as expected
- [ ] **Performance**: Acceptable performance overhead (<5ms typical)
- [ ] **Integration**: Wrapper integrates with existing systems
- [ ] **Testing**: Comprehensive test suite passes

### Comprehensive Validation Script

```python
# migration_validation.py
import asyncio
import time
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult
from src.orchestrator.core.wrapper_config import BaseWrapperConfig
from src.orchestrator.core.feature_flags import FeatureFlagManager
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

async def comprehensive_migration_validation():
    """Comprehensive validation of wrapper architecture migration."""
    
    print("Wrapper Architecture Migration Validation")
    print("=" * 50)
    
    validation_results = {
        "passed": [],
        "failed": [],
        "warnings": []
    }
    
    # Test 1: Core framework components
    try:
        # Test base wrapper class
        assert BaseWrapper is not None
        assert WrapperResult is not None
        assert BaseWrapperConfig is not None
        validation_results["passed"].append("Core framework components available")
    except Exception as e:
        validation_results["failed"].append(f"Core components: {e}")
    
    # Test 2: Feature flag system
    try:
        flag_manager = FeatureFlagManager()
        
        # Test flag operations
        test_flag = "test_migration_flag"
        flag_manager.create_flag(test_flag, default_value=False)
        flag_manager.update_flag(test_flag, True)
        
        assert flag_manager.is_enabled(test_flag) == True
        validation_results["passed"].append("Feature flag system functional")
    except Exception as e:
        validation_results["failed"].append(f"Feature flags: {e}")
    
    # Test 3: Monitoring system
    try:
        monitoring = WrapperMonitoring()
        
        # Test basic monitoring operations
        operation_id = monitoring.start_operation("test_wrapper", "test_operation")
        monitoring.record_success(operation_id, {"test": "data"})
        monitoring.end_operation(operation_id)
        
        # Check health status
        health = monitoring.get_wrapper_health("test_wrapper")
        assert health is not None
        
        validation_results["passed"].append("Monitoring system functional")
    except Exception as e:
        validation_results["failed"].append(f"Monitoring system: {e}")
    
    # Test 4: Configuration system
    try:
        from dataclasses import dataclass
        
        @dataclass
        class TestConfig(BaseWrapperConfig):
            test_param: str = "default"
            
            def get_config_fields(self):
                return {
                    "test_param": ConfigField("test_param", str, self.test_param)
                }
        
        config = TestConfig(test_param="test_value")
        fields = config.get_config_fields()
        
        assert "test_param" in fields
        assert fields["test_param"].default_value == "test_value"
        
        validation_results["passed"].append("Configuration system functional")
    except Exception as e:
        validation_results["failed"].append(f"Configuration system: {e}")
    
    # Test 5: Performance benchmark
    try:
        # Simple performance test
        start_time = time.time()
        
        # Simulate wrapper operations
        for i in range(100):
            flag_manager = FeatureFlagManager()
            flag_manager.create_flag(f"perf_test_{i}", default_value=True)
        
        duration = (time.time() - start_time) * 1000  # ms
        
        if duration < 1000:  # Less than 1 second for 100 operations
            validation_results["passed"].append(f"Performance benchmark: {duration:.1f}ms for 100 ops")
        else:
            validation_results["warnings"].append(f"Performance slower than expected: {duration:.1f}ms")
    except Exception as e:
        validation_results["warnings"].append(f"Performance benchmark: {e}")
    
    # Test 6: Integration compatibility
    try:
        # Test async compatibility
        async def test_async_operation():
            return "async_test_passed"
        
        result = await test_async_operation()
        assert result == "async_test_passed"
        
        validation_results["passed"].append("Async integration compatible")
    except Exception as e:
        validation_results["failed"].append(f"Async integration: {e}")
    
    # Display results
    print("\nValidation Results:")
    print("-" * 30)
    
    if validation_results["passed"]:
        print("✅ PASSED:")
        for test in validation_results["passed"]:
            print(f"  ✓ {test}")
    
    if validation_results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results["warnings"]:
            print(f"  ⚠️  {warning}")
    
    if validation_results["failed"]:
        print("\n❌ FAILED:")
        for failure in validation_results["failed"]:
            print(f"  ❌ {failure}")
    
    # Overall assessment
    total_tests = len(validation_results["passed"]) + len(validation_results["warnings"]) + len(validation_results["failed"])
    success_rate = len(validation_results["passed"]) / total_tests if total_tests > 0 else 0
    
    print(f"\nOverall Success Rate: {success_rate:.1%}")
    print(f"Migration Status: {'✅ SUCCESS' if len(validation_results['failed']) == 0 else '❌ FAILED'}")
    
    return validation_results

if __name__ == "__main__":
    asyncio.run(comprehensive_migration_validation())
```

## Best Practices

### Wrapper Design Principles

1. **Single Responsibility**: Each wrapper should handle one external integration
2. **Type Safety**: Use generic types for compile-time checking
3. **Async First**: Design for async/await patterns from the start
4. **Error Handling**: Always implement comprehensive fallback mechanisms
5. **Configuration**: Make all behavior configurable through the config system

### Migration Strategy Best Practices

1. **Start Small**: Begin with simple, low-risk integrations
2. **Feature Flags**: Use feature flags for all rollouts
3. **Monitor Everything**: Enable comprehensive monitoring from day one
4. **Test Thoroughly**: Implement comprehensive test suites
5. **Document Well**: Keep migration and configuration documentation updated

### Operational Best Practices

1. **Gradual Rollout**: Use percentage-based rollouts for safety
2. **Health Monitoring**: Monitor wrapper health and performance continuously
3. **Alert Management**: Set up appropriate alerts for failures and performance issues
4. **Configuration Management**: Use environment-specific configurations
5. **Regular Review**: Regularly review and optimize wrapper performance

## Troubleshooting

### Common Issues

#### Issue: Import errors with wrapper components
```python
# Check if wrapper framework is properly installed
try:
    from src.orchestrator.core.wrapper_base import BaseWrapper
    print("✓ Wrapper framework available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Solution: Ensure orchestrator-framework is installed with wrapper support")
```

#### Issue: Configuration validation failures
```python
# Debug configuration issues
from src.orchestrator.core.wrapper_config import ConfigManager

config_manager = ConfigManager()
try:
    validation_result = config_manager.validate_config(your_config)
    print("✓ Configuration valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    # Check required fields, types, and constraints
```

#### Issue: Feature flags not working
```python
# Debug feature flag issues
flag_manager = FeatureFlagManager()

# Check flag registration
flags = flag_manager.get_all_flags()
print(f"Registered flags: {list(flags.keys())}")

# Check flag evaluation
flag_name = "your_flag_name"
is_enabled = flag_manager.is_enabled(flag_name)
print(f"Flag '{flag_name}' enabled: {is_enabled}")
```

#### Issue: Poor wrapper performance
```python
# Profile wrapper performance
import time

start_time = time.time()
result = await your_wrapper.execute(operation_type="test")
duration = (time.time() - start_time) * 1000

print(f"Wrapper execution time: {duration:.2f}ms")

if duration > 100:  # More than 100ms
    print("⚠️  Wrapper performance may be slow")
    print("Consider: connection pooling, caching, async optimization")
```

### Support Resources

- **Framework Documentation**: Complete API reference in `docs/wrapper_development/`
- **Examples**: Working wrapper examples in `examples/wrappers/`
- **Testing**: Comprehensive testing patterns and utilities
- **Monitoring**: Built-in monitoring and alerting capabilities

## Expected Benefits

After successful migration, you should see:

- **Consistency**: Standardized patterns across all external integrations
- **Reliability**: Comprehensive error handling and fallback mechanisms
- **Observability**: Built-in monitoring, metrics, and health checking
- **Flexibility**: Feature flag control and configuration management
- **Performance**: Optimized async patterns and resource management
- **Maintainability**: Clear interfaces and testing frameworks

## Next Steps

1. **Optimize Configurations**: Fine-tune wrapper configurations for your use cases
2. **Implement Custom Wrappers**: Build wrappers for your specific integrations
3. **Enhance Monitoring**: Add custom alerts and integrate with your monitoring systems
4. **Advanced Features**: Explore advanced wrapper capabilities and patterns
5. **Community Contribution**: Share successful wrapper patterns with the community

This wrapper architecture migration guide provides a comprehensive foundation for adopting the unified wrapper framework. The incremental approach ensures safe migration while providing immediate benefits for improved integration patterns and operational excellence.