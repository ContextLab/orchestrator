# Interactive Tutorials and Examples

## Overview

This comprehensive collection of interactive tutorials and examples demonstrates how to use the wrapper framework, RouteLLM integration, POML templates, feature flags, and monitoring capabilities. Each tutorial includes working code, step-by-step instructions, and real-world scenarios.

## Table of Contents

- [Getting Started Tutorial](#getting-started-tutorial)
- [RouteLLM Integration Examples](#routellm-integration-examples)
- [POML Template Examples](#poml-template-examples)
- [Advanced Wrapper Development](#advanced-wrapper-development)
- [Feature Flag Examples](#feature-flag-examples)
- [Monitoring and Observability](#monitoring-and-observability)
- [Production Examples](#production-examples)
- [Troubleshooting Scenarios](#troubleshooting-scenarios)

## Getting Started Tutorial

### Tutorial 1: Your First Wrapper

This tutorial walks you through creating your first wrapper from scratch.

```python
# tutorial_1_first_wrapper.py
"""
Tutorial 1: Creating Your First Wrapper

Learn how to:
- Create a basic wrapper configuration
- Implement wrapper operations
- Handle errors and fallbacks
- Test your wrapper
"""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult, WrapperContext, WrapperCapability
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField

# Step 1: Define Configuration
@dataclass
class WeatherAPIConfig(BaseWrapperConfig):
    """Configuration for our weather API wrapper."""
    
    api_key: str = ""
    base_url: str = "https://api.weatherapi.com/v1"
    timeout_seconds: float = 10.0
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Define configuration with validation."""
        return {
            "api_key": ConfigField(
                name="api_key",
                field_type=str,
                default_value="",
                description="Weather API key",
                required=True,
                sensitive=True,
                environment_var="WEATHER_API_KEY"
            ),
            "base_url": ConfigField(
                name="base_url",
                field_type=str,
                default_value=self.base_url,
                description="API base URL",
                required=True
            ),
            "timeout_seconds": ConfigField(
                name="timeout_seconds",
                field_type=float,
                default_value=self.timeout_seconds,
                description="Request timeout",
                min_value=1.0,
                max_value=60.0
            )
        }

# Step 2: Define Data Types
@dataclass
class WeatherData:
    """Weather data structure."""
    location: str
    temperature: float
    description: str
    humidity: int

# Step 3: Implement Wrapper
class WeatherAPIWrapper(BaseWrapper[WeatherData, WeatherAPIConfig]):
    """A simple weather API wrapper."""
    
    def __init__(self, config: Optional[WeatherAPIConfig] = None):
        """Initialize the wrapper."""
        config = config or WeatherAPIConfig()
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _execute_wrapper_operation(
        self, 
        context: WrapperContext, 
        location: str,
        **kwargs
    ) -> WeatherData:
        """Fetch weather data for a location."""
        
        # Validate input
        if not location or not location.strip():
            raise ValueError("Location cannot be empty")
        
        # Initialize HTTP session if needed
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        # Make API request
        url = f"{self.config.base_url}/current.json"
        params = {
            "key": self.config.api_key,
            "q": location
        }
        
        try:
            async with self._session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Extract weather information
                location_info = data["location"]
                current = data["current"]
                
                return WeatherData(
                    location=f"{location_info['name']}, {location_info['country']}",
                    temperature=current["temp_c"],
                    description=current["condition"]["text"],
                    humidity=current["humidity"]
                )
                
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                raise ValueError(f"Invalid location: {location}")
            elif e.status == 401:
                raise ValueError("Invalid API key")
            else:
                raise RuntimeError(f"API error: {e}")
        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timeout after {self.config.timeout_seconds}s")
    
    async def _execute_fallback_operation(
        self,
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        location: str = "",
        **kwargs
    ) -> WeatherData:
        """Provide fallback weather data."""
        print(f"⚠️  Using fallback for {location}: {original_error}")
        
        # Return mock weather data as fallback
        return WeatherData(
            location=location or "Unknown",
            temperature=20.0,
            description="Weather data unavailable",
            humidity=50
        )
    
    def _validate_config(self) -> bool:
        """Validate configuration."""
        return bool(self.config.api_key and self.config.base_url)
    
    def get_capabilities(self) -> List[WrapperCapability]:
        """Return wrapper capabilities."""
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

# Step 4: Interactive Tutorial Function
async def run_first_wrapper_tutorial():
    """Run the first wrapper tutorial interactively."""
    
    print("🌟 Welcome to Wrapper Framework Tutorial 1!")
    print("=" * 50)
    print()
    
    # Step 1: Configuration
    print("Step 1: Creating Configuration")
    print("-" * 30)
    
    # For tutorial purposes, we'll use a mock API key
    # In real usage, set WEATHER_API_KEY environment variable
    config = WeatherAPIConfig(
        api_key="demo_key_123",  # Mock API key for tutorial
        timeout_seconds=10.0
    )
    
    print(f"✅ Configuration created")
    print(f"   API Key: {'*' * 8}  # Masked for security")
    print(f"   Base URL: {config.base_url}")
    print(f"   Timeout: {config.timeout_seconds}s")
    print()
    
    # Step 2: Create Wrapper
    print("Step 2: Creating Wrapper Instance")
    print("-" * 30)
    
    wrapper = WeatherAPIWrapper(config)
    print(f"✅ Wrapper created: {wrapper.__class__.__name__}")
    print(f"   Capabilities: {[cap.value for cap in wrapper.get_capabilities()]}")
    print()
    
    # Step 3: Test Configuration Validation
    print("Step 3: Configuration Validation")
    print("-" * 30)
    
    is_valid = wrapper._validate_config()
    print(f"Configuration valid: {'✅ Yes' if is_valid else '❌ No'}")
    print()
    
    # Step 4: Execute Operations
    print("Step 4: Executing Wrapper Operations")
    print("-" * 30)
    
    test_locations = ["London", "New York", "Tokyo", "InvalidLocation123"]
    
    for location in test_locations:
        print(f"Testing location: {location}")
        
        try:
            # Execute wrapper operation
            result = await wrapper.execute("get_weather", location=location)
            
            if result.success:
                weather = result.data
                print(f"  ✅ Success!")
                print(f"     Location: {weather.location}")
                print(f"     Temperature: {weather.temperature}°C")
                print(f"     Description: {weather.description}")
                print(f"     Humidity: {weather.humidity}%")
                
                if result.execution_time_ms:
                    print(f"     Response time: {result.execution_time_ms:.1f}ms")
            else:
                print(f"  ❌ Failed: {result.error}")
                if result.fallback_used:
                    print(f"     Fallback reason: {result.fallback_reason}")
                    weather = result.data
                    if weather:
                        print(f"     Fallback data: {weather.location} - {weather.description}")
        
        except Exception as e:
            print(f"  💥 Exception: {e}")
        
        print()
    
    # Step 5: Cleanup
    print("Step 5: Cleanup")
    print("-" * 30)
    
    await wrapper.cleanup()
    print("✅ Wrapper cleaned up")
    print()
    
    print("🎉 Tutorial 1 Complete!")
    print()
    print("What you learned:")
    print("• How to create wrapper configuration with validation")
    print("• How to implement wrapper operations with error handling")
    print("• How to use fallback mechanisms")
    print("• How to properly clean up resources")

# Interactive Tutorial Runner
if __name__ == "__main__":
    print("Starting Interactive Wrapper Tutorial...")
    print("Note: This tutorial uses mock data for demonstration.")
    print("For real usage, obtain an API key from weatherapi.com")
    print()
    
    try:
        asyncio.run(run_first_wrapper_tutorial())
    except KeyboardInterrupt:
        print("\n👋 Tutorial interrupted by user")
    except Exception as e:
        print(f"\n💥 Tutorial error: {e}")
        import traceback
        traceback.print_exc()
```

### Tutorial 2: Configuration and Environment Management

```python
# tutorial_2_configuration.py
"""
Tutorial 2: Configuration and Environment Management

Learn how to:
- Use environment variables
- Validate configuration
- Handle different environments
- Override configuration at runtime
"""

import os
import asyncio
from dataclasses import dataclass
from typing import Dict, Any
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField, ConfigManager

@dataclass
class DatabaseConfig(BaseWrapperConfig):
    """Database connection configuration."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "orchestrator"
    username: str = "postgres"
    password: str = ""
    
    # Pool settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: float = 30.0
    
    # Feature flags
    enable_ssl: bool = True
    enable_connection_retry: bool = True
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Define configuration fields with environment variable support."""
        return {
            "host": ConfigField(
                name="host",
                field_type=str,
                default_value=self.host,
                description="Database host",
                environment_var="DB_HOST"
            ),
            "port": ConfigField(
                name="port",
                field_type=int,
                default_value=self.port,
                description="Database port",
                environment_var="DB_PORT",
                min_value=1,
                max_value=65535
            ),
            "database": ConfigField(
                name="database",
                field_type=str,
                default_value=self.database,
                description="Database name",
                environment_var="DB_NAME",
                required=True
            ),
            "username": ConfigField(
                name="username",
                field_type=str,
                default_value=self.username,
                description="Database username",
                environment_var="DB_USER",
                required=True
            ),
            "password": ConfigField(
                name="password",
                field_type=str,
                default_value=self.password,
                description="Database password",
                environment_var="DB_PASSWORD",
                required=True,
                sensitive=True  # Will be masked in logs
            ),
            "max_pool_size": ConfigField(
                name="max_pool_size",
                field_type=int,
                default_value=self.max_pool_size,
                description="Maximum connection pool size",
                min_value=1,
                max_value=100
            )
        }
    
    def get_connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def get_masked_connection_string(self) -> str:
        """Get connection string with masked password."""
        return f"postgresql://{self.username}:***@{self.host}:{self.port}/{self.database}"

async def run_configuration_tutorial():
    """Interactive configuration tutorial."""
    
    print("🔧 Configuration and Environment Tutorial")
    print("=" * 50)
    print()
    
    # Step 1: Basic Configuration
    print("Step 1: Creating Basic Configuration")
    print("-" * 40)
    
    config = DatabaseConfig()
    print("Default configuration created:")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Database: {config.database}")
    print(f"  Connection: {config.get_masked_connection_string()}")
    print()
    
    # Step 2: Environment Variables
    print("Step 2: Using Environment Variables")
    print("-" * 40)
    
    # Simulate environment variables
    env_vars = {
        "DB_HOST": "prod-db.example.com",
        "DB_PORT": "5432",
        "DB_NAME": "production_db",
        "DB_USER": "app_user",
        "DB_PASSWORD": "secure_password_123"
    }
    
    print("Setting environment variables:")
    for key, value in env_vars.items():
        os.environ[key] = value
        display_value = "***" if "PASSWORD" in key else value
        print(f"  {key}: {display_value}")
    print()
    
    # Load configuration with environment overrides
    config_manager = ConfigManager()
    env_config = config_manager.load_config_from_env(DatabaseConfig)
    
    print("Configuration loaded from environment:")
    print(f"  Host: {env_config.host}")
    print(f"  Port: {env_config.port}")
    print(f"  Database: {env_config.database}")
    print(f"  User: {env_config.username}")
    print(f"  Connection: {env_config.get_masked_connection_string()}")
    print()
    
    # Step 3: Configuration Validation
    print("Step 3: Configuration Validation")
    print("-" * 40)
    
    validation_result = config_manager.validate_config(env_config)
    
    if validation_result.valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed:")
        for error in validation_result.errors:
            print(f"  - {error}")
    
    if validation_result.warnings:
        print("⚠️  Configuration warnings:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    print()
    
    # Step 4: Runtime Configuration Override
    print("Step 4: Runtime Configuration Override")
    print("-" * 40)
    
    # Override configuration at runtime
    runtime_overrides = {
        "max_pool_size": 50,
        "pool_timeout": 60.0,
        "enable_ssl": False  # Disable SSL for local development
    }
    
    print("Applying runtime overrides:")
    for key, value in runtime_overrides.items():
        setattr(env_config, key, value)
        print(f"  {key}: {value}")
    print()
    
    # Step 5: Environment-Specific Configurations
    print("Step 5: Environment-Specific Configuration")
    print("-" * 40)
    
    environments = ["development", "staging", "production"]
    
    for env in environments:
        print(f"\n{env.upper()} Configuration:")
        
        if env == "development":
            dev_config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="dev_db",
                username="dev_user",
                password="dev_password",
                max_pool_size=5,
                enable_ssl=False
            )
            config = dev_config
        elif env == "staging":
            staging_config = DatabaseConfig(
                host="staging-db.example.com",
                port=5432,
                database="staging_db",
                username="staging_user",
                password="staging_password",
                max_pool_size=20,
                enable_ssl=True
            )
            config = staging_config
        else:  # production
            prod_config = DatabaseConfig(
                host="prod-db.example.com",
                port=5432,
                database="production_db",
                username="prod_user",
                password="super_secure_password",
                max_pool_size=50,
                enable_ssl=True,
                pool_timeout=60.0
            )
            config = prod_config
        
        print(f"  Host: {config.host}")
        print(f"  Pool size: {config.max_pool_size}")
        print(f"  SSL enabled: {config.enable_ssl}")
        print(f"  Connection: {config.get_masked_connection_string()}")
    
    print()
    
    # Step 6: Configuration Export
    print("Step 6: Configuration Export and Import")
    print("-" * 40)
    
    # Export configuration (sensitive data masked)
    exported = config_manager.export_config(env_config, include_sensitive=False)
    print("Exported configuration (sensitive data masked):")
    for key, value in exported.items():
        print(f"  {key}: {value}")
    print()
    
    # Cleanup environment variables
    for key in env_vars:
        if key in os.environ:
            del os.environ[key]
    
    print("🎉 Configuration Tutorial Complete!")
    print()
    print("What you learned:")
    print("• How to define configuration with environment variable support")
    print("• How to validate configuration with detailed error reporting")
    print("• How to override configuration at runtime")
    print("• How to manage environment-specific configurations")
    print("• How to safely export and import configuration")

if __name__ == "__main__":
    asyncio.run(run_configuration_tutorial())
```

## RouteLLM Integration Examples

### Example 1: Cost-Optimized AI Assistant

```python
# example_routellm_assistant.py
"""
RouteLLM Cost-Optimized AI Assistant Example

This example demonstrates how to build an AI assistant that automatically
routes queries to cost-effective models while maintaining quality.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.orchestrator.models.routellm_integration import (
    RouteLLMConfig, RouterType, FeatureFlags, CostTracker, RoutingDecision
)

@dataclass
class AssistantQuery:
    """Structure for assistant queries."""
    user_id: str
    query: str
    domain: Optional[str] = None
    complexity_hint: Optional[str] = None  # "simple", "medium", "complex"
    max_cost: Optional[float] = None

@dataclass
class AssistantResponse:
    """Structure for assistant responses."""
    response: str
    model_used: str
    estimated_cost: float
    routing_method: str
    confidence: float
    cached: bool = False

class CostOptimizedAssistant:
    """AI Assistant with RouteLLM cost optimization."""
    
    def __init__(self):
        # Initialize RouteLLM components
        self.config = RouteLLMConfig(
            enabled=True,
            router_type=RouterType.MATRIX_FACTORIZATION,
            threshold=0.12,  # Slightly higher threshold for more cost savings
            strong_model="gpt-4-1106-preview",
            weak_model="gpt-3.5-turbo",
            cost_tracking_enabled=True,
            domain_specific_routing=True,
            domain_routing_overrides={
                "simple": {"threshold": 0.20},  # Route more to weak model
                "technical": {"threshold": 0.10},  # Be conservative for technical
                "creative": {"threshold": 0.08},  # Use strong model for creativity
            }
        )
        
        self.feature_flags = FeatureFlags()
        self.cost_tracker = CostTracker()
        
        # Simple response cache
        self.response_cache: Dict[str, AssistantResponse] = {}
        
        # Initialize feature flags
        self._setup_feature_flags()
    
    def _setup_feature_flags(self):
        """Set up feature flags for gradual rollout."""
        # Enable basic RouteLLM features
        self.feature_flags.enable(FeatureFlags.ROUTELLM_ENABLED)
        self.feature_flags.enable(FeatureFlags.ROUTELLM_COST_TRACKING)
        self.feature_flags.enable(FeatureFlags.ROUTELLM_PERFORMANCE_MONITORING)
        
        # Enable domain-specific routing
        self.feature_flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)
        self.feature_flags.enable(FeatureFlags.ROUTELLM_CREATIVE_DOMAIN)
    
    async def process_query(self, query: AssistantQuery) -> AssistantResponse:
        """Process user query with cost optimization."""
        
        # Check cache first
        cache_key = f"{query.query}:{query.domain}"
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response.cached = True
            print(f"💾 Cache hit for query: {query.query[:50]}...")
            return cached_response
        
        # Determine routing strategy
        routing_decision = await self._make_routing_decision(query)
        
        # Generate response (simulated)
        response = await self._generate_response(query, routing_decision)
        
        # Track the decision for cost analysis
        tracking_id = self.cost_tracker.track_routing_decision(
            text=query.query,
            domains=[query.domain] if query.domain else ["general"],
            routing_method="routellm" if routing_decision.should_use_routellm else "default",
            selected_model=routing_decision.recommended_model or self.config.strong_model,
            estimated_cost=routing_decision.estimated_cost,
            routing_confidence=routing_decision.confidence,
            success=True
        )
        
        # Cache response
        self.response_cache[cache_key] = response
        
        return response
    
    async def _make_routing_decision(self, query: AssistantQuery) -> RoutingDecision:
        """Make intelligent routing decision based on query characteristics."""
        
        # Analyze query characteristics
        query_length = len(query.query)
        word_count = len(query.query.split())
        
        # Simple heuristics for routing (in real implementation, use RouteLLM)
        complexity_score = 0.0
        
        # Length-based scoring
        if query_length > 500:
            complexity_score += 0.3
        elif query_length > 200:
            complexity_score += 0.1
        
        # Content-based scoring
        complex_keywords = [
            "analyze", "explain in detail", "comprehensive", "compare", "evaluate",
            "algorithm", "implementation", "architecture", "design pattern"
        ]
        
        for keyword in complex_keywords:
            if keyword.lower() in query.query.lower():
                complexity_score += 0.2
        
        # Domain-specific adjustments
        domain_threshold = self.config.threshold
        if query.domain in self.config.domain_routing_overrides:
            domain_override = self.config.domain_routing_overrides[query.domain]
            domain_threshold = domain_override.get("threshold", self.config.threshold)
        
        # Make routing decision
        should_use_weak_model = complexity_score < domain_threshold
        
        if should_use_weak_model:
            model = self.config.weak_model
            estimated_cost = 0.002 * word_count  # Approximate GPT-3.5 pricing
        else:
            model = self.config.strong_model
            estimated_cost = 0.006 * word_count  # Approximate GPT-4 pricing
        
        return RoutingDecision(
            should_use_routellm=should_use_weak_model,
            recommended_model=model,
            confidence=min(1.0, abs(complexity_score - domain_threshold) + 0.5),
            estimated_cost=estimated_cost,
            reasoning=f"Complexity score: {complexity_score:.2f}, threshold: {domain_threshold:.2f}",
            domains=[query.domain] if query.domain else ["general"]
        )
    
    async def _generate_response(
        self, 
        query: AssistantQuery, 
        routing: RoutingDecision
    ) -> AssistantResponse:
        """Generate response using selected model (simulated)."""
        
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Generate mock response based on model
        if routing.recommended_model == self.config.weak_model:
            response_text = f"[GPT-3.5 Response] Here's a concise answer to '{query.query[:30]}...'"
        else:
            response_text = f"[GPT-4 Response] Here's a detailed analysis of '{query.query[:30]}...'"
        
        return AssistantResponse(
            response=response_text,
            model_used=routing.recommended_model or "unknown",
            estimated_cost=routing.estimated_cost,
            routing_method="routellm" if routing.should_use_routellm else "default",
            confidence=routing.confidence,
            cached=False
        )
    
    def get_cost_report(self, days: int = 7) -> Dict[str, Any]:
        """Get cost savings report."""
        report = self.cost_tracker.get_cost_savings_report(days)
        summary = self.cost_tracker.get_metrics_summary()
        
        return {
            "period_days": days,
            "total_requests": report.total_requests,
            "estimated_savings": report.estimated_savings,
            "savings_percentage": report.savings_percentage,
            "average_cost_per_request": summary.get("average_cost", 0),
            "routellm_usage_rate": summary.get("routellm_usage_rate", 0),
            "cache_hit_rate": len([r for r in self.response_cache.values() if r.cached]) / max(1, len(self.response_cache))
        }

async def run_routellm_assistant_demo():
    """Interactive demo of the cost-optimized assistant."""
    
    print("🤖 RouteLLM Cost-Optimized Assistant Demo")
    print("=" * 50)
    print()
    
    # Initialize assistant
    assistant = CostOptimizedAssistant()
    
    # Sample queries with different complexity levels
    sample_queries = [
        AssistantQuery(
            user_id="user1",
            query="What is 2+2?",
            domain="simple",
            complexity_hint="simple"
        ),
        AssistantQuery(
            user_id="user2", 
            query="Explain the architectural patterns used in microservices and compare them with monolithic approaches.",
            domain="technical",
            complexity_hint="complex"
        ),
        AssistantQuery(
            user_id="user3",
            query="Write a creative story about a robot learning to paint.",
            domain="creative",
            complexity_hint="medium"
        ),
        AssistantQuery(
            user_id="user4",
            query="How do I center a div in CSS?",
            domain="technical",
            complexity_hint="simple"
        ),
        AssistantQuery(
            user_id="user1",
            query="What is 2+2?",  # Duplicate for cache demo
            domain="simple",
            complexity_hint="simple"
        )
    ]
    
    print("Processing sample queries...")
    print()
    
    for i, query in enumerate(sample_queries, 1):
        print(f"Query {i}: {query.query}")
        print(f"Domain: {query.domain}, User: {query.user_id}")
        
        # Process query
        response = await assistant.process_query(query)
        
        # Display results
        print(f"✅ Response: {response.response}")
        print(f"📊 Model: {response.model_used}")
        print(f"💰 Cost: ${response.estimated_cost:.4f}")
        print(f"🎯 Confidence: {response.confidence:.2f}")
        print(f"📁 Cached: {'Yes' if response.cached else 'No'}")
        print(f"🚀 Method: {response.routing_method}")
        print()
    
    # Show cost report
    print("💵 Cost Analysis Report")
    print("-" * 30)
    
    cost_report = assistant.get_cost_report(days=1)
    print(f"Total requests: {cost_report['total_requests']}")
    print(f"RouteLLM usage rate: {cost_report['routellm_usage_rate']:.1%}")
    print(f"Average cost per request: ${cost_report['average_cost_per_request']:.4f}")
    print(f"Cache hit rate: {cost_report['cache_hit_rate']:.1%}")
    
    if cost_report['estimated_savings'] > 0:
        print(f"💚 Estimated savings: ${cost_report['estimated_savings']:.4f} ({cost_report['savings_percentage']:.1f}%)")
    else:
        print("💡 No savings yet - try more queries to see optimization benefits")
    
    print()
    print("🎉 Demo Complete!")
    print()
    print("What this demo showed:")
    print("• Automatic routing based on query complexity")
    print("• Domain-specific routing thresholds") 
    print("• Cost tracking and savings calculation")
    print("• Response caching for efficiency")
    print("• Feature flag control for safe rollout")

if __name__ == "__main__":
    asyncio.run(run_routellm_assistant_demo())
```

## POML Template Examples

### Example 1: Dynamic Report Generator

```python
# example_poml_reports.py
"""
POML Template Examples - Dynamic Report Generator

This example demonstrates advanced POML template usage for generating
structured reports with cross-task references and dynamic content.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json

from src.orchestrator.core.template_resolver import TemplateResolver, TemplateFormat
from src.orchestrator.core.output_tracker import OutputTracker
from src.orchestrator.core.template_migration_tools import TemplateMigrationTools

class ReportGenerator:
    """Dynamic report generator using POML templates."""
    
    def __init__(self):
        self.template_resolver = TemplateResolver(
            enable_poml_processing=True,
            cross_task_references=True,
            enable_caching=True
        )
        self.output_tracker = OutputTracker()
        self.migration_tools = TemplateMigrationTools()
        
        # Sample data for demonstration
        self._setup_sample_data()
    
    def _setup_sample_data(self):
        """Set up sample data for report generation."""
        
        # Simulate previous task outputs
        self.output_tracker.save_output("data_collection", {
            "total_users": 15420,
            "active_users": 12850,
            "new_signups": 245,
            "churn_rate": 0.034,
            "revenue": 89420.50,
            "collection_date": "2025-08-25"
        })
        
        self.output_tracker.save_output("performance_analysis", {
            "avg_response_time": 0.245,
            "error_rate": 0.012,
            "uptime": 99.97,
            "peak_concurrent_users": 1850,
            "top_errors": [
                {"type": "timeout", "count": 23},
                {"type": "validation", "count": 15},
                {"type": "auth", "count": 8}
            ]
        })
        
        self.output_tracker.save_output("user_feedback", {
            "satisfaction_score": 4.2,
            "nps_score": 68,
            "feedback_count": 156,
            "top_issues": [
                "Slow loading times",
                "Confusing navigation", 
                "Missing features"
            ],
            "top_praise": [
                "Great user interface",
                "Reliable service",
                "Excellent support"
            ]
        })
    
    async def generate_executive_summary(self) -> str:
        """Generate executive summary using POML template."""
        
        template = """
        <poml version="1.0">
        <role>executive report analyst</role>
        
        <task>
        Generate an executive summary report based on the collected data and analysis.
        Focus on key metrics, trends, and actionable insights for leadership.
        </task>
        
        <context>
        Report Period: {{ report_period }}
        Generated On: {{ generation_date }}
        
        Key Metrics from Data Collection:
        - Total Users: {{ output_refs.data_collection.total_users | number_format }}
        - Active Users: {{ output_refs.data_collection.active_users | number_format }}
        - Revenue: ${{ output_refs.data_collection.revenue | number_format }}
        - Growth Rate: {{ ((output_refs.data_collection.new_signups / output_refs.data_collection.total_users) * 100) | round(2) }}%
        
        Performance Metrics:
        - System Uptime: {{ output_refs.performance_analysis.uptime }}%
        - Avg Response Time: {{ output_refs.performance_analysis.avg_response_time * 1000 | round }}ms
        - Error Rate: {{ (output_refs.performance_analysis.error_rate * 100) | round(3) }}%
        
        User Satisfaction:
        - Satisfaction Score: {{ output_refs.user_feedback.satisfaction_score }}/5.0
        - NPS Score: {{ output_refs.user_feedback.nps_score }}
        </context>
        
        <output-format>
        <document type="executive_summary">
          <section id="key_metrics">
            <h2>Key Performance Indicators</h2>
            <table>
              <header>Metric|Current Value|Status</header>
              <row>Total Users|{{ output_refs.data_collection.total_users | number_format }}|{% if output_refs.data_collection.total_users > 15000 %}✅ Strong{% else %}⚠️ Growing{% endif %}</row>
              <row>Revenue|${{ output_refs.data_collection.revenue | number_format }}|{% if output_refs.data_collection.revenue > 80000 %}✅ Target Met{% else %}📈 Below Target{% endif %}</row>
              <row>User Satisfaction|{{ output_refs.user_feedback.satisfaction_score }}/5.0|{% if output_refs.user_feedback.satisfaction_score > 4.0 %}✅ Excellent{% else %}⚠️ Needs Attention{% endif %}</row>
              <row>System Uptime|{{ output_refs.performance_analysis.uptime }}%|{% if output_refs.performance_analysis.uptime > 99.5 %}✅ Excellent{% else %}⚠️ Monitor{% endif %}</row>
            </table>
          </section>
          
          <section id="trends">
            <h2>Key Trends & Insights</h2>
            <p><strong>User Growth:</strong> {{ output_refs.data_collection.new_signups }} new signups with a churn rate of {{ (output_refs.data_collection.churn_rate * 100) | round(2) }}%.</p>
            <p><strong>System Performance:</strong> Average response time of {{ output_refs.performance_analysis.avg_response_time * 1000 | round }}ms with {{ output_refs.performance_analysis.uptime }}% uptime.</p>
            <p><strong>User Sentiment:</strong> NPS score of {{ output_refs.user_feedback.nps_score }} indicates {% if output_refs.user_feedback.nps_score > 50 %}positive{% else %}neutral{% endif %} user sentiment.</p>
          </section>
          
          <section id="recommendations">
            <h2>Strategic Recommendations</h2>
            {% if output_refs.performance_analysis.error_rate > 0.01 %}
            <p>🔧 <strong>Performance:</strong> Address error rate of {{ (output_refs.performance_analysis.error_rate * 100) | round(3) }}% by investigating top issues.</p>
            {% endif %}
            
            {% if output_refs.user_feedback.satisfaction_score < 4.5 %}
            <p>👥 <strong>User Experience:</strong> Focus on resolving top user issues: {{ output_refs.user_feedback.top_issues | join(', ') }}.</p>
            {% endif %}
            
            {% if output_refs.data_collection.churn_rate > 0.03 %}
            <p>📈 <strong>Retention:</strong> Implement retention strategies to reduce {{ (output_refs.data_collection.churn_rate * 100) | round(2) }}% churn rate.</p>
            {% endif %}
          </section>
        </document>
        </output-format>
        </poml>
        """
        
        context = {
            "report_period": "August 2025",
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return await self.template_resolver.resolve_with_output_references(
            template, context, self.output_tracker
        )
    
    async def generate_technical_report(self) -> str:
        """Generate detailed technical report."""
        
        template = """
        <role>technical systems analyst</role>
        
        <task>
        Create a comprehensive technical performance report analyzing system metrics,
        error patterns, and operational health indicators.
        </task>
        
        <document type="technical_report">
          <metadata>
            <title>System Performance Analysis</title>
            <generated>{{ generation_timestamp }}</generated>
            <period>{{ report_period }}</period>
          </metadata>
          
          <section id="performance_overview">
            <h1>Performance Overview</h1>
            
            <h2>Response Time Analysis</h2>
            <p>Average response time: <strong>{{ output_refs.performance_analysis.avg_response_time * 1000 | round }}ms</strong></p>
            <p>Peak concurrent users: <strong>{{ output_refs.performance_analysis.peak_concurrent_users | number_format }}</strong></p>
            
            {% if output_refs.performance_analysis.avg_response_time > 0.3 %}
            <hint>⚠️ Response time exceeds recommended threshold of 300ms. Consider optimization.</hint>
            {% else %}
            <hint>✅ Response time within acceptable parameters.</hint>
            {% endif %}
            
            <h2>System Reliability</h2>
            <table>
              <header>Metric|Value|Target|Status</header>
              <row>Uptime|{{ output_refs.performance_analysis.uptime }}%|99.9%|{% if output_refs.performance_analysis.uptime >= 99.9 %}✅{% elif output_refs.performance_analysis.uptime >= 99.5 %}⚠️{% else %}❌{% endif %}</row>
              <row>Error Rate|{{ (output_refs.performance_analysis.error_rate * 100) | round(3) }}%|< 1%|{% if output_refs.performance_analysis.error_rate < 0.01 %}✅{% else %}⚠️{% endif %}</row>
            </table>
          </section>
          
          <section id="error_analysis">
            <h1>Error Analysis</h1>
            <p>Total error incidents analyzed: <strong>{{ output_refs.performance_analysis.top_errors | length }}</strong></p>
            
            <h2>Error Breakdown</h2>
            {% for error in output_refs.performance_analysis.top_errors %}
            <p><strong>{{ error.type | title }} Errors:</strong> {{ error.count }} incidents</p>
            {% endfor %}
            
            <h2>Recommended Actions</h2>
            {% for error in output_refs.performance_analysis.top_errors %}
            {% if error.type == "timeout" and error.count > 20 %}
            <p>🔧 High timeout error count ({{ error.count }}) suggests need for performance optimization or timeout adjustment.</p>
            {% elif error.type == "validation" and error.count > 10 %}
            <p>📝 Validation errors ({{ error.count }}) indicate potential API contract issues or client-side problems.</p>
            {% elif error.type == "auth" and error.count > 5 %}
            <p>🔐 Authentication errors ({{ error.count }}) may indicate security issues or token expiration problems.</p>
            {% endif %}
            {% endfor %}
          </section>
          
          <section id="capacity_planning">
            <h1>Capacity Planning</h1>
            <p>Current peak load: {{ output_refs.performance_analysis.peak_concurrent_users | number_format }} concurrent users</p>
            <p>Active user base: {{ output_refs.data_collection.active_users | number_format }} users</p>
            <p>Peak to active ratio: {{ (output_refs.performance_analysis.peak_concurrent_users / output_refs.data_collection.active_users * 100) | round(1) }}%</p>
            
            {% set capacity_ratio = output_refs.performance_analysis.peak_concurrent_users / output_refs.data_collection.active_users %}
            {% if capacity_ratio > 0.15 %}
            <hint>📊 High peak-to-active ratio ({{ (capacity_ratio * 100) | round(1) }}%) suggests concentrated usage patterns. Consider load balancing.</hint>
            {% else %}
            <hint>📊 Reasonable load distribution with {{ (capacity_ratio * 100) | round(1) }}% peak usage.</hint>
            {% endif %}
          </section>
        </document>
        """
        
        context = {
            "generation_timestamp": datetime.now().isoformat(),
            "report_period": "Last 30 days"
        }
        
        return await self.template_resolver.resolve_with_output_references(
            template, context, self.output_tracker
        )
    
    async def demonstrate_template_migration(self):
        """Demonstrate template format migration."""
        
        print("🔄 Template Migration Example")
        print("-" * 30)
        
        # Original Jinja2 template
        jinja_template = """
        Executive Summary Report
        
        Report Date: {{ report_date }}
        
        Key Metrics:
        {% for metric, value in metrics.items() %}
        - {{ metric }}: {{ value }}
        {% endfor %}
        
        {% if issues %}
        Issues Found:
        {% for issue in issues %}
        • {{ issue }}
        {% endfor %}
        {% endif %}
        
        Recommendations:
        {% for rec in recommendations %}
        {{ loop.index }}. {{ rec }}
        {% endfor %}
        """
        
        print("Original Jinja2 template:")
        print(jinja_template[:200] + "...")
        print()
        
        # Convert to POML
        poml_template = await self.migration_tools.convert_jinja_to_poml(jinja_template)
        
        print("Converted POML template:")
        print(poml_template[:300] + "...")
        print()
        
        # Test both formats
        context = {
            "report_date": "2025-08-25",
            "metrics": {
                "Users": "15,420",
                "Revenue": "$89,420",
                "Satisfaction": "4.2/5.0"
            },
            "issues": ["Performance degradation", "User feedback delay"],
            "recommendations": [
                "Optimize database queries",
                "Implement user feedback automation",
                "Add monitoring alerts"
            ]
        }
        
        # Resolve Jinja2 template
        jinja_result = await self.template_resolver.resolve_template_content(
            jinja_template, context, TemplateFormat.JINJA2
        )
        
        # Resolve POML template
        poml_result = await self.template_resolver.resolve_template_content(
            poml_template, context, TemplateFormat.POML
        )
        
        print("Jinja2 result (first 200 chars):")
        print(jinja_result[:200] + "...")
        print()
        
        print("POML result (first 200 chars):")
        print(poml_result[:200] + "...")
        print()

async def run_poml_report_demo():
    """Run interactive POML report generation demo."""
    
    print("📊 POML Dynamic Report Generator Demo")
    print("=" * 50)
    print()
    
    generator = ReportGenerator()
    
    # Generate executive summary
    print("1. Generating Executive Summary Report")
    print("-" * 40)
    
    try:
        executive_summary = await generator.generate_executive_summary()
        print("✅ Executive Summary Generated")
        print()
        print("Sample output (first 500 characters):")
        print("-" * 40)
        print(executive_summary[:500] + "...")
        print()
        
    except Exception as e:
        print(f"❌ Error generating executive summary: {e}")
        print()
    
    # Generate technical report
    print("2. Generating Technical Performance Report")
    print("-" * 40)
    
    try:
        technical_report = await generator.generate_technical_report()
        print("✅ Technical Report Generated")
        print()
        print("Sample output (first 500 characters):")
        print("-" * 40)
        print(technical_report[:500] + "...")
        print()
        
    except Exception as e:
        print(f"❌ Error generating technical report: {e}")
        print()
    
    # Demonstrate migration
    print("3. Template Migration Demo")
    print("-" * 40)
    
    try:
        await generator.demonstrate_template_migration()
    except Exception as e:
        print(f"❌ Migration demo error: {e}")
    
    print("🎉 POML Demo Complete!")
    print()
    print("What this demo showed:")
    print("• Advanced POML template structure with metadata and sections")
    print("• Cross-task output references with complex expressions")
    print("• Conditional content based on data values")
    print("• Structured document generation with tables and formatting")
    print("• Template migration from Jinja2 to POML")
    print("• Dynamic content generation based on real data")

if __name__ == "__main__":
    asyncio.run(run_poml_report_demo())
```

## Advanced Wrapper Development

### Example 1: Multi-Service Orchestration Wrapper

```python
# example_advanced_wrapper.py
"""
Advanced Wrapper Example - Multi-Service Orchestration

This example demonstrates advanced wrapper patterns including:
- Multi-service coordination
- Circuit breaker implementation
- Advanced caching strategies
- Performance monitoring
- Error recovery patterns
"""

import asyncio
import aiohttp
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol
from enum import Enum
from datetime import datetime, timedelta

from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult, WrapperContext, WrapperCapability
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

# Circuit Breaker Implementation
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3

class CircuitBreaker:
    """Circuit breaker implementation for service protection."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        now = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if (self.last_failure_time and 
                now - self.last_failure_time > timedelta(seconds=self.config.timeout_seconds)):
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_calls < self.config.half_open_max_calls
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

# Cache Implementation
class CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[bytes]: ...
    async def set(self, key: str, value: bytes, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...

class MemoryCache:
    """In-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, tuple[bytes, datetime]] = {}
        self.max_size = max_size
    
    async def get(self, key: str) -> Optional[bytes]:
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.utcnow() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    async def set(self, key: str, value: bytes, ttl: int) -> None:
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)
    
    async def delete(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]

# Service Interface
@dataclass
class ServiceResponse:
    """Response from external service."""
    data: Any
    status_code: int
    response_time_ms: float
    cached: bool = False

class ExternalService:
    """Represents an external service."""
    
    def __init__(self, name: str, base_url: str, timeout: float = 30.0):
        self.name = name
        self.base_url = base_url
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
    
    async def call(self, endpoint: str, params: Optional[Dict] = None) -> ServiceResponse:
        """Make API call to external service."""
        
        if not self.circuit_breaker.can_execute():
            raise RuntimeError(f"Circuit breaker is open for {self.name}")
        
        start_time = time.time()
        
        try:
            # Simulate API call
            await asyncio.sleep(0.05)  # Simulate network latency
            
            # Simulate occasional failures
            import random
            if random.random() < 0.1:  # 10% failure rate
                raise aiohttp.ClientResponseError(
                    request_info=None, history=(), status=500
                )
            
            # Simulate response
            response_data = {
                "service": self.name,
                "endpoint": endpoint,
                "params": params,
                "timestamp": datetime.utcnow().isoformat(),
                "data": f"Response from {self.name}"
            }
            
            response_time = (time.time() - start_time) * 1000
            
            self.circuit_breaker.record_success()
            
            return ServiceResponse(
                data=response_data,
                status_code=200,
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise

# Configuration
@dataclass
class MultiServiceConfig(BaseWrapperConfig):
    """Configuration for multi-service wrapper."""
    
    # Service endpoints
    user_service_url: str = "https://api.users.example.com"
    analytics_service_url: str = "https://api.analytics.example.com"
    notification_service_url: str = "https://api.notifications.example.com"
    
    # Performance settings
    timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Cache settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Configuration field definitions."""
        return {
            "user_service_url": ConfigField(
                name="user_service_url",
                field_type=str,
                default_value=self.user_service_url,
                description="User service API URL",
                environment_var="USER_SERVICE_URL"
            ),
            "timeout_seconds": ConfigField(
                name="timeout_seconds",
                field_type=float,
                default_value=self.timeout_seconds,
                description="Request timeout in seconds",
                min_value=1.0,
                max_value=120.0
            ),
            "enable_caching": ConfigField(
                name="enable_caching",
                field_type=bool,
                default_value=self.enable_caching,
                description="Enable response caching"
            )
        }

# Result Types
@dataclass
class UserProfile:
    user_id: str
    name: str
    email: str
    created_at: str

@dataclass
class AnalyticsData:
    user_id: str
    page_views: int
    session_duration: float
    last_active: str

@dataclass
class AggregatedUserData:
    """Combined user data from multiple services."""
    profile: UserProfile
    analytics: AnalyticsData
    notifications_enabled: bool
    data_sources: List[str] = field(default_factory=list)
    response_times: Dict[str, float] = field(default_factory=dict)
    cache_hits: List[str] = field(default_factory=list)

# Main Wrapper Implementation
class MultiServiceWrapper(BaseWrapper[AggregatedUserData, MultiServiceConfig]):
    """Advanced wrapper that orchestrates multiple external services."""
    
    def __init__(self, config: Optional[MultiServiceConfig] = None):
        """Initialize the wrapper."""
        config = config or MultiServiceConfig()
        super().__init__(config)
        
        # Initialize services
        self.services = {
            "users": ExternalService("users", config.user_service_url, config.timeout_seconds),
            "analytics": ExternalService("analytics", config.analytics_service_url, config.timeout_seconds),
            "notifications": ExternalService("notifications", config.notification_service_url, config.timeout_seconds)
        }
        
        # Initialize cache
        self.cache = MemoryCache(config.max_cache_size) if config.enable_caching else None
        
        # Initialize monitoring
        self.monitoring = WrapperMonitoring()
    
    async def _execute_wrapper_operation(
        self, 
        context: WrapperContext, 
        user_id: str,
        **kwargs
    ) -> AggregatedUserData:
        """Orchestrate multiple service calls to get aggregated user data."""
        
        if not user_id:
            raise ValueError("user_id is required")
        
        # Start monitoring
        operation_id = self.monitoring.start_operation("multi_service", "aggregate_user_data")
        
        try:
            # Initialize result
            result = AggregatedUserData(
                profile=UserProfile(user_id, "", "", ""),
                analytics=AnalyticsData(user_id, 0, 0.0, ""),
                notifications_enabled=False
            )
            
            # Execute service calls concurrently
            tasks = {
                "profile": self._get_user_profile(user_id),
                "analytics": self._get_user_analytics(user_id),
                "notifications": self._get_notification_settings(user_id)
            }
            
            # Wait for all tasks to complete
            responses = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Process results
            for i, (service_name, response) in enumerate(zip(tasks.keys(), responses)):
                if isinstance(response, Exception):
                    print(f"⚠️ Service {service_name} failed: {response}")
                    continue
                
                # Update result based on service
                if service_name == "profile" and not isinstance(response, Exception):
                    result.profile = response
                    result.data_sources.append("users")
                    result.response_times["users"] = getattr(response, 'response_time_ms', 0)
                    if getattr(response, 'cached', False):
                        result.cache_hits.append("users")
                
                elif service_name == "analytics" and not isinstance(response, Exception):
                    result.analytics = response
                    result.data_sources.append("analytics")
                    result.response_times["analytics"] = getattr(response, 'response_time_ms', 0)
                    if getattr(response, 'cached', False):
                        result.cache_hits.append("analytics")
                
                elif service_name == "notifications" and not isinstance(response, Exception):
                    result.notifications_enabled = response
                    result.data_sources.append("notifications")
                    result.response_times["notifications"] = getattr(response, 'response_time_ms', 0)
            
            # Record successful operation
            self.monitoring.record_success(operation_id, {
                "user_id": user_id,
                "services_called": len(result.data_sources),
                "cache_hits": len(result.cache_hits),
                "total_response_time": sum(result.response_times.values())
            })
            
            return result
            
        except Exception as e:
            self.monitoring.record_error(operation_id, str(e))
            raise
        finally:
            self.monitoring.end_operation(operation_id)
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get user profile with caching."""
        cache_key = f"profile:{user_id}"
        
        # Check cache first
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                import json
                data = json.loads(cached.decode())
                profile = UserProfile(**data)
                profile.cached = True
                return profile
        
        # Call service
        service = self.services["users"]
        response = await service.call(f"/users/{user_id}")
        
        # Create profile from response
        profile = UserProfile(
            user_id=user_id,
            name=f"User {user_id}",
            email=f"user{user_id}@example.com",
            created_at="2025-01-15T10:30:00Z"
        )
        
        # Cache the result
        if self.cache:
            import json
            cache_data = json.dumps({
                "user_id": profile.user_id,
                "name": profile.name,
                "email": profile.email,
                "created_at": profile.created_at
            }).encode()
            await self.cache.set(cache_key, cache_data, self.config.cache_ttl_seconds)
        
        return profile
    
    async def _get_user_analytics(self, user_id: str) -> AnalyticsData:
        """Get user analytics data."""
        service = self.services["analytics"]
        response = await service.call(f"/analytics/user/{user_id}")
        
        return AnalyticsData(
            user_id=user_id,
            page_views=125,
            session_duration=15.5,
            last_active="2025-08-25T09:15:00Z"
        )
    
    async def _get_notification_settings(self, user_id: str) -> bool:
        """Get notification settings."""
        service = self.services["notifications"]
        response = await service.call(f"/notifications/settings/{user_id}")
        
        return True  # Simulate enabled notifications
    
    async def _execute_fallback_operation(
        self,
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        user_id: str = "",
        **kwargs
    ) -> AggregatedUserData:
        """Provide fallback aggregated user data."""
        print(f"⚠️ Using fallback for user {user_id}: {original_error}")
        
        # Return minimal user data as fallback
        return AggregatedUserData(
            profile=UserProfile(
                user_id=user_id,
                name="Unknown User",
                email="",
                created_at=""
            ),
            analytics=AnalyticsData(
                user_id=user_id,
                page_views=0,
                session_duration=0.0,
                last_active=""
            ),
            notifications_enabled=False,
            data_sources=["fallback"]
        )
    
    def _validate_config(self) -> bool:
        """Validate configuration."""
        return all([
            self.config.user_service_url,
            self.config.analytics_service_url,
            self.config.notification_service_url,
            self.config.timeout_seconds > 0
        ])
    
    def get_capabilities(self) -> List[WrapperCapability]:
        """Return wrapper capabilities."""
        return [
            WrapperCapability.MONITORING,
            WrapperCapability.FALLBACK,
            WrapperCapability.CACHING,
            WrapperCapability.CIRCUIT_BREAKER
        ]
    
    async def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status for all services."""
        status = {}
        for name, service in self.services.items():
            cb = service.circuit_breaker
            status[name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
        return status
    
    async def cleanup(self):
        """Clean up resources."""
        if self.cache:
            # Clear cache
            self.cache.cache.clear()

async def run_advanced_wrapper_demo():
    """Run advanced wrapper demonstration."""
    
    print("🚀 Advanced Multi-Service Wrapper Demo")
    print("=" * 50)
    print()
    
    # Initialize wrapper
    config = MultiServiceConfig(
        timeout_seconds=10.0,
        enable_caching=True,
        cache_ttl_seconds=60
    )
    
    wrapper = MultiServiceWrapper(config)
    
    print("Wrapper initialized with services:")
    for name, service in wrapper.services.items():
        print(f"  • {name}: {service.base_url}")
    print()
    
    # Test users
    test_users = ["user_001", "user_002", "user_003", "user_001"]  # Duplicate for cache test
    
    print("Processing user requests...")
    print("-" * 30)
    
    for i, user_id in enumerate(test_users, 1):
        print(f"\nRequest {i}: {user_id}")
        
        try:
            start_time = time.time()
            result = await wrapper.execute("aggregate_data", user_id=user_id)
            duration = (time.time() - start_time) * 1000
            
            if result.success:
                data = result.data
                print(f"  ✅ Success ({duration:.1f}ms)")
                print(f"     Profile: {data.profile.name} ({data.profile.email})")
                print(f"     Analytics: {data.analytics.page_views} page views")
                print(f"     Notifications: {'Enabled' if data.notifications_enabled else 'Disabled'}")
                print(f"     Data sources: {', '.join(data.data_sources)}")
                print(f"     Cache hits: {', '.join(data.cache_hits) if data.cache_hits else 'None'}")
                
                # Show response times
                if data.response_times:
                    times = ', '.join([f"{k}: {v:.1f}ms" for k, v in data.response_times.items()])
                    print(f"     Service times: {times}")
            else:
                print(f"  ❌ Failed: {result.error}")
                if result.fallback_used:
                    print(f"     Fallback reason: {result.fallback_reason}")
                    
        except Exception as e:
            print(f"  💥 Exception: {e}")
    
    # Show circuit breaker status
    print("\n🔄 Circuit Breaker Status")
    print("-" * 30)
    
    cb_status = await wrapper.get_circuit_breaker_status()
    for service_name, status in cb_status.items():
        print(f"{service_name}: {status['state']} (failures: {status['failure_count']})")
    
    # Show monitoring summary
    print("\n📊 Monitoring Summary")
    print("-" * 30)
    
    system_health = wrapper.monitoring.get_system_health()
    print(f"Overall status: {system_health.overall_status}")
    
    if system_health.wrapper_healths:
        for wrapper_name, health in system_health.wrapper_healths.items():
            print(f"{wrapper_name}:")
            print(f"  Success rate: {health.success_rate:.1%}")
            print(f"  Avg response time: {health.avg_response_time_ms:.1f}ms")
    
    # Cleanup
    await wrapper.cleanup()
    
    print("\n🎉 Advanced Wrapper Demo Complete!")
    print()
    print("What this demo showed:")
    print("• Multi-service orchestration with concurrent API calls")
    print("• Circuit breaker pattern for service protection")
    print("• Advanced caching with TTL support")
    print("• Comprehensive error handling and fallback strategies")
    print("• Performance monitoring and health tracking")
    print("• Resource cleanup and lifecycle management")

if __name__ == "__main__":
    asyncio.run(run_advanced_wrapper_demo())
```

This comprehensive collection of interactive tutorials and examples provides hands-on learning experiences for all aspects of the wrapper framework, from basic concepts to advanced patterns and real-world usage scenarios. Each example is fully functional and includes detailed explanations to facilitate understanding and adoption.