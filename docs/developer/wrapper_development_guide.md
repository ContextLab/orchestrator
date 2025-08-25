# Wrapper Development Guide

## Overview

This comprehensive guide covers everything you need to know to develop, test, and deploy custom wrapper integrations within the orchestrator framework. Whether you're integrating external APIs, databases, or custom tools, this guide will help you build robust, maintainable, and scalable wrapper implementations.

## Table of Contents

- [Getting Started](#getting-started)
- [Architecture Fundamentals](#architecture-fundamentals)
- [Development Workflow](#development-workflow)
- [Implementation Patterns](#implementation-patterns)
- [Testing Strategies](#testing-strategies)
- [Performance Optimization](#performance-optimization)
- [Deployment and Operations](#deployment-and-operations)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)

## Getting Started

### Prerequisites

Before starting wrapper development, ensure you have:

- **Python 3.8+**: Required for the wrapper framework
- **Orchestrator Framework**: Latest version installed
- **Development Environment**: IDE with Python support and type checking
- **Testing Tools**: pytest, asyncio testing capabilities
- **Version Control**: Git for code management

### Development Environment Setup

```bash
# Create development environment
python -m venv wrapper_dev
source wrapper_dev/bin/activate  # On Windows: wrapper_dev\Scripts\activate

# Install dependencies
pip install orchestrator-framework[dev]
pip install pytest pytest-asyncio pytest-cov
pip install mypy black flake8

# Verify installation
python -c "from src.orchestrator.core.wrapper_base import BaseWrapper; print('Framework ready')"
```

### Project Structure

Organize your wrapper project with this recommended structure:

```
my_wrapper_project/
├── src/
│   └── my_wrapper/
│       ├── __init__.py
│       ├── wrapper.py          # Main wrapper implementation
│       ├── config.py           # Configuration classes
│       ├── client.py           # External service client
│       └── exceptions.py       # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── test_wrapper.py         # Wrapper tests
│   ├── test_config.py          # Configuration tests
│   └── fixtures/               # Test fixtures and data
├── examples/
│   ├── basic_usage.py          # Basic usage examples
│   └── advanced_features.py    # Advanced feature examples
├── docs/
│   ├── README.md              # Wrapper documentation
│   └── api_reference.md       # API reference
├── requirements.txt           # Dependencies
├── setup.py                  # Package setup
└── pyproject.toml            # Build configuration
```

## Architecture Fundamentals

### Core Components Overview

Understanding the wrapper framework architecture is essential for effective development:

```python
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult, WrapperContext
from src.orchestrator.core.wrapper_config import BaseWrapperConfig
from src.orchestrator.core.feature_flags import FeatureFlagManager
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

# The four pillars of wrapper architecture:
# 1. BaseWrapper - Core functionality and lifecycle
# 2. Configuration System - Validated, environment-aware configuration
# 3. Feature Flags - Safe rollout and A/B testing
# 4. Monitoring - Comprehensive observability
```

### Wrapper Lifecycle

Every wrapper operation follows this lifecycle:

```
1. Configuration Validation
2. Feature Flag Evaluation  
3. Context Creation
4. Monitoring Start
5. Wrapper Operation Execution
   ├─ Success → Result Processing
   └─ Failure → Fallback Execution
6. Monitoring End
7. Result Return
```

### Generic Type System

The wrapper framework uses generics for type safety:

```python
from typing import TypeVar, Generic
from dataclasses import dataclass

# Define your result type
@dataclass
class APIResult:
    data: dict
    status_code: int
    headers: dict

# Define your configuration type
@dataclass  
class APIConfig(BaseWrapperConfig):
    endpoint: str
    api_key: str

# Use generics in your wrapper
class MyAPIWrapper(BaseWrapper[APIResult, APIConfig]):
    # TypeVar T = APIResult
    # TypeVar C = APIConfig
    pass
```

## Development Workflow

### Step 1: Design Your Wrapper

Before coding, design your wrapper by answering these questions:

1. **What external service/tool are you integrating?**
2. **What data types will you return?**
3. **What configuration parameters are needed?**
4. **What operations will you support?**
5. **What fallback strategies make sense?**
6. **What monitoring metrics are important?**

### Step 2: Define Configuration

Create a configuration class that extends `BaseWrapperConfig`:

```python
# src/my_wrapper/config.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField

@dataclass
class WeatherAPIConfig(BaseWrapperConfig):
    """Configuration for Weather API wrapper."""
    
    # Required fields
    api_key: str = ""
    base_url: str = "https://api.weatherapi.com/v1"
    
    # Optional fields with defaults
    timeout_seconds: float = 30.0
    max_retries: int = 3
    cache_duration_minutes: int = 15
    
    # Advanced settings
    default_units: str = "metric"
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr"])
    rate_limit_per_minute: int = 1000
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Define configuration fields with validation rules."""
        return {
            "api_key": ConfigField(
                name="api_key",
                field_type=str,
                default_value="",
                description="Weather API authentication key",
                required=True,
                sensitive=True,  # Will be masked in logs
                environment_var="WEATHER_API_KEY"
            ),
            "base_url": ConfigField(
                name="base_url", 
                field_type=str,
                default_value=self.base_url,
                description="API base URL",
                required=True,
                validator=self._validate_url
            ),
            "timeout_seconds": ConfigField(
                name="timeout_seconds",
                field_type=float,
                default_value=self.timeout_seconds,
                description="Request timeout in seconds",
                min_value=1.0,
                max_value=120.0
            ),
            "default_units": ConfigField(
                name="default_units",
                field_type=str,
                default_value=self.default_units,
                description="Default unit system",
                allowed_values=["metric", "imperial", "kelvin"]
            )
        }
    
    def _validate_url(self, url: str) -> bool:
        """Custom URL validation."""
        return url.startswith(("http://", "https://"))
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "Orchestrator-WeatherWrapper/1.0"
        }
```

### Step 3: Define Data Types

Create type-safe data structures for your wrapper:

```python
# src/my_wrapper/types.py
from dataclasses import dataclass
from typing import Optional, List, Any
from datetime import datetime

@dataclass
class WeatherCondition:
    """Weather condition data."""
    temperature: float
    humidity: int
    pressure: float
    description: str
    icon_code: str

@dataclass
class WeatherForecast:
    """Weather forecast data."""
    date: datetime
    high_temp: float
    low_temp: float
    condition: WeatherCondition
    precipitation_chance: int

@dataclass
class WeatherResult:
    """Complete weather API result."""
    location: str
    current: WeatherCondition
    forecast: List[WeatherForecast]
    last_updated: datetime
    units: str
    
    # Metadata
    api_response_time_ms: Optional[float] = None
    cache_hit: bool = False
    data_age_minutes: Optional[float] = None
```

### Step 4: Implement the Wrapper

Create your main wrapper class:

```python
# src/my_wrapper/wrapper.py
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult, WrapperContext, WrapperCapability

from .config import WeatherAPIConfig
from .types import WeatherResult, WeatherCondition, WeatherForecast
from .exceptions import WeatherAPIError, RateLimitError

logger = logging.getLogger(__name__)

class WeatherAPIWrapper(BaseWrapper[WeatherResult, WeatherAPIConfig]):
    """Wrapper for Weather API integration with caching and rate limiting."""
    
    def __init__(self, config: Optional[WeatherAPIConfig] = None):
        """Initialize wrapper with configuration and caching."""
        config = config or WeatherAPIConfig()
        super().__init__(config)
        
        # Internal state
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, tuple] = {}  # Simple in-memory cache
        self._rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
    
    async def _execute_wrapper_operation(
        self, 
        context: WrapperContext, 
        location: str,
        days: int = 1,
        units: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> WeatherResult:
        """Execute weather API request with caching and rate limiting."""
        
        # Validate inputs
        if not location or not location.strip():
            raise WeatherAPIError("Location cannot be empty")
        
        if days < 1 or days > 10:
            raise WeatherAPIError("Days must be between 1 and 10")
        
        units = units or self.config.default_units
        
        # Check cache first
        cache_key = f"{location}:{days}:{units}:{language}"
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit for location: {location}")
            return cached_result
        
        # Rate limiting
        await self._rate_limiter.acquire()
        
        # Initialize HTTP session if needed
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.get_headers()
            )
        
        # Prepare API request
        url = f"{self.config.base_url}/forecast.json"
        params = {
            "q": location,
            "days": days,
            "units": units,
            "lang": language
        }
        
        start_time = datetime.utcnow()
        
        try:
            # Make API request with retries
            async with self._session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Process response
                result = self._process_api_response(data, units)
                result.api_response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Cache result
                self._cache_result(cache_key, result)
                
                logger.info(f"Weather data retrieved for {location} in {result.api_response_time_ms:.1f}ms")
                return result
                
        except aiohttp.ClientResponseError as e:
            if e.status == 429:  # Rate limited
                raise RateLimitError(f"Rate limit exceeded: {e}")
            elif e.status == 401:
                raise WeatherAPIError(f"Invalid API key: {e}")
            elif e.status == 404:
                raise WeatherAPIError(f"Location not found: {location}")
            else:
                raise WeatherAPIError(f"API request failed: {e}")
                
        except asyncio.TimeoutError:
            raise WeatherAPIError(f"Request timeout after {self.config.timeout_seconds}s")
            
        except Exception as e:
            logger.error(f"Unexpected error in weather API request: {e}")
            raise WeatherAPIError(f"Unexpected error: {e}")
    
    async def _execute_fallback_operation(
        self,
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        location: str = "",
        **kwargs
    ) -> WeatherResult:
        """Execute fallback when primary operation fails."""
        
        logger.warning(f"Using fallback for location: {location}, error: {original_error}")
        
        # Strategy 1: Return cached data even if stale
        cache_key = f"{location}:1:{self.config.default_units}:en"  
        stale_result = self._check_cache(cache_key, allow_stale=True)
        if stale_result:
            logger.info("Using stale cached data as fallback")
            return stale_result
        
        # Strategy 2: Return mock data for development/testing
        if self.config.enable_fallback_mock:
            return self._create_mock_weather_result(location)
        
        # Strategy 3: Return error result with fallback data
        return WeatherResult(
            location=location,
            current=WeatherCondition(
                temperature=20.0,
                humidity=50,
                pressure=1013.25,
                description="Data unavailable", 
                icon_code="unknown"
            ),
            forecast=[],
            last_updated=datetime.utcnow(),
            units=self.config.default_units,
            data_age_minutes=None
        )
    
    def _validate_config(self) -> bool:
        """Validate wrapper configuration."""
        errors = []
        
        if not self.config.api_key:
            errors.append("API key is required")
        
        if not self.config.base_url:
            errors.append("Base URL is required")
        
        if self.config.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        if self.config.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        if self.config.rate_limit_per_minute <= 0:
            errors.append("Rate limit must be positive")
        
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return False
        
        return True
    
    def get_capabilities(self) -> List[WrapperCapability]:
        """Return wrapper capabilities."""
        return [
            WrapperCapability.MONITORING,
            WrapperCapability.FALLBACK,
            WrapperCapability.CONFIGURATION_MANAGEMENT,
            WrapperCapability.CACHING,
            WrapperCapability.RATE_LIMITING
        ]
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
        
        # Clear cache
        self._cache.clear()
        
        logger.info("Weather API wrapper cleaned up")
    
    # Helper methods
    def _process_api_response(self, data: Dict[str, Any], units: str) -> WeatherResult:
        """Process raw API response into structured result."""
        location_data = data.get("location", {})
        current_data = data.get("current", {})
        forecast_data = data.get("forecast", {}).get("forecastday", [])
        
        # Process current conditions
        current = WeatherCondition(
            temperature=current_data.get("temp_c" if units == "metric" else "temp_f", 0),
            humidity=current_data.get("humidity", 0),
            pressure=current_data.get("pressure_mb", 0),
            description=current_data.get("condition", {}).get("text", ""),
            icon_code=current_data.get("condition", {}).get("code", "")
        )
        
        # Process forecast
        forecast = []
        for day_data in forecast_data:
            day = day_data.get("day", {})
            forecast.append(WeatherForecast(
                date=datetime.fromisoformat(day_data.get("date")),
                high_temp=day.get("maxtemp_c" if units == "metric" else "maxtemp_f", 0),
                low_temp=day.get("mintemp_c" if units == "metric" else "mintemp_f", 0),
                condition=WeatherCondition(
                    temperature=day.get("avgtemp_c" if units == "metric" else "avgtemp_f", 0),
                    humidity=day.get("avghumidity", 0),
                    pressure=0,  # Not available in forecast
                    description=day.get("condition", {}).get("text", ""),
                    icon_code=day.get("condition", {}).get("code", "")
                ),
                precipitation_chance=day.get("daily_chance_of_rain", 0)
            ))
        
        return WeatherResult(
            location=f"{location_data.get('name', '')}, {location_data.get('country', '')}",
            current=current,
            forecast=forecast,
            last_updated=datetime.utcnow(),
            units=units
        )
    
    def _check_cache(self, key: str, allow_stale: bool = False) -> Optional[WeatherResult]:
        """Check cache for existing result."""
        if key not in self._cache:
            return None
        
        result, timestamp = self._cache[key]
        age_minutes = (datetime.utcnow() - timestamp).total_seconds() / 60
        
        # Check if cache is still valid
        if not allow_stale and age_minutes > self.config.cache_duration_minutes:
            del self._cache[key]
            return None
        
        # Update metadata
        result.cache_hit = True
        result.data_age_minutes = age_minutes
        
        return result
    
    def _cache_result(self, key: str, result: WeatherResult) -> None:
        """Cache result with timestamp."""
        self._cache[key] = (result, datetime.utcnow())
    
    def _create_mock_weather_result(self, location: str) -> WeatherResult:
        """Create mock weather result for fallback."""
        return WeatherResult(
            location=location,
            current=WeatherCondition(
                temperature=22.5,
                humidity=65,
                pressure=1015.3,
                description="Mock data - partly cloudy",
                icon_code="mock"
            ),
            forecast=[],
            last_updated=datetime.utcnow(),
            units=self.config.default_units
        )

class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def acquire(self) -> None:
        """Acquire rate limit token."""
        now = datetime.utcnow()
        
        # Remove old requests
        cutoff = now - timedelta(minutes=1)
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        # Check if we can make request
        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make request
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.requests.append(now)
```

### Step 5: Define Custom Exceptions

Create specific exceptions for your wrapper:

```python
# src/my_wrapper/exceptions.py
class WeatherAPIException(Exception):
    """Base exception for Weather API wrapper."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code

class WeatherAPIError(WeatherAPIException):
    """General Weather API error."""
    pass

class RateLimitError(WeatherAPIException):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str):
        super().__init__(message, "RATE_LIMIT_EXCEEDED")

class AuthenticationError(WeatherAPIException):
    """Authentication error."""
    
    def __init__(self, message: str):
        super().__init__(message, "AUTHENTICATION_FAILED")

class LocationNotFoundError(WeatherAPIException):
    """Location not found error."""
    
    def __init__(self, location: str):
        super().__init__(f"Location not found: {location}", "LOCATION_NOT_FOUND")
        self.location = location
```

## Testing Strategies

### Unit Testing

Create comprehensive unit tests for your wrapper:

```python
# tests/test_wrapper.py
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import aiohttp

from src.my_wrapper.wrapper import WeatherAPIWrapper
from src.my_wrapper.config import WeatherAPIConfig
from src.my_wrapper.types import WeatherResult, WeatherCondition
from src.my_wrapper.exceptions import WeatherAPIError, RateLimitError

class TestWeatherAPIWrapper:
    """Test suite for WeatherAPIWrapper."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return WeatherAPIConfig(
            api_key="test_key_123",
            base_url="https://api.test.com/v1",
            timeout_seconds=10.0,
            cache_duration_minutes=5
        )
    
    @pytest.fixture
    def wrapper(self, config):
        """Create wrapper instance."""
        return WeatherAPIWrapper(config)
    
    @pytest.fixture
    def mock_api_response(self):
        """Mock API response data."""
        return {
            "location": {"name": "London", "country": "UK"},
            "current": {
                "temp_c": 18.5,
                "humidity": 72,
                "pressure_mb": 1012.0,
                "condition": {"text": "Partly cloudy", "code": "116"}
            },
            "forecast": {
                "forecastday": [
                    {
                        "date": "2025-08-25",
                        "day": {
                            "maxtemp_c": 22.0,
                            "mintemp_c": 15.0,
                            "avgtemp_c": 18.5,
                            "avghumidity": 70,
                            "condition": {"text": "Sunny", "code": "113"},
                            "daily_chance_of_rain": 10
                        }
                    }
                ]
            }
        }
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = WeatherAPIConfig(api_key="test", base_url="https://api.test.com")
        wrapper = WeatherAPIWrapper(config)
        assert wrapper._validate_config() is True
        
        # Invalid config - missing API key
        config = WeatherAPIConfig(api_key="", base_url="https://api.test.com")
        wrapper = WeatherAPIWrapper(config)
        assert wrapper._validate_config() is False
        
        # Invalid config - bad timeout
        config = WeatherAPIConfig(api_key="test", timeout_seconds=-1)
        wrapper = WeatherAPIWrapper(config)
        assert wrapper._validate_config() is False
    
    @pytest.mark.asyncio
    async def test_successful_api_request(self, wrapper, mock_api_response):
        """Test successful API request."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Execute request
            result = await wrapper.execute("forecast", location="London", days=1)
            
            # Verify result
            assert result.success is True
            assert isinstance(result.data, WeatherResult)
            assert result.data.location == "London, UK"
            assert result.data.current.temperature == 18.5
            assert len(result.data.forecast) == 1
            assert result.data.units == "metric"
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, wrapper):
        """Test API error handling."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock error response
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=404
            )
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Execute request
            result = await wrapper.execute("forecast", location="InvalidLocation")
            
            # Should use fallback
            assert result.success is False
            assert result.fallback_used is True
            assert "Location not found" in result.error
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, wrapper):
        """Test rate limiting functionality."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup rate limit error
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
                request_info=Mock(),
                history=(),
                status=429
            )
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Execute request
            result = await wrapper.execute("forecast", location="London")
            
            # Should handle rate limit
            assert result.success is False
            assert result.fallback_used is True
    
    @pytest.mark.asyncio
    async def test_caching(self, wrapper, mock_api_response):
        """Test caching functionality."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_api_response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # First request
            result1 = await wrapper.execute("forecast", location="London")
            assert result1.success is True
            assert result1.data.cache_hit is False
            
            # Second request (should be cached)
            result2 = await wrapper.execute("forecast", location="London")
            assert result2.success is True
            assert result2.data.cache_hit is True
            
            # Verify API was only called once
            assert mock_get.call_count == 1
    
    @pytest.mark.asyncio
    async def test_fallback_strategies(self, wrapper):
        """Test different fallback strategies."""
        # Test fallback with stale cache
        wrapper._cache["London:1:metric:en"] = (
            WeatherResult(
                location="London",
                current=WeatherCondition(20.0, 60, 1010.0, "Cached", "cache"),
                forecast=[],
                last_updated=datetime.utcnow(),
                units="metric"
            ),
            datetime.utcnow()
        )
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Setup API failure
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = Exception("API Down")
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Execute request
            result = await wrapper.execute("forecast", location="London")
            
            # Should use cached data
            assert result.success is False  # Primary failed
            assert result.fallback_used is True
            assert result.data.current.description == "Cached"
    
    @pytest.mark.asyncio
    async def test_input_validation(self, wrapper):
        """Test input validation."""
        # Empty location
        result = await wrapper.execute("forecast", location="")
        assert result.success is False
        assert "empty" in result.error.lower()
        
        # Invalid days
        result = await wrapper.execute("forecast", location="London", days=15)
        assert result.success is False
        assert "between 1 and 10" in result.error
    
    @pytest.mark.asyncio
    async def test_cleanup(self, wrapper):
        """Test resource cleanup."""
        # Initialize session
        await wrapper.execute("forecast", location="London")
        
        # Cleanup
        await wrapper.cleanup()
        
        # Verify session is closed
        assert wrapper._session is None
        assert len(wrapper._cache) == 0
    
    def test_capabilities(self, wrapper):
        """Test wrapper capabilities."""
        capabilities = wrapper.get_capabilities()
        expected_capabilities = [
            "monitoring", "fallback", "configuration_management", 
            "caching", "rate_limiting"
        ]
        
        for capability in expected_capabilities:
            assert any(cap.value == capability for cap in capabilities)
```

### Integration Testing

Create integration tests that use real or mock services:

```python
# tests/test_integration.py
import pytest
import asyncio
import os
from src.my_wrapper.wrapper import WeatherAPIWrapper
from src.my_wrapper.config import WeatherAPIConfig

class TestWeatherAPIIntegration:
    """Integration tests for Weather API wrapper."""
    
    @pytest.fixture
    def real_config(self):
        """Configuration for real API testing (if API key available)."""
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration testing")
        
        return WeatherAPIConfig(
            api_key=api_key,
            timeout_seconds=30.0,
            cache_duration_minutes=1
        )
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_api_request(self, real_config):
        """Test with real API (requires API key)."""
        wrapper = WeatherAPIWrapper(real_config)
        
        try:
            result = await wrapper.execute("forecast", location="London", days=3)
            
            # Verify result structure
            assert result.success is True
            assert result.data is not None
            assert result.data.location
            assert result.data.current
            assert len(result.data.forecast) <= 3
            assert result.data.api_response_time_ms > 0
            
        finally:
            await wrapper.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, real_config):
        """Test performance characteristics."""
        wrapper = WeatherAPIWrapper(real_config)
        
        try:
            # Benchmark multiple requests
            locations = ["London", "Paris", "Tokyo", "New York", "Sydney"]
            response_times = []
            
            for location in locations:
                result = await wrapper.execute("forecast", location=location)
                if result.success and result.data.api_response_time_ms:
                    response_times.append(result.data.api_response_time_ms)
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                print(f"Average response time: {avg_response_time:.2f}ms")
                
                # Performance assertions
                assert avg_response_time < 2000  # Less than 2 seconds
                assert max(response_times) < 5000  # No request over 5 seconds
                
        finally:
            await wrapper.cleanup()
```

### Performance Testing

Create performance benchmarks:

```python
# tests/test_performance.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from src.my_wrapper.wrapper import WeatherAPIWrapper
from src.my_wrapper.config import WeatherAPIConfig

class TestPerformance:
    """Performance tests for wrapper."""
    
    @pytest.fixture
    def performance_config(self):
        """Configuration optimized for performance testing."""
        return WeatherAPIConfig(
            api_key="test_key",
            timeout_seconds=1.0,  # Short timeout for testing
            cache_duration_minutes=60,  # Long cache for testing
            rate_limit_per_minute=1000  # High rate limit
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, performance_config):
        """Test concurrent request handling."""
        wrapper = WeatherAPIWrapper(performance_config)
        
        try:
            # Mock successful responses
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_response = AsyncMock()
                mock_response.json.return_value = {"mock": "data"}
                mock_response.raise_for_status.return_value = None
                mock_get.return_value.__aenter__.return_value = mock_response
                
                # Execute concurrent requests
                tasks = []
                for i in range(50):
                    task = wrapper.execute("forecast", location=f"City{i}")
                    tasks.append(task)
                
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start_time
                
                # Analyze results
                successful = sum(1 for r in results if r.success if hasattr(r, 'success') else 0)
                
                print(f"Processed {len(tasks)} requests in {duration:.2f}s")
                print(f"Success rate: {successful/len(tasks):.2%}")
                print(f"Throughput: {len(tasks)/duration:.1f} req/s")
                
                # Performance assertions
                assert duration < 10.0  # Complete within 10 seconds
                assert successful >= len(tasks) * 0.95  # 95% success rate
                
        finally:
            await wrapper.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, performance_config):
        """Test memory usage under load."""
        import psutil
        import gc
        
        wrapper = WeatherAPIWrapper(performance_config)
        process = psutil.Process()
        
        try:
            # Measure baseline memory
            gc.collect()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute many requests to stress memory
            with patch('aiohttp.ClientSession.get') as mock_get:
                mock_response = AsyncMock()
                mock_response.json.return_value = {"large": "data" * 1000}
                mock_response.raise_for_status.return_value = None
                mock_get.return_value.__aenter__.return_value = mock_response
                
                for i in range(100):
                    await wrapper.execute("forecast", location=f"City{i}")
                
                # Measure memory after requests
                gc.collect()
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - baseline_memory
                
                print(f"Memory increase: {memory_increase:.1f}MB")
                
                # Memory usage should be reasonable
                assert memory_increase < 100  # Less than 100MB increase
                
        finally:
            await wrapper.cleanup()
```

## Performance Optimization

### Async Best Practices

```python
# Use connection pooling
class OptimizedWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)
        # Reuse session across requests
        self._session = None
        # Connection pool configuration
        self._connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            enable_cleanup_closed=True
        )
    
    async def _get_session(self):
        """Get or create HTTP session with optimal settings."""
        if not self._session or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout_seconds,
                connect=5.0,  # Connection timeout
                sock_read=30.0  # Socket read timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers=self.config.get_headers()
            )
        
        return self._session
```

### Caching Strategies

```python
# Implement intelligent caching
from typing import Protocol
import hashlib
import json

class CacheBackend(Protocol):
    async def get(self, key: str) -> Optional[bytes]: ...
    async def set(self, key: str, value: bytes, ttl: int) -> None: ...
    async def delete(self, key: str) -> None: ...

class RedisCache:
    """Redis-based cache backend."""
    
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def get(self, key: str) -> Optional[bytes]:
        return await self.redis.get(key)
    
    async def set(self, key: str, value: bytes, ttl: int) -> None:
        await self.redis.setex(key, ttl, value)

class CachedWrapper(BaseWrapper):
    def __init__(self, config, cache_backend: CacheBackend):
        super().__init__(config)
        self.cache = cache_backend
    
    def _generate_cache_key(self, operation: str, **params) -> str:
        """Generate deterministic cache key."""
        # Include wrapper version for cache invalidation
        key_data = {
            "wrapper_version": "1.0.0",
            "operation": operation,
            "params": sorted(params.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"wrapper:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[WrapperResult]:
        """Get result from cache."""
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            try:
                data = json.loads(cached_data.decode())
                return WrapperResult.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                # Invalid cache data, delete it
                await self.cache.delete(cache_key)
        return None
    
    async def _cache_result(self, cache_key: str, result: WrapperResult, ttl: int):
        """Cache result."""
        try:
            data = result.to_dict()
            await self.cache.set(cache_key, json.dumps(data).encode(), ttl)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
```

### Error Handling Optimization

```python
# Implement circuit breaker pattern
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"       # Failing, bypass to fallback
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if operation can execute."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if datetime.utcnow() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class ResilientWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.circuit_breaker = CircuitBreaker()
    
    async def execute(self, operation_type: str, **kwargs):
        """Execute with circuit breaker protection."""
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker open, using fallback")
            return await self._execute_fallback_operation(
                WrapperContext(operation_id="", operation_type=operation_type),
                original_error=Exception("Circuit breaker open"),
                **kwargs
            )
        
        try:
            result = await super().execute(operation_type, **kwargs)
            if result.success:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
```

## Deployment and Operations

### Docker Configuration

Create a Dockerfile for containerized deployment:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY examples/ examples/
COPY docs/ docs/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV WRAPPER_ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.my_wrapper.wrapper import WeatherAPIWrapper; print('OK')" || exit 1

# Run application
CMD ["python", "-m", "src.my_wrapper"]
```

### Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  weather-wrapper:
    build: .
    environment:
      - WEATHER_API_KEY=${WEATHER_API_KEY}
      - WRAPPER_LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    ports:
      - "8080:8080"
    volumes:
      - ./logs:/app/logs
    networks:
      - wrapper-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - wrapper-network

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - wrapper-network

networks:
  wrapper-network:
    driver: bridge
```

### Monitoring and Observability

```python
# monitoring.py - Add comprehensive monitoring
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Metrics
request_counter = Counter('wrapper_requests_total', 'Total requests', ['wrapper', 'operation', 'status'])
request_duration = Histogram('wrapper_request_duration_seconds', 'Request duration', ['wrapper', 'operation'])
cache_hits = Counter('wrapper_cache_hits_total', 'Cache hits', ['wrapper'])
error_counter = Counter('wrapper_errors_total', 'Total errors', ['wrapper', 'error_type'])

class MonitoredWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.logger = structlog.get_logger(wrapper=self.__class__.__name__)
    
    async def execute(self, operation_type: str, **kwargs):
        """Execute with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            result = await super().execute(operation_type, **kwargs)
            
            # Record metrics
            status = "success" if result.success else "error"
            request_counter.labels(
                wrapper=self.__class__.__name__,
                operation=operation_type,
                status=status
            ).inc()
            
            request_duration.labels(
                wrapper=self.__class__.__name__,
                operation=operation_type
            ).observe(time.time() - start_time)
            
            if result.data and hasattr(result.data, 'cache_hit') and result.data.cache_hit:
                cache_hits.labels(wrapper=self.__class__.__name__).inc()
            
            # Structured logging
            self.logger.info(
                "wrapper_request_completed",
                operation=operation_type,
                success=result.success,
                duration=time.time() - start_time,
                fallback_used=result.fallback_used
            )
            
            return result
            
        except Exception as e:
            error_counter.labels(
                wrapper=self.__class__.__name__,
                error_type=type(e).__name__
            ).inc()
            
            self.logger.error(
                "wrapper_request_failed",
                operation=operation_type,
                error=str(e),
                duration=time.time() - start_time
            )
            raise

# Start metrics server
def start_monitoring():
    start_http_server(8000)
    logger.info("Monitoring server started on port 8000")
```

### Configuration Management

```python
# config_management.py - Environment-aware configuration
import os
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    environment: str = field(default_factory=lambda: os.getenv("WRAPPER_ENVIRONMENT", "development"))
    
    def get_config_overrides(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides."""
        if self.environment == "production":
            return {
                "timeout_seconds": 45.0,
                "max_retries": 5,
                "cache_duration_minutes": 30,
                "rate_limit_per_minute": 10000,
                "enable_monitoring": True,
                "log_level": "INFO"
            }
        elif self.environment == "staging":
            return {
                "timeout_seconds": 30.0,
                "max_retries": 3,
                "cache_duration_minutes": 15,
                "rate_limit_per_minute": 5000,
                "enable_monitoring": True,
                "log_level": "DEBUG"
            }
        else:  # development
            return {
                "timeout_seconds": 10.0,
                "max_retries": 1,
                "cache_duration_minutes": 5,
                "rate_limit_per_minute": 100,
                "enable_monitoring": False,
                "log_level": "DEBUG"
            }

class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self):
        self.env_config = EnvironmentConfig()
    
    def create_wrapper_config(self, base_config_class, **overrides):
        """Create wrapper configuration with environment overrides."""
        # Start with base configuration
        config = base_config_class()
        
        # Apply environment-specific overrides
        env_overrides = self.env_config.get_config_overrides()
        for key, value in env_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Apply explicit overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Load from environment variables
        for field_name in dir(config):
            if not field_name.startswith("_"):
                env_var = f"{config.__class__.__name__.upper()}_{field_name.upper()}"
                env_value = os.getenv(env_var)
                if env_value:
                    # Convert string to appropriate type
                    current_value = getattr(config, field_name)
                    if isinstance(current_value, bool):
                        setattr(config, field_name, env_value.lower() in ("true", "1", "yes"))
                    elif isinstance(current_value, int):
                        setattr(config, field_name, int(env_value))
                    elif isinstance(current_value, float):
                        setattr(config, field_name, float(env_value))
                    else:
                        setattr(config, field_name, env_value)
        
        return config
```

## Best Practices

### 1. Configuration Design

- **Use dataclasses** with type hints for configuration
- **Provide sensible defaults** for all optional fields
- **Implement validation** with clear error messages
- **Support environment variables** for deployment flexibility
- **Mark sensitive fields** to prevent logging

### 2. Error Handling

- **Create specific exceptions** for different error types
- **Implement comprehensive fallback** strategies
- **Log errors with context** for debugging
- **Use circuit breakers** for external service reliability
- **Provide meaningful error messages** to users

### 3. Performance

- **Use connection pooling** for HTTP clients
- **Implement intelligent caching** with TTL and invalidation
- **Add rate limiting** to respect API limits
- **Monitor performance metrics** and set up alerts
- **Use async/await consistently** throughout

### 4. Testing

- **Write comprehensive unit tests** with high coverage
- **Create integration tests** for real-world scenarios
- **Use mocking strategically** for external dependencies
- **Test error conditions** and edge cases
- **Benchmark performance** characteristics

### 5. Monitoring

- **Emit structured logs** with consistent format
- **Track key metrics** (latency, errors, cache hits)
- **Set up health checks** for deployment environments
- **Monitor resource usage** (memory, connections)
- **Create dashboards** for operational visibility

### 6. Security

- **Never log sensitive data** (API keys, tokens)
- **Validate all inputs** to prevent injection attacks
- **Use HTTPS** for all external communications
- **Implement proper authentication** handling
- **Regular security updates** for dependencies

This comprehensive guide provides everything needed to develop professional-grade wrapper integrations. Follow these patterns and practices to create robust, scalable, and maintainable wrapper implementations.