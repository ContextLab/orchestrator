"""Direct import tests for model module to achieve coverage measurement."""

import pytest

# Direct import to ensure coverage measurement
from src.orchestrator.core.model import (
    ModelCapabilities,
    ModelRequirements,
    ModelMetrics,
    Model,
    MockModel
)


class TestModelCapabilities:
    """Test ModelCapabilities class."""
    
    def test_model_capabilities_creation_minimal(self):
        """Test minimal ModelCapabilities creation."""
        caps = ModelCapabilities(supported_tasks=["text_generation"])
        
        assert caps.supported_tasks == ["text_generation"]
        assert caps.context_window == 4096
        assert caps.supports_function_calling is False
        assert caps.supports_structured_output is False
        assert caps.supports_streaming is False
        assert caps.languages == ["en"]
        assert caps.max_tokens is None
        assert caps.temperature_range == (0.0, 2.0)
    
    def test_model_capabilities_creation_full(self):
        """Test full ModelCapabilities creation."""
        caps = ModelCapabilities(
            supported_tasks=["text_generation", "summarization"],
            context_window=8192,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_streaming=True,
            languages=["en", "es", "fr"],
            max_tokens=4096,
            temperature_range=(0.1, 1.5)
        )
        
        assert caps.supported_tasks == ["text_generation", "summarization"]
        assert caps.context_window == 8192
        assert caps.supports_function_calling is True
        assert caps.supports_structured_output is True
        assert caps.supports_streaming is True
        assert caps.languages == ["en", "es", "fr"]
        assert caps.max_tokens == 4096
        assert caps.temperature_range == (0.1, 1.5)
    
    def test_model_capabilities_validation_context_window(self):
        """Test validation of context window."""
        with pytest.raises(ValueError, match="Context window must be positive"):
            ModelCapabilities(supported_tasks=["test"], context_window=0)
        
        with pytest.raises(ValueError, match="Context window must be positive"):
            ModelCapabilities(supported_tasks=["test"], context_window=-1)
    
    def test_model_capabilities_validation_tasks(self):
        """Test validation of supported tasks."""
        with pytest.raises(ValueError, match="Model must support at least one task"):
            ModelCapabilities(supported_tasks=[])
    
    def test_model_capabilities_validation_languages(self):
        """Test validation of languages."""
        with pytest.raises(ValueError, match="Model must support at least one language"):
            ModelCapabilities(supported_tasks=["test"], languages=[])
    
    def test_model_capabilities_validation_temperature_range(self):
        """Test validation of temperature range."""
        # Wrong number of values
        with pytest.raises(ValueError, match="Temperature range must be a tuple of two values"):
            ModelCapabilities(supported_tasks=["test"], temperature_range=(0.1,))
        
        with pytest.raises(ValueError, match="Temperature range must be a tuple of two values"):
            ModelCapabilities(supported_tasks=["test"], temperature_range=(0.1, 0.5, 1.0))
        
        # Min > Max
        with pytest.raises(ValueError, match="Temperature range min must be <= max"):
            ModelCapabilities(supported_tasks=["test"], temperature_range=(1.0, 0.5))
    
    def test_model_capabilities_supports_task(self):
        """Test task support checking."""
        caps = ModelCapabilities(supported_tasks=["text_generation", "summarization"])
        
        assert caps.supports_task("text_generation") is True
        assert caps.supports_task("summarization") is True
        assert caps.supports_task("translation") is False
    
    def test_model_capabilities_supports_language(self):
        """Test language support checking."""
        caps = ModelCapabilities(supported_tasks=["test"], languages=["en", "es"])
        
        assert caps.supports_language("en") is True
        assert caps.supports_language("es") is True
        assert caps.supports_language("fr") is False
    
    def test_model_capabilities_to_dict(self):
        """Test conversion to dictionary."""
        caps = ModelCapabilities(
            supported_tasks=["text_generation"],
            context_window=8192,
            supports_function_calling=True,
            languages=["en", "es"]
        )
        
        caps_dict = caps.to_dict()
        
        assert caps_dict["supported_tasks"] == ["text_generation"]
        assert caps_dict["context_window"] == 8192
        assert caps_dict["supports_function_calling"] is True
        assert caps_dict["supports_structured_output"] is False
        assert caps_dict["supports_streaming"] is False
        assert caps_dict["languages"] == ["en", "es"]
        assert caps_dict["max_tokens"] is None
        assert caps_dict["temperature_range"] == (0.0, 2.0)
    
    def test_model_capabilities_from_dict(self):
        """Test creation from dictionary."""
        caps_dict = {
            "supported_tasks": ["text_generation", "chat"],
            "context_window": 16384,
            "supports_function_calling": True,
            "supports_structured_output": True,
            "supports_streaming": False,
            "languages": ["en", "fr", "de"],
            "max_tokens": 8192,
            "temperature_range": (0.2, 1.8)
        }
        
        caps = ModelCapabilities.from_dict(caps_dict)
        
        assert caps.supported_tasks == ["text_generation", "chat"]
        assert caps.context_window == 16384
        assert caps.supports_function_calling is True
        assert caps.supports_structured_output is True
        assert caps.supports_streaming is False
        assert caps.languages == ["en", "fr", "de"]
        assert caps.max_tokens == 8192
        assert caps.temperature_range == (0.2, 1.8)


class TestModelRequirements:
    """Test ModelRequirements class."""
    
    def test_model_requirements_creation_minimal(self):
        """Test minimal ModelRequirements creation."""
        req = ModelRequirements()
        
        assert req.memory_gb == 1.0
        assert req.gpu_memory_gb is None
        assert req.cpu_cores == 1
        assert req.supports_quantization == []
        assert req.min_python_version == "3.8"
        assert req.requires_gpu is False
        assert req.disk_space_gb == 1.0
    
    def test_model_requirements_creation_full(self):
        """Test full ModelRequirements creation."""
        req = ModelRequirements(
            memory_gb=8.0,
            gpu_memory_gb=16.0,
            cpu_cores=4,
            supports_quantization=["int8", "fp16"],
            min_python_version="3.9",
            requires_gpu=True,
            disk_space_gb=10.0
        )
        
        assert req.memory_gb == 8.0
        assert req.gpu_memory_gb == 16.0
        assert req.cpu_cores == 4
        assert req.supports_quantization == ["int8", "fp16"]
        assert req.min_python_version == "3.9"
        assert req.requires_gpu is True
        assert req.disk_space_gb == 10.0
    
    def test_model_requirements_validation_memory(self):
        """Test memory validation."""
        with pytest.raises(ValueError, match="Memory requirement must be positive"):
            ModelRequirements(memory_gb=0)
        
        with pytest.raises(ValueError, match="Memory requirement must be positive"):
            ModelRequirements(memory_gb=-1)
    
    def test_model_requirements_validation_gpu_memory(self):
        """Test GPU memory validation."""
        with pytest.raises(ValueError, match="GPU memory requirement must be positive"):
            ModelRequirements(gpu_memory_gb=0)
        
        with pytest.raises(ValueError, match="GPU memory requirement must be positive"):
            ModelRequirements(gpu_memory_gb=-1)
    
    def test_model_requirements_validation_cpu_cores(self):
        """Test CPU cores validation."""
        with pytest.raises(ValueError, match="CPU cores requirement must be positive"):
            ModelRequirements(cpu_cores=0)
        
        with pytest.raises(ValueError, match="CPU cores requirement must be positive"):
            ModelRequirements(cpu_cores=-1)
    
    def test_model_requirements_validation_disk_space(self):
        """Test disk space validation."""
        with pytest.raises(ValueError, match="Disk space requirement must be positive"):
            ModelRequirements(disk_space_gb=0)
        
        with pytest.raises(ValueError, match="Disk space requirement must be positive"):
            ModelRequirements(disk_space_gb=-1)
    
    def test_model_requirements_to_dict(self):
        """Test conversion to dictionary."""
        req = ModelRequirements(
            memory_gb=4.0,
            gpu_memory_gb=8.0,
            cpu_cores=2,
            supports_quantization=["fp16"],
            requires_gpu=True
        )
        
        req_dict = req.to_dict()
        
        assert req_dict["memory_gb"] == 4.0
        assert req_dict["gpu_memory_gb"] == 8.0
        assert req_dict["cpu_cores"] == 2
        assert req_dict["supports_quantization"] == ["fp16"]
        assert req_dict["min_python_version"] == "3.8"
        assert req_dict["requires_gpu"] is True
        assert req_dict["disk_space_gb"] == 1.0
    
    def test_model_requirements_from_dict(self):
        """Test creation from dictionary."""
        req_dict = {
            "memory_gb": 6.0,
            "gpu_memory_gb": 12.0,
            "cpu_cores": 3,
            "supports_quantization": ["int8", "fp16"],
            "min_python_version": "3.10",
            "requires_gpu": True,
            "disk_space_gb": 5.0
        }
        
        req = ModelRequirements.from_dict(req_dict)
        
        assert req.memory_gb == 6.0
        assert req.gpu_memory_gb == 12.0
        assert req.cpu_cores == 3
        assert req.supports_quantization == ["int8", "fp16"]
        assert req.min_python_version == "3.10"
        assert req.requires_gpu is True
        assert req.disk_space_gb == 5.0


class TestModelMetrics:
    """Test ModelMetrics class."""
    
    def test_model_metrics_creation_minimal(self):
        """Test minimal ModelMetrics creation."""
        metrics = ModelMetrics()
        
        assert metrics.latency_p50 == 0.0
        assert metrics.latency_p95 == 0.0
        assert metrics.throughput == 0.0
        assert metrics.accuracy == 0.0
        assert metrics.cost_per_token == 0.0
        assert metrics.success_rate == 1.0
    
    def test_model_metrics_creation_full(self):
        """Test full ModelMetrics creation."""
        metrics = ModelMetrics(
            latency_p50=100.0,
            latency_p95=200.0,
            throughput=10.5,
            accuracy=0.95,
            cost_per_token=0.001,
            success_rate=0.99
        )
        
        assert metrics.latency_p50 == 100.0
        assert metrics.latency_p95 == 200.0
        assert metrics.throughput == 10.5
        assert metrics.accuracy == 0.95
        assert metrics.cost_per_token == 0.001
        assert metrics.success_rate == 0.99
    
    def test_model_metrics_validation_latency(self):
        """Test latency validation."""
        with pytest.raises(ValueError, match="Latency P50 must be non-negative"):
            ModelMetrics(latency_p50=-1)
        
        with pytest.raises(ValueError, match="Latency P95 must be non-negative"):
            ModelMetrics(latency_p95=-1)
    
    def test_model_metrics_validation_throughput(self):
        """Test throughput validation."""
        with pytest.raises(ValueError, match="Throughput must be non-negative"):
            ModelMetrics(throughput=-1)
    
    def test_model_metrics_validation_accuracy(self):
        """Test accuracy validation."""
        with pytest.raises(ValueError, match="Accuracy must be between 0 and 1"):
            ModelMetrics(accuracy=-0.1)
        
        with pytest.raises(ValueError, match="Accuracy must be between 0 and 1"):
            ModelMetrics(accuracy=1.1)
    
    def test_model_metrics_validation_cost(self):
        """Test cost validation."""
        with pytest.raises(ValueError, match="Cost per token must be non-negative"):
            ModelMetrics(cost_per_token=-0.001)
    
    def test_model_metrics_validation_success_rate(self):
        """Test success rate validation."""
        with pytest.raises(ValueError, match="Success rate must be between 0 and 1"):
            ModelMetrics(success_rate=-0.1)
        
        with pytest.raises(ValueError, match="Success rate must be between 0 and 1"):
            ModelMetrics(success_rate=1.1)
    
    def test_model_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ModelMetrics(
            latency_p50=50.0,
            latency_p95=100.0,
            throughput=5.0,
            accuracy=0.9,
            cost_per_token=0.002,
            success_rate=0.98
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["latency_p50"] == 50.0
        assert metrics_dict["latency_p95"] == 100.0
        assert metrics_dict["throughput"] == 5.0
        assert metrics_dict["accuracy"] == 0.9
        assert metrics_dict["cost_per_token"] == 0.002
        assert metrics_dict["success_rate"] == 0.98
    
    def test_model_metrics_from_dict(self):
        """Test creation from dictionary."""
        metrics_dict = {
            "latency_p50": 75.0,
            "latency_p95": 150.0,
            "throughput": 8.0,
            "accuracy": 0.85,
            "cost_per_token": 0.003,
            "success_rate": 0.97
        }
        
        metrics = ModelMetrics.from_dict(metrics_dict)
        
        assert metrics.latency_p50 == 75.0
        assert metrics.latency_p95 == 150.0
        assert metrics.throughput == 8.0
        assert metrics.accuracy == 0.85
        assert metrics.cost_per_token == 0.003
        assert metrics.success_rate == 0.97


class TestModel:
    """Test Model abstract class."""
    
    def test_model_abstract(self):
        """Test that Model is abstract."""
        with pytest.raises(TypeError):
            Model("test", "test")  # Cannot instantiate abstract class
    
    def test_model_validation_empty_name(self):
        """Test validation with empty name."""
        class ConcreteModel(Model):
            async def generate(self, prompt, temperature=0.7, max_tokens=None, **kwargs):
                return "test"
            async def generate_structured(self, prompt, schema, temperature=0.7, **kwargs):
                return {}
            async def health_check(self):
                return True
            async def estimate_cost(self, prompt, max_tokens=None):
                return 0.0
        
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ConcreteModel("", "test-provider")
    
    def test_model_validation_empty_provider(self):
        """Test validation with empty provider."""
        class ConcreteModel(Model):
            async def generate(self, prompt, temperature=0.7, max_tokens=None, **kwargs):
                return "test"
            async def generate_structured(self, prompt, schema, temperature=0.7, **kwargs):
                return {}
            async def health_check(self):
                return True
            async def estimate_cost(self, prompt, max_tokens=None):
                return 0.0
        
        with pytest.raises(ValueError, match="Provider name cannot be empty"):
            ConcreteModel("test-model", "")


class TestMockModel:
    """Test MockModel implementation."""
    
    def test_mock_model_creation_minimal(self):
        """Test minimal MockModel creation."""
        model = MockModel()
        
        assert model.name == "mock-model"
        assert model.provider == "mock"
        assert model.is_available is True
        assert model.capabilities.supports_function_calling is True
        assert model.capabilities.supports_structured_output is True
    
    def test_mock_model_creation_full(self):
        """Test full MockModel creation."""
        capabilities = ModelCapabilities(
            supported_tasks=["custom_task"],
            context_window=8192
        )
        
        model = MockModel(
            name="custom-mock",
            provider="custom-provider",
            capabilities=capabilities
        )
        
        assert model.name == "custom-mock"
        assert model.provider == "custom-provider"
        assert model.capabilities.context_window == 8192
        assert model.capabilities.supported_tasks == ["custom_task"]
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_default(self):
        """Test default generation without canned responses."""
        model = MockModel()
        
        result = await model.generate("Test prompt")
        
        assert "Mock response for: Test prompt" in result
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_canned_response(self):
        """Test generation with canned response."""
        model = MockModel()
        model.set_response("specific prompt", "specific response")
        
        result = await model.generate("specific prompt")
        
        assert result == "specific response"
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_partial_match(self):
        """Test generation with partial match."""
        model = MockModel()
        model.set_response("keyword", "partial match response")
        
        result = await model.generate("This contains the keyword in it")
        
        assert result == "partial match response"
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_exception(self):
        """Test generation with exception response."""
        model = MockModel()
        model.set_response("error prompt", Exception("Test error"))
        
        with pytest.raises(Exception, match="Test error"):
            await model.generate("error prompt")
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_non_string_response(self):
        """Test generation with non-string response."""
        model = MockModel()
        model.set_response("number prompt", 42)
        
        result = await model.generate("number prompt")
        
        assert result == "42"
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_structured_canned(self):
        """Test structured generation with canned response."""
        model = MockModel()
        expected_response = {"result": "structured data", "score": 0.95}
        model.set_response("structured prompt", expected_response)
        
        result = await model.generate_structured(
            "structured prompt", 
            {"type": "object", "properties": {"result": {"type": "string"}}}
        )
        
        assert result == expected_response
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_structured_default(self):
        """Test structured generation without canned response."""
        model = MockModel()
        
        result = await model.generate_structured(
            "Test structured prompt",
            {"type": "object", "properties": {"result": {"type": "string"}}}
        )
        
        assert "result" in result
        assert "Mock structured response for: Test structured prompt" in result["result"]
    
    @pytest.mark.asyncio
    async def test_mock_model_health_check(self):
        """Test health check."""
        model = MockModel()
        
        is_healthy = await model.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_mock_model_estimate_cost(self):
        """Test cost estimation."""
        model = MockModel()
        
        cost = await model.estimate_cost("Test prompt")
        
        assert cost == 0.001
    
    def test_mock_model_can_handle_task(self):
        """Test task handling check."""
        model = MockModel()
        
        assert model.can_handle_task("generate") is True
        assert model.can_handle_task("analyze") is True
        assert model.can_handle_task("transform") is True
        assert model.can_handle_task("unknown") is False
    
    def test_mock_model_can_handle_language(self):
        """Test language handling check."""
        model = MockModel()
        
        assert model.can_handle_language("en") is True
        assert model.can_handle_language("unknown") is False
    
    def test_mock_model_meets_requirements_context_window(self):
        """Test context window requirement checking."""
        model = MockModel()
        
        # Should meet smaller requirements
        assert model.meets_requirements({"context_window": 2048}) is True
        
        # Should not meet larger requirements
        assert model.meets_requirements({"context_window": 8192}) is False
    
    def test_mock_model_meets_requirements_function_calling(self):
        """Test function calling requirement checking."""
        model = MockModel()
        
        # Should meet function calling requirement
        assert model.meets_requirements({"supports_function_calling": True}) is True
        
        # Should always meet False requirement
        assert model.meets_requirements({"supports_function_calling": False}) is True
    
    def test_mock_model_meets_requirements_structured_output(self):
        """Test structured output requirement checking."""
        model = MockModel()
        
        # Should meet structured output requirement
        assert model.meets_requirements({"supports_structured_output": True}) is True
        
        # Should always meet False requirement
        assert model.meets_requirements({"supports_structured_output": False}) is True
    
    def test_mock_model_meets_requirements_tasks(self):
        """Test task requirements checking."""
        model = MockModel()
        
        # Should meet supported tasks
        assert model.meets_requirements({"tasks": ["generate", "analyze"]}) is True
        
        # Should not meet unsupported tasks
        assert model.meets_requirements({"tasks": ["generate", "unsupported"]}) is False
    
    def test_mock_model_meets_requirements_languages(self):
        """Test language requirements checking."""
        model = MockModel()
        
        # Should meet supported languages
        assert model.meets_requirements({"languages": ["en"]}) is True
        
        # Should not meet unsupported languages
        assert model.meets_requirements({"languages": ["en", "unsupported"]}) is False
    
    def test_mock_model_to_dict(self):
        """Test conversion to dictionary."""
        model = MockModel(name="test-model", provider="test-provider")
        
        model_dict = model.to_dict()
        
        assert model_dict["name"] == "test-model"
        assert model_dict["provider"] == "test-provider"
        assert model_dict["is_available"] is True
        assert "capabilities" in model_dict
        assert "requirements" in model_dict
        assert "metrics" in model_dict
    
    def test_mock_model_repr(self):
        """Test string representation."""
        model = MockModel(name="test-model", provider="test-provider")
        
        repr_str = repr(model)
        
        assert "Model(" in repr_str
        assert "name='test-model'" in repr_str
        assert "provider='test-provider'" in repr_str
    
    def test_mock_model_equality(self):
        """Test equality comparison."""
        model1 = MockModel(name="same-model", provider="same-provider")
        model2 = MockModel(name="same-model", provider="same-provider")
        model3 = MockModel(name="different-model", provider="same-provider")
        model4 = MockModel(name="same-model", provider="different-provider")
        
        # Same name and provider should be equal
        assert model1 == model2
        
        # Different name should not be equal
        assert model1 != model3
        
        # Different provider should not be equal
        assert model1 != model4
        
        # Should not be equal to non-Model objects
        assert model1 != "not a model"
        assert model1 != 123
    
    def test_mock_model_hash(self):
        """Test hash function."""
        model1 = MockModel(name="test-model", provider="test-provider")
        model2 = MockModel(name="test-model", provider="test-provider")
        model3 = MockModel(name="different-model", provider="test-provider")
        
        # Same name and provider should have same hash
        assert hash(model1) == hash(model2)
        
        # Different name/provider should have different hash
        assert hash(model1) != hash(model3)
        
        # Should be usable in sets
        model_set = {model1, model2, model3}
        assert len(model_set) == 2  # model1 and model2 are considered same