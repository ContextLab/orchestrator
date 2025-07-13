"""Tests for Model classes."""

import pytest

from orchestrator.core.model import (
    Model,
    ModelCapabilities,
    ModelRequirements,
    ModelMetrics,
    MockModel,
)


class TestModelCapabilities:
    """Test cases for ModelCapabilities class."""
    
    def test_capabilities_creation(self):
        """Test basic capabilities creation."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate", "analyze"],
            context_window=8192,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_streaming=True,
            languages=["en", "es", "fr"],
            max_tokens=4096,
            temperature_range=(0.0, 1.0),
        )
        
        assert capabilities.supported_tasks == ["generate", "analyze"]
        assert capabilities.context_window == 8192
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_structured_output is True
        assert capabilities.supports_streaming is True
        assert capabilities.languages == ["en", "es", "fr"]
        assert capabilities.max_tokens == 4096
        assert capabilities.temperature_range == (0.0, 1.0)
    
    def test_capabilities_defaults(self):
        """Test default capabilities values."""
        capabilities = ModelCapabilities(supported_tasks=["generate"])
        
        assert capabilities.supported_tasks == ["generate"]
        assert capabilities.context_window == 4096
        assert capabilities.supports_function_calling is False
        assert capabilities.supports_structured_output is False
        assert capabilities.supports_streaming is False
        assert capabilities.languages == ["en"]
        assert capabilities.max_tokens is None
        assert capabilities.temperature_range == (0.0, 2.0)
    
    def test_capabilities_validation_context_window(self):
        """Test validation of context window."""
        with pytest.raises(ValueError, match="Context window must be positive"):
            ModelCapabilities(context_window=0)
        
        with pytest.raises(ValueError, match="Context window must be positive"):
            ModelCapabilities(context_window=-1)
    
    def test_capabilities_validation_empty_tasks(self):
        """Test validation of empty supported tasks."""
        with pytest.raises(ValueError, match="must support at least one task"):
            ModelCapabilities(supported_tasks=[])
    
    def test_capabilities_validation_empty_languages(self):
        """Test validation of empty languages."""
        with pytest.raises(ValueError, match="must support at least one language"):
            ModelCapabilities(supported_tasks=["generate"], languages=[])
    
    def test_capabilities_validation_temperature_range(self):
        """Test validation of temperature range."""
        with pytest.raises(ValueError, match="must be a tuple of two values"):
            ModelCapabilities(supported_tasks=["generate"], temperature_range=(0.0,))
        
        with pytest.raises(ValueError, match="Temperature range min must be"):
            ModelCapabilities(supported_tasks=["generate"], temperature_range=(1.0, 0.0))
    
    def test_supports_task(self):
        """Test supports_task method."""
        capabilities = ModelCapabilities(supported_tasks=["generate", "analyze"])
        
        assert capabilities.supports_task("generate")
        assert capabilities.supports_task("analyze")
        assert not capabilities.supports_task("translate")
    
    def test_supports_language(self):
        """Test supports_language method."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            languages=["en", "es", "fr"]
        )
        
        assert capabilities.supports_language("en")
        assert capabilities.supports_language("es")
        assert capabilities.supports_language("fr")
        assert not capabilities.supports_language("de")
    
    def test_to_dict(self):
        """Test to_dict method."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            context_window=8192,
            supports_function_calling=True,
        )
        
        capabilities_dict = capabilities.to_dict()
        
        assert capabilities_dict["supported_tasks"] == ["generate"]
        assert capabilities_dict["context_window"] == 8192
        assert capabilities_dict["supports_function_calling"] is True
    
    def test_from_dict(self):
        """Test from_dict method."""
        capabilities_dict = {
            "supported_tasks": ["generate", "analyze"],
            "context_window": 8192,
            "supports_function_calling": True,
            "supports_structured_output": False,
            "supports_streaming": True,
            "languages": ["en", "es"],
            "max_tokens": 4096,
            "temperature_range": (0.0, 1.0),
        }
        
        capabilities = ModelCapabilities.from_dict(capabilities_dict)
        
        assert capabilities.supported_tasks == ["generate", "analyze"]
        assert capabilities.context_window == 8192
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_structured_output is False
        assert capabilities.supports_streaming is True
        assert capabilities.languages == ["en", "es"]
        assert capabilities.max_tokens == 4096
        assert capabilities.temperature_range == (0.0, 1.0)


class TestModelRequirements:
    """Test cases for ModelRequirements class."""
    
    def test_requirements_creation(self):
        """Test basic requirements creation."""
        requirements = ModelRequirements(
            memory_gb=8.0,
            gpu_memory_gb=4.0,
            cpu_cores=4,
            supports_quantization=["int8", "int4"],
            min_python_version="3.9",
            requires_gpu=True,
            disk_space_gb=10.0,
        )
        
        assert requirements.memory_gb == 8.0
        assert requirements.gpu_memory_gb == 4.0
        assert requirements.cpu_cores == 4
        assert requirements.supports_quantization == ["int8", "int4"]
        assert requirements.min_python_version == "3.9"
        assert requirements.requires_gpu is True
        assert requirements.disk_space_gb == 10.0
    
    def test_requirements_defaults(self):
        """Test default requirements values."""
        requirements = ModelRequirements()
        
        assert requirements.memory_gb == 1.0
        assert requirements.gpu_memory_gb is None
        assert requirements.cpu_cores == 1
        assert requirements.supports_quantization == []
        assert requirements.min_python_version == "3.8"
        assert requirements.requires_gpu is False
        assert requirements.disk_space_gb == 1.0
    
    def test_requirements_validation_memory(self):
        """Test validation of memory requirement."""
        with pytest.raises(ValueError, match="Memory requirement must be positive"):
            ModelRequirements(memory_gb=0)
        
        with pytest.raises(ValueError, match="Memory requirement must be positive"):
            ModelRequirements(memory_gb=-1)
    
    def test_requirements_validation_gpu_memory(self):
        """Test validation of GPU memory requirement."""
        with pytest.raises(ValueError, match="GPU memory requirement must be positive"):
            ModelRequirements(gpu_memory_gb=0)
        
        with pytest.raises(ValueError, match="GPU memory requirement must be positive"):
            ModelRequirements(gpu_memory_gb=-1)
    
    def test_requirements_validation_cpu_cores(self):
        """Test validation of CPU cores requirement."""
        with pytest.raises(ValueError, match="CPU cores requirement must be positive"):
            ModelRequirements(cpu_cores=0)
        
        with pytest.raises(ValueError, match="CPU cores requirement must be positive"):
            ModelRequirements(cpu_cores=-1)
    
    def test_requirements_validation_disk_space(self):
        """Test validation of disk space requirement."""
        with pytest.raises(ValueError, match="Disk space requirement must be positive"):
            ModelRequirements(disk_space_gb=0)
        
        with pytest.raises(ValueError, match="Disk space requirement must be positive"):
            ModelRequirements(disk_space_gb=-1)
    
    def test_to_dict(self):
        """Test to_dict method."""
        requirements = ModelRequirements(
            memory_gb=8.0,
            gpu_memory_gb=4.0,
            cpu_cores=4,
        )
        
        requirements_dict = requirements.to_dict()
        
        assert requirements_dict["memory_gb"] == 8.0
        assert requirements_dict["gpu_memory_gb"] == 4.0
        assert requirements_dict["cpu_cores"] == 4
    
    def test_from_dict(self):
        """Test from_dict method."""
        requirements_dict = {
            "memory_gb": 8.0,
            "gpu_memory_gb": 4.0,
            "cpu_cores": 4,
            "supports_quantization": ["int8"],
            "min_python_version": "3.9",
            "requires_gpu": True,
            "disk_space_gb": 10.0,
        }
        
        requirements = ModelRequirements.from_dict(requirements_dict)
        
        assert requirements.memory_gb == 8.0
        assert requirements.gpu_memory_gb == 4.0
        assert requirements.cpu_cores == 4
        assert requirements.supports_quantization == ["int8"]
        assert requirements.min_python_version == "3.9"
        assert requirements.requires_gpu is True
        assert requirements.disk_space_gb == 10.0


class TestModelMetrics:
    """Test cases for ModelMetrics class."""
    
    def test_metrics_creation(self):
        """Test basic metrics creation."""
        metrics = ModelMetrics(
            latency_p50=1.5,
            latency_p95=3.0,
            throughput=10.0,
            accuracy=0.95,
            cost_per_token=0.001,
            success_rate=0.99,
        )
        
        assert metrics.latency_p50 == 1.5
        assert metrics.latency_p95 == 3.0
        assert metrics.throughput == 10.0
        assert metrics.accuracy == 0.95
        assert metrics.cost_per_token == 0.001
        assert metrics.success_rate == 0.99
    
    def test_metrics_defaults(self):
        """Test default metrics values."""
        metrics = ModelMetrics()
        
        assert metrics.latency_p50 == 0.0
        assert metrics.latency_p95 == 0.0
        assert metrics.throughput == 0.0
        assert metrics.accuracy == 0.0
        assert metrics.cost_per_token == 0.0
        assert metrics.success_rate == 1.0
    
    def test_metrics_validation_latency(self):
        """Test validation of latency metrics."""
        with pytest.raises(ValueError, match="Latency P50 must be non-negative"):
            ModelMetrics(latency_p50=-1)
        
        with pytest.raises(ValueError, match="Latency P95 must be non-negative"):
            ModelMetrics(latency_p95=-1)
    
    def test_metrics_validation_throughput(self):
        """Test validation of throughput metric."""
        with pytest.raises(ValueError, match="Throughput must be non-negative"):
            ModelMetrics(throughput=-1)
    
    def test_metrics_validation_accuracy(self):
        """Test validation of accuracy metric."""
        with pytest.raises(ValueError, match="Accuracy must be between 0 and 1"):
            ModelMetrics(accuracy=-0.1)
        
        with pytest.raises(ValueError, match="Accuracy must be between 0 and 1"):
            ModelMetrics(accuracy=1.1)
    
    def test_metrics_validation_cost(self):
        """Test validation of cost metric."""
        with pytest.raises(ValueError, match="Cost per token must be non-negative"):
            ModelMetrics(cost_per_token=-0.001)
    
    def test_metrics_validation_success_rate(self):
        """Test validation of success rate metric."""
        with pytest.raises(ValueError, match="Success rate must be between 0 and 1"):
            ModelMetrics(success_rate=-0.1)
        
        with pytest.raises(ValueError, match="Success rate must be between 0 and 1"):
            ModelMetrics(success_rate=1.1)
    
    def test_to_dict(self):
        """Test to_dict method."""
        metrics = ModelMetrics(
            latency_p50=1.5,
            accuracy=0.95,
            cost_per_token=0.001,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["latency_p50"] == 1.5
        assert metrics_dict["accuracy"] == 0.95
        assert metrics_dict["cost_per_token"] == 0.001
    
    def test_from_dict(self):
        """Test from_dict method."""
        metrics_dict = {
            "latency_p50": 1.5,
            "latency_p95": 3.0,
            "throughput": 10.0,
            "accuracy": 0.95,
            "cost_per_token": 0.001,
            "success_rate": 0.99,
        }
        
        metrics = ModelMetrics.from_dict(metrics_dict)
        
        assert metrics.latency_p50 == 1.5
        assert metrics.latency_p95 == 3.0
        assert metrics.throughput == 10.0
        assert metrics.accuracy == 0.95
        assert metrics.cost_per_token == 0.001
        assert metrics.success_rate == 0.99


class TestModel:
    """Test cases for Model abstract class."""
    
    def test_model_creation(self):
        """Test basic model creation."""
        capabilities = ModelCapabilities(supported_tasks=["generate"])
        requirements = ModelRequirements(memory_gb=2.0)
        metrics = ModelMetrics(accuracy=0.9)
        
        model = MockModel(
            name="test-model",
            provider="test-provider",
            capabilities=capabilities,
            requirements=requirements,
            metrics=metrics,
        )
        
        assert model.name == "test-model"
        assert model.provider == "test-provider"
        assert model.capabilities == capabilities
        assert model.requirements == requirements
        assert model.metrics == metrics
    
    def test_model_validation_empty_name(self):
        """Test model validation with empty name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            MockModel(name="", provider="test")
    
    def test_model_validation_empty_provider(self):
        """Test model validation with empty provider."""
        with pytest.raises(ValueError, match="Provider name cannot be empty"):
            MockModel(name="test", provider="")
    
    def test_model_defaults(self):
        """Test model default values."""
        model = MockModel(name="test", provider="test")
        
        assert isinstance(model.capabilities, ModelCapabilities)
        assert isinstance(model.requirements, ModelRequirements)
        assert isinstance(model.metrics, ModelMetrics)
        assert model.is_available is True  # MockModel sets this to True
    
    def test_can_handle_task(self):
        """Test can_handle_task method."""
        capabilities = ModelCapabilities(supported_tasks=["generate", "analyze"])
        model = MockModel(name="test", provider="test", capabilities=capabilities)
        
        assert model.can_handle_task("generate")
        assert model.can_handle_task("analyze")
        assert not model.can_handle_task("translate")
    
    def test_can_handle_language(self):
        """Test can_handle_language method."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            languages=["en", "es", "fr"]
        )
        model = MockModel(name="test", provider="test", capabilities=capabilities)
        
        assert model.can_handle_language("en")
        assert model.can_handle_language("es")
        assert model.can_handle_language("fr")
        assert not model.can_handle_language("de")
    
    def test_meets_requirements_context_window(self):
        """Test meets_requirements for context window."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            context_window=8192
        )
        model = MockModel(name="test", provider="test", capabilities=capabilities)
        
        assert model.meets_requirements({"context_window": 4096})
        assert model.meets_requirements({"context_window": 8192})
        assert not model.meets_requirements({"context_window": 16384})
    
    def test_meets_requirements_function_calling(self):
        """Test meets_requirements for function calling."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=True
        )
        model = MockModel(name="test", provider="test", capabilities=capabilities)
        
        assert model.meets_requirements({"supports_function_calling": True})
        assert model.meets_requirements({"supports_function_calling": False})
        
        # Test with model that doesn't support function calling
        capabilities_no_func = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=False
        )
        model_no_func = MockModel(name="test", provider="test", capabilities=capabilities_no_func)
        
        assert not model_no_func.meets_requirements({"supports_function_calling": True})
        assert model_no_func.meets_requirements({"supports_function_calling": False})
    
    def test_meets_requirements_structured_output(self):
        """Test meets_requirements for structured output."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            supports_structured_output=True
        )
        model = MockModel(name="test", provider="test", capabilities=capabilities)
        
        assert model.meets_requirements({"supports_structured_output": True})
        assert model.meets_requirements({"supports_structured_output": False})
        
        # Test with model that doesn't support structured output
        capabilities_no_struct = ModelCapabilities(
            supported_tasks=["generate"],
            supports_structured_output=False
        )
        model_no_struct = MockModel(name="test", provider="test", capabilities=capabilities_no_struct)
        
        assert not model_no_struct.meets_requirements({"supports_structured_output": True})
        assert model_no_struct.meets_requirements({"supports_structured_output": False})
    
    def test_meets_requirements_tasks(self):
        """Test meets_requirements for tasks."""
        capabilities = ModelCapabilities(supported_tasks=["generate", "analyze"])
        model = MockModel(name="test", provider="test", capabilities=capabilities)
        
        assert model.meets_requirements({"tasks": ["generate"]})
        assert model.meets_requirements({"tasks": ["analyze"]})
        assert model.meets_requirements({"tasks": ["generate", "analyze"]})
        assert not model.meets_requirements({"tasks": ["translate"]})
        assert not model.meets_requirements({"tasks": ["generate", "translate"]})
    
    def test_meets_requirements_languages(self):
        """Test meets_requirements for languages."""
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            languages=["en", "es", "fr"]
        )
        model = MockModel(name="test", provider="test", capabilities=capabilities)
        
        assert model.meets_requirements({"languages": ["en"]})
        assert model.meets_requirements({"languages": ["es"]})
        assert model.meets_requirements({"languages": ["en", "es"]})
        assert not model.meets_requirements({"languages": ["de"]})
        assert not model.meets_requirements({"languages": ["en", "de"]})
    
    def test_to_dict(self):
        """Test to_dict method."""
        model = MockModel(name="test-model", provider="test-provider")
        
        model_dict = model.to_dict()
        
        assert model_dict["name"] == "test-model"
        assert model_dict["provider"] == "test-provider"
        assert "capabilities" in model_dict
        assert "requirements" in model_dict
        assert "metrics" in model_dict
        assert "is_available" in model_dict
    
    def test_repr(self):
        """Test string representation."""
        model = MockModel(name="test-model", provider="test-provider")
        
        repr_str = repr(model)
        
        assert "test-model" in repr_str
        assert "test-provider" in repr_str
    
    def test_equality(self):
        """Test model equality."""
        model1 = MockModel(name="test", provider="provider1")
        model2 = MockModel(name="test", provider="provider1")
        model3 = MockModel(name="test", provider="provider2")
        model4 = MockModel(name="other", provider="provider1")
        
        assert model1 == model2  # Same name and provider
        assert model1 != model3  # Different provider
        assert model1 != model4  # Different name
        assert model1 != "not_a_model"  # Different type
    
    def test_hash(self):
        """Test model hashing."""
        model1 = MockModel(name="test", provider="provider1")
        model2 = MockModel(name="test", provider="provider1")
        model3 = MockModel(name="test", provider="provider2")
        
        assert hash(model1) == hash(model2)  # Same name and provider
        assert hash(model1) != hash(model3)  # Different provider
        
        # Test use in set
        model_set = {model1, model2, model3}
        assert len(model_set) == 2  # model1 and model2 are the same


class TestMockModel:
    """Test cases for MockModel class."""
    
    @pytest.mark.asyncio
    async def test_mock_model_generate(self):
        """Test MockModel generate method."""
        model = MockModel()
        
        result = await model.generate("Test prompt")
        
        assert isinstance(result, str)
        assert "Test prompt" in result
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_with_response(self):
        """Test MockModel generate with canned response."""
        model = MockModel()
        model.set_response("Test prompt", "Canned response")
        
        result = await model.generate("Test prompt")
        
        assert result == "Canned response"
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_with_non_string_response(self):
        """Test MockModel generate with non-string canned response."""
        model = MockModel()
        # Set a non-string response (like an integer or dict)
        model.set_response("Test prompt", 42)
        
        result = await model.generate("Test prompt")
        
        # Should be converted to string
        assert result == "42"
        assert isinstance(result, str)
        
        # Test with complex object
        model.set_response("Complex prompt", {"key": "value", "number": 123})
        
        result = await model.generate("Complex prompt")
        
        # Should be converted to string representation
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_structured(self):
        """Test MockModel generate_structured method."""
        model = MockModel()
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        
        result = await model.generate_structured("Test prompt", schema)
        
        assert isinstance(result, dict)
        assert "result" in result
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_structured_with_response(self):
        """Test MockModel generate_structured with canned response."""
        model = MockModel()
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        canned_response = {"result": "Canned structured response"}
        
        model.set_response("Test prompt", canned_response)
        
        result = await model.generate_structured("Test prompt", schema)
        
        assert result == canned_response
    
    @pytest.mark.asyncio
    async def test_mock_model_health_check(self):
        """Test MockModel health_check method."""
        model = MockModel()
        
        result = await model.health_check()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_mock_model_estimate_cost(self):
        """Test MockModel estimate_cost method."""
        model = MockModel()
        
        result = await model.estimate_cost("Test prompt")
        
        assert isinstance(result, float)
        assert result > 0
    
    def test_mock_model_set_response(self):
        """Test MockModel set_response method."""
        model = MockModel()
        
        model.set_response("prompt1", "response1")
        model.set_response("prompt2", {"key": "value"})
        
        assert model._responses["prompt1"] == "response1"
        assert model._responses["prompt2"] == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_with_exception_response(self):
        """Test MockModel generate with exception response."""
        model = MockModel()
        test_exception = ValueError("Test error")
        model.set_response("Error prompt", test_exception)
        
        with pytest.raises(ValueError, match="Test error"):
            await model.generate("Error prompt")
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_with_partial_match_exception(self):
        """Test MockModel generate with partial match exception."""
        model = MockModel()
        test_exception = RuntimeError("Partial match error")
        model.set_response("error", test_exception)
        
        with pytest.raises(RuntimeError, match="Partial match error"):
            await model.generate("This contains error in the prompt")
    
    @pytest.mark.asyncio
    async def test_mock_model_generate_with_partial_match_non_string(self):
        """Test MockModel generate with partial match non-string response."""
        model = MockModel()
        model.set_response("test", {"result": "partial match object"})
        
        result = await model.generate("This prompt contains test keyword")
        
        # Should convert non-string to string for partial match
        assert isinstance(result, str)
        assert "result" in result