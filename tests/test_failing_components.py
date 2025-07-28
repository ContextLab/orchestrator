"""Components that can be configured to fail for testing."""

from typing import Any, Dict, List

from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.control_system import ControlSystem
from src.orchestrator.core.resource_allocator import (
    ResourceAllocator,
    ResourceType,
    ResourceQuota)


class FailingModelRegistry(ModelRegistry):
    """Model registry that can be configured to fail."""

    def __init__(self):
        super().__init__()
        self.should_fail = False
        self.failure_message = "Registry failed"

    def set_failure_mode(self, should_fail: bool, message: str = "Registry failed"):
        """Configure the registry to fail or succeed."""
        self.should_fail = should_fail
        self.failure_message = message

    async def get_available_models(self) -> List[str]:
        """Get available models - fails if configured to."""
        if self.should_fail:
            raise Exception(self.failure_message)
        return await super().get_available_models()


class FailingControlSystem(ControlSystem):
    """Control system that can be configured to fail."""

    def __init__(self):
        super().__init__("failing-control", {})
        self.should_fail = False
        self.failure_message = "Control failed"

    def set_failure_mode(self, should_fail: bool, message: str = "Control failed"):
        """Configure the control system to fail or succeed."""
        self.should_fail = should_fail
        self.failure_message = message

    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities - fails if configured to."""
        if self.should_fail:
            raise Exception(self.failure_message)
        return super().get_capabilities()

    async def execute_task(self, task: Any, context: Dict[str, Any] = None) -> Any:
        """Execute task - fails if configured to."""
        if self.should_fail:
            raise Exception(self.failure_message)
        return {"status": "completed", "result": "test"}

    async def health_check(self) -> bool:
        """Health check - returns False if configured to fail."""
        return not self.should_fail


class TrackingResourceAllocator(ResourceAllocator):
    """Resource allocator that tracks calls for testing."""

    def __init__(self):
        super().__init__()
        self.allocation_requests = []
        self.release_calls = []
        self.always_allocate = True

        # Add default resource pools
        self.add_resource_pool(
            ResourceType.CPU,
            ResourceQuota(ResourceType.CPU, limit=8.0, unit="cores", renewable=False))
        self.add_resource_pool(
            ResourceType.MEMORY,
            ResourceQuota(ResourceType.MEMORY, limit=16.0, unit="GB", renewable=False))

    async def request_resources(self, request: Any) -> bool:
        """Track resource requests and return configured result."""
        self.allocation_requests.append(request)

        if self.always_allocate:
            # Actually allocate the resources
            return await super().request_resources(request)
        return False

    async def release_resources(self, task_id: str):
        """Track resource releases."""
        self.release_calls.append(task_id)
        await super().release_resources(task_id)

    def was_released(self, task_id: str) -> bool:
        """Check if resources were released for a task."""
        return task_id in self.release_calls

    def reset_tracking(self):
        """Reset tracking data."""
        self.allocation_requests.clear()
        self.release_calls.clear()
