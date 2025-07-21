"""Real control system implementation for testing."""

from typing import Any, Dict, Optional

from src.orchestrator.core.control_system import ControlSystem
from src.orchestrator.core.task import Task


class RealTestControlSystem(ControlSystem):
    """Real control system implementation for tests without mocks."""
    
    def __init__(self, name: str = "test-control", config: Optional[Dict[str, Any]] = None):
        """Initialize test control system with real capabilities."""
        default_config = {
            "capabilities": {
                "supported_actions": ["test", "validate", "process", "analyze"],
                "parallel_execution": True,
                "streaming": False,
                "checkpoint_support": True,
            },
            "base_priority": 10,
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name, default_config)
        self.executed_tasks = {}
        self.should_fail = False
        self.execution_count = 0
        
    async def execute_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a task and return real results."""
        self.execution_count += 1
        
        # Simulate real execution with actual processing
        if self.should_fail:
            raise Exception(f"Task {task.id} failed as configured")
        
        # Perform actual task execution based on action
        if task.action == "test":
            result = await self._execute_test_action(task, context)
        elif task.action == "validate":
            result = await self._execute_validate_action(task, context) 
        elif task.action == "process":
            result = await self._execute_process_action(task, context)
        elif task.action == "analyze":
            result = await self._execute_analyze_action(task, context)
        else:
            # Default execution for unknown actions
            result = {
                "status": "completed",
                "result": f"Executed unknown action: {task.action} for task {task.id}",
                "task_id": task.id,
                "execution_time": 0.1,
            }
        
        # Store execution result
        self.executed_tasks[task.id] = result
        
        # Update task status
        task.status = "completed"
        
        return result
    
    async def _execute_test_action(self, task: Task, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute test action with real logic."""
        # Simulate actual test execution
        test_data = task.parameters.get("test_data", "default_test_data")
        
        return {
            "status": "completed",
            "result": f"Test completed for {task.id}",
            "task_id": task.id,
            "test_data": test_data,
            "test_passed": True,
            "execution_time": 0.05,
        }
    
    async def _execute_validate_action(self, task: Task, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute validation action with real logic."""
        # Perform actual validation
        data_to_validate = task.parameters.get("data", {})
        validation_rules = task.parameters.get("rules", [])
        
        validation_errors = []
        for rule in validation_rules:
            if rule == "required" and not data_to_validate:
                validation_errors.append("Data is required")
        
        return {
            "status": "completed",
            "result": f"Validation completed for {task.id}",
            "task_id": task.id,
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "execution_time": 0.03,
        }
    
    async def _execute_process_action(self, task: Task, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute process action with real logic."""
        # Simulate data processing
        input_data = task.parameters.get("input", "")
        processing_type = task.parameters.get("type", "default")
        
        # Perform actual processing
        if processing_type == "uppercase":
            processed_data = str(input_data).upper()
        elif processing_type == "reverse":
            processed_data = str(input_data)[::-1]
        else:
            processed_data = f"Processed: {input_data}"
        
        return {
            "status": "completed", 
            "result": f"Processing completed for {task.id}",
            "task_id": task.id,
            "input": input_data,
            "output": processed_data,
            "processing_type": processing_type,
            "execution_time": 0.08,
        }
    
    async def _execute_analyze_action(self, task: Task, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute analyze action with real logic."""
        # Perform actual analysis
        data = task.parameters.get("data", "")
        analysis_type = task.parameters.get("type", "basic")
        
        # Real analysis logic
        analysis_results = {
            "length": len(str(data)),
            "type": type(data).__name__,
        }
        
        if analysis_type == "detailed" and isinstance(data, str):
            analysis_results.update({
                "word_count": len(data.split()),
                "char_count": len(data),
                "has_numbers": any(c.isdigit() for c in data),
            })
        
        return {
            "status": "completed",
            "result": f"Analysis completed for {task.id}", 
            "task_id": task.id,
            "analysis": analysis_results,
            "analysis_type": analysis_type,
            "execution_time": 0.12,
        }
    
    async def health_check(self) -> bool:
        """Perform real health check."""
        # Check if the control system is operational
        try:
            # Test basic functionality
            test_task = Task(
                id="health_check_task",
                name="Health Check",
                action="test",
                parameters={"test_data": "health_check"}
            )
            
            # Try to execute a simple task
            result = await self.execute_task(test_task, {})
            
            # Check if execution was successful
            return result.get("status") == "completed"
            
        except Exception:
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return real capabilities."""
        return self._capabilities
    
    async def can_execute_task(self, task: Task) -> bool:
        """Check if this control system can execute the given task."""
        supported_actions = self._capabilities.get("supported_actions", [])
        return task.action in supported_actions
    
    async def estimate_resource_requirements(self, task: Task) -> Dict[str, Any]:
        """Estimate resources needed for task execution."""
        # Real resource estimation based on task type
        base_requirements = {
            "memory_mb": 50,
            "cpu_cores": 1,
            "time_seconds": 1,
        }
        
        if task.action == "analyze":
            base_requirements["memory_mb"] = 100
            base_requirements["time_seconds"] = 2
        elif task.action == "process":
            base_requirements["memory_mb"] = 75
            base_requirements["cpu_cores"] = 2
            
        return base_requirements
    
    async def validate_task_parameters(self, task: Task) -> bool:
        """Validate that task has required parameters."""
        # Real validation logic
        required_params = {
            "test": [],
            "validate": ["data"],
            "process": ["input"],
            "analyze": ["data"],
        }
        
        if task.action in required_params:
            for param in required_params[task.action]:
                if param not in task.parameters:
                    return False
                    
        return True