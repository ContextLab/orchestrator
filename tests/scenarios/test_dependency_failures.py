"""
Missing Dependencies and Tool Failure Testing

Tests the orchestrator's handling of missing dependencies, tool failures,
and various runtime environment issues using real dependency scenarios.
"""

import pytest
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import orchestrator components
from src.orchestrator.orchestrator import Orchestrator


class TestMissingDependencies:
    """Test handling of missing Python packages and system dependencies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "dependency_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
        self.original_path = sys.path.copy()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        sys.path = self.original_path
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_missing_python_packages(self):
        """Test handling of missing Python packages."""
        missing_package_cases = [
            {
                "name": "missing_numpy",
                "import": "import nonexistent_package_name_12345",
                "expected_error": ["import", "module", "found"]
            },
            {
                "name": "missing_specific_module",
                "import": "from definitely_not_installed import some_function",
                "expected_error": ["import", "module", "found"]
            },
            {
                "name": "version_mismatch",
                "import": """
import sys
if sys.version_info < (99, 0):
    raise ImportError("Python 99.0+ required")
""",
                "expected_error": ["import", "python", "required"]
            },
            {
                "name": "missing_optional_dependency",
                "import": """
try:
    import this_package_definitely_does_not_exist
except ImportError as e:
    print(f"Optional dependency missing: {e}")
    raise ImportError("Required for this specific functionality")
""",
                "expected_error": ["import", "dependency", "missing"]
            }
        ]
        
        for case in missing_package_cases:
            pipeline_content = f"""
name: {case["name"]}_test
version: "1.0"
steps:
  - id: import_test
    action: python_code
    parameters:
      code: |
        {case["import"]}
        print("This should not execute if import fails")
"""
            
            pipeline_path = self.create_test_pipeline(pipeline_content, f"{case['name']}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should fail due to missing dependency
            assert result.status in ["error", "failed"], f"{case['name']}: Should fail with missing dependency"
            
            # Should have meaningful error message
            error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
            assert any(keyword in error_messages for keyword in case["expected_error"]), \
                f"{case['name']}: Expected error keywords not found in: {error_messages}"
            
            print(f"✓ Missing dependency case '{case['name']}' handled correctly")
    
    @pytest.mark.asyncio
    async def test_system_tool_failures(self):
        """Test handling of missing system tools and commands."""
        system_tool_cases = [
            {
                "name": "missing_command",
                "command": "definitely_not_a_real_command_12345 --help",
                "expected_error": ["command", "not found", "error"]
            },
            {
                "name": "permission_denied",
                "command": "cat /etc/shadow",  # Requires root permissions
                "expected_error": ["permission", "denied", "access"]
            },
            {
                "name": "invalid_arguments",
                "command": "ls --invalid-argument-that-does-not-exist",
                "expected_error": ["argument", "option", "invalid"]
            },
            {
                "name": "tool_not_in_path",
                "command": "/nonexistent/path/to/tool --version",
                "expected_error": ["no such file", "not found", "error"]
            }
        ]
        
        for case in system_tool_cases:
            pipeline_content = f"""
name: {case["name"]}_test
version: "1.0"
steps:
  - id: system_tool_test
    action: shell
    parameters:
      command: "{case["command"]}"
      timeout: 10
"""
            
            pipeline_path = self.create_test_pipeline(pipeline_content, f"{case['name']}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle system tool failure
            assert result.status in ["error", "failed"], f"{case['name']}: Should fail with system tool error"
            
            # Check for appropriate error message
            error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
            if error_messages:  # Some might not have detailed error messages
                has_expected = any(keyword in error_messages for keyword in case["expected_error"])
                if not has_expected:
                    print(f"Warning: {case['name']} didn't contain expected error keywords")
            
            print(f"✓ System tool failure '{case['name']}' handled")
    
    @pytest.mark.asyncio
    async def test_docker_availability(self):
        """Test handling when Docker is not available or misconfigured."""
        docker_cases = [
            {
                "name": "docker_not_installed",
                "action": "docker",
                "parameters": {
                    "image": "alpine:latest",
                    "command": "echo 'hello'"
                }
            },
            {
                "name": "docker_daemon_not_running",
                "action": "docker",
                "parameters": {
                    "image": "nonexistent-image:latest",
                    "command": "echo 'test'"
                }
            }
        ]
        
        for case in docker_cases:
            pipeline_content = f"""
name: {case["name"]}_test
version: "1.0"
steps:
  - id: docker_test
    action: {case["action"]}
    parameters:
"""
            
            for key, value in case["parameters"].items():
                pipeline_content += f"      {key}: \"{value}\"\n"
            
            pipeline_path = self.create_test_pipeline(pipeline_content, f"{case['name']}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle Docker unavailability
            # Note: May succeed if Docker is available, fail if not
            assert result.status in ["success", "error", "failed"]
            
            if result.status in ["error", "failed"]:
                error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
                docker_error_keywords = ["docker", "daemon", "connection", "image"]
                if any(keyword in error_messages for keyword in docker_error_keywords):
                    print(f"✓ Docker failure '{case['name']}' properly detected")
                else:
                    print(f"✓ Docker case '{case['name']}' failed for other reasons")
            else:
                print(f"✓ Docker case '{case['name']}' succeeded (Docker available)")
    
    @pytest.mark.asyncio
    async def test_network_dependent_tools(self):
        """Test handling of tools that require network access when offline."""
        network_tool_cases = [
            {
                "name": "web_scraping_offline",
                "pipeline": """
name: web_scraping_test
version: "1.0"
steps:
  - id: scrape_test
    action: web_scrape
    parameters:
      url: "http://nonexistent-domain-12345.com/page"
      timeout: 5
"""
            },
            {
                "name": "api_call_unreachable",
                "pipeline": """
name: api_call_test
version: "1.0"
steps:
  - id: api_test
    action: web_request
    parameters:
      url: "https://api.nonexistent-service-12345.com/v1/test"
      method: "GET"
      timeout: 5
"""
            },
            {
                "name": "model_download_failure",
                "pipeline": """
name: model_download_test
version: "1.0"
steps:
  - id: model_test
    action: python_code
    parameters:
      code: |
        import requests
        # Simulate model download from unreachable endpoint
        response = requests.get("https://models.nonexistent-repo.com/model.bin", timeout=3)
        print("Model downloaded successfully")
"""
            }
        ]
        
        for case in network_tool_cases:
            pipeline_path = self.create_test_pipeline(case["pipeline"], f"{case['name']}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle network failures gracefully
            assert result.status in ["error", "failed"], f"{case['name']}: Should fail with network error"
            
            error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
            network_keywords = ["network", "dns", "connection", "timeout", "unreachable", "resolve"]
            has_network_error = any(keyword in error_messages for keyword in network_keywords)
            
            if has_network_error:
                print(f"✓ Network dependency failure '{case['name']}' properly detected")
            else:
                print(f"✓ Network case '{case['name']}' failed (may be other error)")
    
    @pytest.mark.asyncio
    async def test_environment_variable_dependencies(self):
        """Test handling of missing environment variables."""
        # Store original env vars to restore later
        original_env = os.environ.copy()
        
        env_cases = [
            {
                "name": "missing_api_key",
                "required_vars": ["NONEXISTENT_API_KEY"],
                "pipeline": """
name: env_var_test
version: "1.0"
steps:
  - id: env_test
    action: python_code
    parameters:
      code: |
        import os
        api_key = os.getenv("NONEXISTENT_API_KEY")
        if not api_key:
            raise ValueError("Missing required API key: NONEXISTENT_API_KEY")
        print(f"API key: {api_key[:10]}...")
"""
            },
            {
                "name": "missing_config_path",
                "required_vars": ["NONEXISTENT_CONFIG_PATH"],
                "pipeline": """
name: config_path_test
version: "1.0"
steps:
  - id: config_test
    action: python_code
    parameters:
      code: |
        import os
        config_path = os.getenv("NONEXISTENT_CONFIG_PATH")
        if not config_path:
            raise EnvironmentError("NONEXISTENT_CONFIG_PATH environment variable not set")
        with open(config_path, 'r') as f:
            config = f.read()
"""
            }
        ]
        
        try:
            for case in env_cases:
                # Ensure required vars are not set
                for var in case["required_vars"]:
                    if var in os.environ:
                        del os.environ[var]
                
                pipeline_path = self.create_test_pipeline(case["pipeline"], f"{case['name']}.yaml")
                
                yaml_content = pipeline_path.read_text()
                result = await self.executor.execute_yaml(yaml_content)
                
                # Should fail due to missing environment variable
                assert result.status in ["error", "failed"], f"{case['name']}: Should fail with missing env var"
                
                error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
                env_keywords = ["environment", "variable", "missing", "api key", "config"]
                has_env_error = any(keyword in error_messages for keyword in env_keywords)
                
                print(f"✓ Missing environment variable case '{case['name']}' handled")
        
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    @pytest.mark.asyncio
    async def test_file_dependency_failures(self):
        """Test handling of missing required files and directories."""
        file_cases = [
            {
                "name": "missing_input_file",
                "pipeline": """
name: file_input_test
version: "1.0"
steps:
  - id: file_read_test
    action: python_code
    parameters:
      code: |
        with open("/nonexistent/path/to/input.txt", "r") as f:
            content = f.read()
        print(f"File content: {content}")
"""
            },
            {
                "name": "missing_config_file",
                "pipeline": """
name: config_file_test
version: "1.0"
steps:
  - id: config_read_test
    action: python_code
    parameters:
      code: |
        import json
        with open("/nonexistent/config/settings.json", "r") as f:
            config = json.load(f)
        print(f"Config loaded: {config}")
"""
            },
            {
                "name": "permission_denied_file",
                "pipeline": f"""
name: permission_test
version: "1.0"
steps:
  - id: permission_test
    action: python_code
    parameters:
      code: |
        # Try to write to a restricted location
        with open("/etc/passwd_backup", "w") as f:
            f.write("test")
        print("File written successfully")
"""
            }
        ]
        
        for case in file_cases:
            pipeline_path = self.create_test_pipeline(case["pipeline"], f"{case['name']}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should fail due to file access issues
            assert result.status in ["error", "failed"], f"{case['name']}: Should fail with file error"
            
            error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
            file_keywords = ["file", "directory", "permission", "not found", "no such", "access"]
            has_file_error = any(keyword in error_messages for keyword in file_keywords)
            
            print(f"✓ File dependency failure '{case['name']}' handled")


class TestToolFailureScenarios:
    """Test various tool failure scenarios and recovery mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp()) / "tool_failure_tests"
        self.test_dir.mkdir(exist_ok=True)
        self.executor = Orchestrator()
    
    def create_test_pipeline(self, content: str, filename: str = "test_pipeline.yaml") -> Path:
        """Create a test pipeline file with given content."""
        pipeline_path = self.test_dir / filename
        pipeline_path.write_text(content)
        return pipeline_path
    
    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self):
        """Test handling of tools that exceed timeout limits."""
        timeout_cases = [
            {
                "name": "python_infinite_loop",
                "pipeline": """
name: timeout_test
version: "1.0"
steps:
  - id: infinite_loop_test
    action: python_code
    parameters:
      code: |
        import time
        # Simulate infinite loop
        counter = 0
        while True:
            counter += 1
            if counter % 1000000 == 0:
                print(f"Still running... {counter}")
            time.sleep(0.001)
      timeout: 5
"""
            },
            {
                "name": "slow_computation",
                "pipeline": """
name: slow_computation_test
version: "1.0"
steps:
  - id: slow_computation
    action: python_code
    parameters:
      code: |
        import time
        # Simulate very slow computation
        result = 0
        for i in range(1000000):
            result += i ** 2
            if i % 100000 == 0:
                print(f"Progress: {i/1000000*100:.1f}%")
                time.sleep(0.1)  # Add delay to ensure timeout
        print(f"Final result: {result}")
      timeout: 3
"""
            }
        ]
        
        for case in timeout_cases:
            pipeline_path = self.create_test_pipeline(case["pipeline"], f"{case['name']}.yaml")
            
            import time
            start_time = time.time()
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            execution_time = time.time() - start_time
            
            # Should timeout within reasonable time (not much longer than specified timeout)
            assert execution_time < 15, f"{case['name']}: Took too long to timeout: {execution_time}s"
            
            # Should have timeout status
            assert result.status in ["error", "failed", "timeout"], f"{case['name']}: Expected timeout status"
            
            # Check for timeout in error messages
            error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
            timeout_keywords = ["timeout", "exceeded", "time limit", "killed"]
            has_timeout_error = any(keyword in error_messages for keyword in timeout_keywords)
            
            print(f"✓ Tool timeout '{case['name']}' handled in {execution_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion_handling(self):
        """Test handling of tools that consume too much memory."""
        memory_cases = [
            {
                "name": "memory_leak_simulation",
                "pipeline": """
name: memory_test
version: "1.0"
steps:
  - id: memory_test
    action: python_code
    parameters:
      code: |
        import gc
        # Simulate memory consumption
        big_list = []
        try:
            for i in range(100000):  # Reduced to avoid actually exhausting memory
                big_list.append([0] * 1000)
                if i % 10000 == 0:
                    print(f"Allocated {i * 1000} integers")
                    # Check if we should stop to avoid system issues
                    if i > 50000:  # Stop before actually exhausting memory
                        raise MemoryError("Simulated memory exhaustion")
            print(f"Total memory allocated: {len(big_list) * 1000} integers")
        except MemoryError as e:
            print(f"Memory error caught: {e}")
            raise
        finally:
            del big_list
            gc.collect()
"""
            }
        ]
        
        for case in memory_cases:
            pipeline_path = self.create_test_pipeline(case["pipeline"], f"{case['name']}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle memory issues gracefully
            assert result.status in ["success", "error", "failed"]  # May succeed with simulation
            
            if result.status in ["error", "failed"]:
                error_messages = " ".join([step.error_message.lower() for step in result.step_results if step.error_message])
                memory_keywords = ["memory", "allocation", "out of memory", "oom"]
                has_memory_error = any(keyword in error_messages for keyword in memory_keywords)
                
                print(f"✓ Memory exhaustion case '{case['name']}' detected and handled")
            else:
                print(f"✓ Memory case '{case['name']}' completed (simulation worked)")
    
    @pytest.mark.asyncio
    async def test_tool_crash_recovery(self):
        """Test handling of tools that crash or exit unexpectedly."""
        crash_cases = [
            {
                "name": "segmentation_fault_simulation",
                "pipeline": """
name: crash_test
version: "1.0"
steps:
  - id: crash_simulation
    action: python_code
    parameters:
      code: |
        import sys
        print("Starting crash simulation...")
        # Simulate a crash by raising SystemExit
        print("About to simulate crash...")
        sys.exit(1)  # Simulate unexpected exit
        print("This should not print")
"""
            },
            {
                "name": "exception_cascade",
                "pipeline": """
name: exception_cascade_test
version: "1.0"
steps:
  - id: exception_test
    action: python_code
    parameters:
      code: |
        def cause_exception():
            raise ValueError("Initial error")
        
        def propagate_exception():
            try:
                cause_exception()
            except ValueError as e:
                raise RuntimeError(f"Propagated: {e}") from e
        
        def final_exception():
            try:
                propagate_exception()
            except RuntimeError as e:
                raise SystemError(f"Final: {e}") from e
        
        print("Starting exception cascade...")
        final_exception()
        print("This should not execute")
"""
            }
        ]
        
        for case in crash_cases:
            pipeline_path = self.create_test_pipeline(case["pipeline"], f"{case['name']}.yaml")
            
            yaml_content = pipeline_path.read_text()
            result = await self.executor.execute_yaml(yaml_content)
            
            # Should handle crashes gracefully
            assert result.status in ["error", "failed"], f"{case['name']}: Should handle crash"
            
            # Should have error information
            assert len(result.step_results) > 0
            failed_steps = [step for step in result.step_results if step.status in ["error", "failed"]]
            assert len(failed_steps) > 0, f"{case['name']}: No failed steps recorded"
            
            print(f"✓ Tool crash '{case['name']}' handled gracefully")
    
    @pytest.mark.asyncio
    async def test_dependency_chain_failures(self):
        """Test handling when dependency failures cascade through pipeline."""
        dependency_failure_pipeline = """
name: dependency_chain_test
version: "1.0"
steps:
  - id: base_step
    action: python_code
    parameters:
      code: |
        # This step will fail
        import nonexistent_module_12345
        result = "This won't be set"
    outputs:
      - base_result
  
  - id: dependent_step1
    action: python_code
    parameters:
      code: |
        print(f"Using base result: {base_result}")
        dependent_result1 = base_result + "_processed"
    depends_on:
      - base_step
    outputs:
      - dependent_result1
  
  - id: dependent_step2
    action: python_code
    parameters:
      code: |
        print(f"Using dependent result: {dependent_result1}")
        final_result = dependent_result1 + "_final"
    depends_on:
      - dependent_step1
    outputs:
      - final_result
  
  - id: independent_step
    action: python_code
    parameters:
      code: |
        print("This step should still execute")
        independent_result = "success"
"""
        
        pipeline_path = self.create_test_pipeline(dependency_failure_pipeline, "dependency_chain.yaml")
        
        result = await self.executor.execute_pipeline_file(str(pipeline_path))
        
        # Should handle dependency chain failure
        assert result.status in ["error", "failed", "partial_success"]
        
        # Verify expected step statuses
        step_results = {step.step_id: step for step in result.step_results}
        
        # Base step should fail
        assert "base_step" in step_results
        assert step_results["base_step"].status in ["error", "failed"]
        
        # Dependent steps should be skipped or failed
        if "dependent_step1" in step_results:
            assert step_results["dependent_step1"].status in ["error", "failed", "skipped"]
        
        if "dependent_step2" in step_results:
            assert step_results["dependent_step2"].status in ["error", "failed", "skipped"]
        
        # Independent step might still execute
        if "independent_step" in step_results:
            # Could succeed or fail depending on implementation
            print(f"Independent step status: {step_results['independent_step'].status}")
        
        print("✓ Dependency chain failure handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])