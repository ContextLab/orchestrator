"""Code execution tools."""

import asyncio
import sys
import subprocess
import tempfile
import os
from typing import Any, Dict
from pathlib import Path

from .base import Tool


class PythonExecutorTool(Tool):
    """Tool for executing Python code in a sandboxed environment."""

    def __init__(self):
        super().__init__(
            name="python-executor",
            description="Execute Python code in a sandboxed environment"
        )
        self.add_parameter("code", "string", "Python code to execute")
        self.add_parameter(
            "timeout", "integer", "Execution timeout in seconds", required=False, default=30
        )
        self.add_parameter(
            "capture_output", "boolean", "Capture output", required=False, default=True
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute Python code."""
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 30)
        capture_output = kwargs.get("capture_output", True)

        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute the Python code
            if capture_output:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.communicate()
                    return {
                        "code": code,
                        "return_code": -1,
                        "error": f"Code execution timed out after {timeout} seconds",
                        "success": False,
                    }

                return {
                    "code": code,
                    "return_code": process.returncode,
                    "stdout": stdout.decode("utf-8") if stdout else "",
                    "stderr": stderr.decode("utf-8") if stderr else "",
                    "success": process.returncode == 0,
                }
            else:
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_file
                )

                try:
                    return_code = await asyncio.wait_for(process.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return {
                        "code": code,
                        "return_code": -1,
                        "error": f"Code execution timed out after {timeout} seconds",
                        "success": False,
                    }

                return {
                    "code": code,
                    "return_code": return_code,
                    "success": return_code == 0,
                }

        except Exception as e:
            return {
                "code": code,
                "return_code": -1,
                "error": str(e),
                "success": False,
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)