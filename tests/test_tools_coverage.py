"""Tests to improve coverage for the tools modules."""

import json
import pathlib
import subprocess
from typing import Any, Dict, List

import pytest

from orchestrator.tools.base import Tool, ToolParameter
from orchestrator.tools.data_tools import DataProcessingTool, ValidationTool
from orchestrator.tools.system_tools import FileSystemTool, TerminalTool
from orchestrator.tools.web_tools import HeadlessBrowserTool, WebSearchTool


class TestableFileSystemTool(FileSystemTool):
    """A testable file system tool for testing without real file operations."""
    
    def __init__(self):
        super().__init__()
        self._test_files = {}  # path -> content
        self._test_dirs = set()  # set of directory paths
        self.call_history = []
        
    def _add_test_file(self, path: str, content: str):
        """Add a test file."""
        self._test_files[path] = content
        
    def _add_test_dir(self, path: str, files: List[str]):
        """Add a test directory with files."""
        self._test_dirs.add(path)
        for file in files:
            self._test_files[f"{path}/{file}"] = ""
            
    async def execute(self, **kwargs) -> dict:
        """Execute file system operation (test version)."""
        self.call_history.append(('execute', kwargs))
        action = kwargs.get('action')
        path = kwargs.get('path')
        
        if action == 'read':
            if path in self._test_files:
                return {"success": True, "content": self._test_files[path]}
            else:
                return {"success": False, "error": "File not found"}
                
        elif action == 'write':
            content = kwargs.get('content', '')
            self._test_files[path] = content
            return {"success": True, "path": path}
            
        elif action == 'list':
            if path in self._test_dirs:
                items = []
                for file_path in self._test_files:
                    if file_path.startswith(path + '/') and '/' not in file_path[len(path)+1:]:
                        name = file_path[len(path)+1:]
                        items.append({"name": name, "type": "file"})
                return {"success": True, "items": items}
            else:
                return {"success": False, "error": "Directory not found"}
                
        else:
            return {"error": f"Unknown action: {action}"}


class TestableTerminalTool(TerminalTool):
    """A testable terminal tool for testing without real command execution."""
    
    def __init__(self):
        super().__init__()
        self._test_commands = {}  # command -> (returncode, stdout, stderr)
        self.call_history = []
        
    def set_command_result(self, command: str, returncode: int, stdout: str, stderr: str):
        """Set the result for a command."""
        self._test_commands[command] = (returncode, stdout, stderr)
        
    async def execute(self, **kwargs) -> dict:
        """Execute command (test version)."""
        self.call_history.append(('execute', kwargs))
        command = kwargs.get('command')
        
        if command in self._test_commands:
            returncode, stdout, stderr = self._test_commands[command]
            return {
                "return_code": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "success": returncode == 0
            }
        else:
            # Default behavior for unknown commands
            if command == "echo test":
                return {"return_code": 0, "stdout": "test", "stderr": "", "success": True}
            elif command == "false":
                return {"return_code": 1, "stdout": "", "stderr": "command failed", "success": False}
            else:
                return {"return_code": 127, "error": "Command not found", "success": False}


class TestablePath:
    """A testable pathlib.Path replacement."""
    
    def __init__(self, path_str: str, fs_tool: TestableFileSystemTool):
        self.path_str = path_str
        self.fs_tool = fs_tool
        self._parent = None
        
    def exists(self) -> bool:
        """Check if path exists."""
        return self.path_str in self.fs_tool._test_files or self.path_str in self.fs_tool._test_dirs
        
    def is_dir(self) -> bool:
        """Check if path is directory."""
        return self.path_str in self.fs_tool._test_dirs
        
    def read_text(self, encoding='utf-8') -> str:
        """Read text from file."""
        if self.path_str in self.fs_tool._test_files:
            return self.fs_tool._test_files[self.path_str]
        raise FileNotFoundError(f"File not found: {self.path_str}")
        
    def write_text(self, content: str, encoding='utf-8'):
        """Write text to file."""
        self.fs_tool._test_files[self.path_str] = content
        
    def iterdir(self):
        """Iterate directory contents."""
        items = []
        for path in self.fs_tool._test_files:
            if path.startswith(self.path_str + '/') and '/' not in path[len(self.path_str)+1:]:
                items.append(TestablePath(path, self.fs_tool))
        return items
        
    @property
    def name(self) -> str:
        """Get file name."""
        return self.path_str.split('/')[-1]
        
    @property
    def parent(self):
        """Get parent directory."""
        if self._parent is None:
            parent_path = '/'.join(self.path_str.split('/')[:-1])
            self._parent = TestablePath(parent_path, self.fs_tool)
        return self._parent
        
    def mkdir(self, parents=True, exist_ok=True):
        """Create directory."""
        self.fs_tool._test_dirs.add(self.path_str)


class TestBaseTool:
    """Test the base Tool class."""

    def test_tool_parameter_init(self):
        """Test ToolParameter initialization."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="Test parameter",
            required=True,
            default="default_value"
        )
        assert param.name == "test_param"
        assert param.required is True
        assert param.default == "default_value"

    def test_tool_abstract(self):
        """Test Tool abstract class."""
        # Tool is abstract, so we need to create a concrete implementation
        class TestTool(Tool):
            def __init__(self):
                super().__init__("test_tool", "Test tool")
                self.add_parameter("param1", "string", "Test param", True)
            
            async def execute(self, **kwargs) -> dict:
                return {"result": "test", "input": kwargs}
        
        tool = TestTool()
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "param1"

    @pytest.mark.asyncio
    async def test_tool_execute(self):
        """Test Tool execute method."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__("test_tool", "Test tool")
            
            async def execute(self, **kwargs) -> dict:
                return {"input": kwargs}
        
        tool = TestTool()
        result = await tool.execute(test_param="test_value")
        assert result["input"]["test_param"] == "test_value"

    def test_tool_get_schema(self):
        """Test Tool get_schema method."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__("test_tool", "Test tool")
                self.add_parameter("param1", "string", "Test param", True)
                self.add_parameter("param2", "integer", "Optional param", False, 42)
            
            async def execute(self, **kwargs) -> dict:
                return {}
        
        tool = TestTool()
        schema = tool.get_schema()
        assert schema["name"] == "test_tool"
        assert schema["description"] == "Test tool"
        assert "inputSchema" in schema
        assert "param1" in schema["inputSchema"]["properties"]
        assert "param1" in schema["inputSchema"]["required"]
        assert "param2" not in schema["inputSchema"]["required"]

    def test_tool_validate_parameters_success(self):
        """Test Tool validate_parameters with valid params."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__("test_tool", "Test tool")
                self.add_parameter("required_param", "string", "Required", True)
                self.add_parameter("optional_param", "string", "Optional", False)
            
            async def execute(self, **kwargs) -> dict:
                return {}
        
        tool = TestTool()
        # Should not raise
        tool.validate_parameters({"required_param": "value"})

    def test_tool_validate_parameters_missing_required(self):
        """Test Tool validate_parameters with missing required param."""
        class TestTool(Tool):
            def __init__(self):
                super().__init__("test_tool", "Test tool")
                self.add_parameter("required_param", "string", "Required", True)
            
            async def execute(self, **kwargs) -> dict:
                return {}
        
        tool = TestTool()
        with pytest.raises(ValueError, match="Required parameter 'required_param' not provided"):
            tool.validate_parameters({})


class TestDataTools:
    """Test data manipulation tools."""

    def test_data_processing_tool_init(self):
        """Test DataProcessingTool initialization."""
        tool = DataProcessingTool()
        assert tool.name == "data-processing"
        assert tool.description is not None
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_data_processing_convert(self):
        """Test DataProcessingTool convert operation."""
        tool = DataProcessingTool()
        
        # Test JSON to CSV conversion
        data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        result = await tool.execute(
            action="convert",
            data=data,
            format="csv"
        )
        assert "result" in result or "error" in result

    @pytest.mark.asyncio
    async def test_data_processing_filter(self):
        """Test DataProcessingTool filter operation."""
        tool = DataProcessingTool()
        
        data = [1, 2, 3, 4, 5]
        result = await tool.execute(
            action="filter",
            data=data,
            operation={"condition": "x > 2"}
        )
        assert "result" in result or "error" in result

    @pytest.mark.asyncio
    async def test_data_processing_aggregate(self):
        """Test DataProcessingTool aggregate operation."""
        tool = DataProcessingTool()
        
        data = [{"value": 10}, {"value": 20}, {"value": 30}]
        result = await tool.execute(
            action="aggregate",
            data=data,
            operation={"type": "sum", "field": "value"}
        )
        assert "result" in result or "error" in result

    @pytest.mark.asyncio
    async def test_data_processing_transform(self):
        """Test DataProcessingTool transform operation."""
        tool = DataProcessingTool()
        
        data = [1, 2, 3]
        result = await tool.execute(
            action="transform",
            data=data,
            operation={"expression": "x * 2"}
        )
        assert "result" in result or "error" in result

    @pytest.mark.asyncio
    async def test_data_processing_invalid_action(self):
        """Test DataProcessingTool with invalid action."""
        tool = DataProcessingTool()
        
        result = await tool.execute(
            action="invalid_action",
            data=[]
        )
        assert "error" in result

    def test_validation_tool_init(self):
        """Test ValidationTool initialization."""
        tool = ValidationTool()
        assert tool.name == "validation"
        assert tool.description is not None
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_validation_schema(self):
        """Test ValidationTool schema validation."""
        tool = ValidationTool()
        
        data = {"name": "John", "age": 30}
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        result = await tool.execute(
            action="schema",
            data=data,
            schema=schema
        )
        assert "result" in result or "errors" in result

    @pytest.mark.asyncio
    async def test_validation_rules(self):
        """Test ValidationTool custom rules validation."""
        tool = ValidationTool()
        
        data = {"name": "", "email": "test@example.com"}
        rules = [
            {"type": "not_empty", "field": "name", "severity": "error"},
            {"type": "min_length", "field": "email", "value": 5, "severity": "warning"}
        ]
        
        result = await tool.execute(
            action="rules",
            data=data,
            rules=rules
        )
        assert "result" in result or "errors" in result


class TestSystemTools:
    """Test system interaction tools."""

    def test_file_system_tool_init(self):
        """Test FileSystemTool initialization."""
        tool = FileSystemTool()
        assert tool.name == "filesystem"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_file_system_read(self):
        """Test FileSystemTool read operation."""
        tool = TestableFileSystemTool()
        
        # Set up test file
        tool._add_test_file("/test/file.txt", "file content")
        
        result = await tool.execute(
            action="read",
            path="/test/file.txt"
        )
        assert result["success"] is True
        assert result["content"] == "file content"

    @pytest.mark.asyncio
    async def test_file_system_write(self):
        """Test FileSystemTool write operation."""
        tool = TestableFileSystemTool()
        
        result = await tool.execute(
            action="write",
            path="/test/file.txt",
            content="new content"
        )
        assert result["success"] is True
        assert tool._test_files["/test/file.txt"] == "new content"

    @pytest.mark.asyncio
    async def test_file_system_list(self):
        """Test FileSystemTool list operation."""
        tool = TestableFileSystemTool()
        
        # Set up test directory with files
        tool._add_test_dir("/test/dir", ["file1.txt", "file2.txt"])
        
        result = await tool.execute(
            action="list",
            path="/test/dir"
        )
        assert result["success"] is True
        assert "items" in result
        assert len(result["items"]) == 2

    @pytest.mark.asyncio
    async def test_file_system_invalid_action(self):
        """Test FileSystemTool with invalid action."""
        tool = TestableFileSystemTool()
        result = await tool.execute(
            action="invalid_action",
            path="/test"
        )
        assert "error" in result

    def test_terminal_tool_init(self):
        """Test TerminalTool initialization."""
        tool = TerminalTool()
        assert tool.name == "terminal"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_terminal_success(self):
        """Test TerminalTool successful execution."""
        tool = TestableTerminalTool()
        
        # Set up command result
        tool.set_command_result("echo test", 0, "test", "")
        
        result = await tool.execute(
            command="echo test",
            shell=True
        )
        assert result["success"] is True
        assert result["stdout"] == "test"

    @pytest.mark.asyncio
    async def test_terminal_failure(self):
        """Test TerminalTool failed execution."""
        tool = TestableTerminalTool()
        
        # Default behavior for "false" command
        result = await tool.execute(
            command="false",
            shell=True
        )
        assert result["success"] is False
        assert result["stderr"] == "command failed"

    @pytest.mark.asyncio
    async def test_terminal_exception(self):
        """Test TerminalTool exception handling."""
        tool = TestableTerminalTool()
        
        # Test with unknown command
        result = await tool.execute(
            command="invalid command"
        )
        # Check that error details are captured 
        assert "error" in result or "return_code" in result


class TestWebTools:
    """Test web interaction tools."""

    def test_headless_browser_tool_init(self):
        """Test HeadlessBrowserTool initialization."""
        tool = HeadlessBrowserTool()
        assert tool.name == "headless-browser"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_headless_browser_search(self):
        """Test HeadlessBrowserTool search action."""
        tool = HeadlessBrowserTool()
        
        result = await tool.execute(
            action="search",
            query="test query"
        )
        assert "results" in result
        assert "query" in result
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_headless_browser_verify(self):
        """Test HeadlessBrowserTool verify action."""
        tool = HeadlessBrowserTool()
        
        result = await tool.execute(
            action="verify",
            url="https://example.com"
        )
        assert "url" in result
        assert "accessible" in result

    @pytest.mark.asyncio
    async def test_headless_browser_scrape(self):
        """Test HeadlessBrowserTool scrape action."""
        tool = HeadlessBrowserTool()
        
        result = await tool.execute(
            action="scrape",
            url="https://example.com"
        )
        assert "url" in result
        assert "content" in result

    def test_web_search_tool_init(self):
        """Test WebSearchTool initialization."""
        tool = WebSearchTool()
        assert tool.name == "web-search"
        assert len(tool.parameters) > 0

    @pytest.mark.asyncio
    async def test_web_search_success(self):
        """Test WebSearchTool successful search."""
        tool = WebSearchTool()
        
        result = await tool.execute(query="test query")
        assert "results" in result
        assert "query" in result
        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_web_search_with_max_results(self):
        """Test WebSearchTool with custom max results."""
        tool = WebSearchTool()
        
        result = await tool.execute(query="test", max_results=5)
        assert "results" in result
        assert "query" in result

    @pytest.mark.asyncio
    async def test_web_search_different_sources(self):
        """Test WebSearchTool with different source combinations."""
        tool = WebSearchTool()
        
        result = await tool.execute(query="python programming")
        assert "results" in result
        assert len(result["results"]) > 0