"""System-level tools for terminal commands and file operations."""

import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict

from .base import Tool


class TerminalTool(Tool):
    """Tool for executing terminal/shell commands."""

    def __init__(self):
        super().__init__(
            name="terminal",
            description="Execute terminal commands in a sandboxed environment",
        )
        self.add_parameter("command", "string", "Command to execute")
        self.add_parameter(
            "working_dir", "string", "Working directory", required=False, default="."
        )
        self.add_parameter(
            "timeout", "integer", "Timeout in seconds", required=False, default=30
        )
        self.add_parameter(
            "capture_output",
            "boolean",
            "Capture command output",
            required=False,
            default=True,
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute terminal command."""
        command = kwargs.get("command", "")
        working_dir = kwargs.get("working_dir", ".")
        timeout = kwargs.get("timeout", 30)
        capture_output = kwargs.get("capture_output", True)

        # Handle special command prefix '!' for direct shell execution
        if command.startswith("!"):
            command = command[1:]

        try:
            # Create working directory if it doesn't exist
            Path(working_dir).mkdir(parents=True, exist_ok=True)

            # Execute command
            if capture_output:
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=working_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )

                return {
                    "command": command,
                    "return_code": process.returncode,
                    "stdout": stdout.decode("utf-8") if stdout else "",
                    "stderr": stderr.decode("utf-8") if stderr else "",
                    "success": process.returncode == 0,
                    "working_dir": working_dir,
                    "execution_time": timeout,  # Simplified
                }
            else:
                process = await asyncio.create_subprocess_shell(
                    command, cwd=working_dir
                )

                return_code = await asyncio.wait_for(process.wait(), timeout=timeout)

                return {
                    "command": command,
                    "return_code": return_code,
                    "success": return_code == 0,
                    "working_dir": working_dir,
                }

        except asyncio.TimeoutError:
            return {
                "command": command,
                "return_code": -1,
                "error": "Command timed out",
                "success": False,
                "working_dir": working_dir,
            }
        except Exception as e:
            return {
                "command": command,
                "return_code": -1,
                "error": str(e),
                "success": False,
                "working_dir": working_dir,
            }


class FileSystemTool(Tool):
    """Tool for file system operations."""

    def __init__(self):
        super().__init__(
            name="filesystem",
            description="Perform file system operations like read, write, copy, move",
        )
        self.add_parameter(
            "action",
            "string",
            "Action: 'read', 'write', 'copy', 'move', 'delete', 'list'",
        )
        self.add_parameter("path", "string", "File or directory path")
        self.add_parameter(
            "content", "string", "Content to write (for write action)", required=False
        )
        self.add_parameter(
            "destination", "string", "Destination path (for copy/move)", required=False
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute file system operation."""
        action = kwargs.get("action", "")
        path = kwargs.get("path", "")
        
        # Extract template resolution components
        _template_manager = kwargs.get("_template_manager")
        _unified_resolver = kwargs.get("_unified_resolver")
        _resolution_context = kwargs.get("_resolution_context")
        
        # Render path if it contains templates (should already be done, but safety check)
        if isinstance(path, str) and ('{{' in path or '{%' in path):
            if _unified_resolver and _resolution_context:
                # Use unified resolver
                path = _unified_resolver.resolve_templates(path, _resolution_context)
            elif _template_manager:
                # Fallback to legacy template manager
                try:
                    from ..core.template_manager import TemplateManager
                    if isinstance(_template_manager, TemplateManager):
                        path = _template_manager.render(path)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to render path template: {e}")

        try:
            if action == "read":
                return await self._read_file(path)
            elif action == "write":
                content = kwargs.get("content", "")
                return await self._write_file(path, content, _template_manager, _unified_resolver, _resolution_context)
            elif action == "copy":
                destination = kwargs.get("destination", "")
                return await self._copy_file(path, destination)
            elif action == "move":
                destination = kwargs.get("destination", "")
                return await self._move_file(path, destination)
            elif action == "delete":
                return await self._delete_file(path)
            elif action == "list":
                return await self._list_directory(path)
            else:
                raise ValueError(f"Unknown filesystem action: {action}")

        except Exception as e:
            return {"action": action, "path": path, "success": False, "error": str(e)}

    async def _read_file(self, path: str) -> Dict[str, Any]:
        """Read file content."""
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path_obj.is_dir():
            raise IsADirectoryError(f"Path is a directory: {path}")

        content = path_obj.read_text(encoding="utf-8")

        return {
            "action": "read",
            "path": path,
            "content": content,
            "size": len(content),
            "success": True,
        }

    async def _write_file(self, path: str, content: str, _template_manager=None, _unified_resolver=None, _resolution_context=None) -> Dict[str, Any]:
        """Write content to file with optional runtime template rendering."""
        path_obj = Path(path)

        # Create parent directories if they don't exist
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"FileSystemTool._write_file called with path={path}")
        logger.info(f"Template manager provided: {_template_manager is not None}")
        logger.info(f"Content has templates: {isinstance(content, str) and ('{{' in content or '{%' in content)}")
        if _template_manager:
            # Log available context
            from ..core.template_manager import TemplateManager
            if isinstance(_template_manager, TemplateManager):
                logger.info(f"Template manager context keys: {list(_template_manager.context.keys())}")
        
        # Render content templates if available (should already be done, but safety check for runtime rendering)
        if isinstance(content, str) and ('{{' in content or '{%' in content):
            if _unified_resolver and _resolution_context:
                # Use unified resolver (preferred)
                try:
                    logger.info("Using unified resolver for content template rendering")
                    rendered = _unified_resolver.resolve_templates(content, _resolution_context)
                    logger.info(f"Unified resolver: Original had templates: {('{{' in content)}, Rendered has templates: {('{{' in rendered)}")
                    content = rendered
                except Exception as e:
                    logger.error(f"Failed to render templates with unified resolver: {e}", exc_info=True)
            elif _template_manager:
                # Fallback to legacy template manager
                try:
                    # Import here to avoid circular dependency
                    from ..core.template_manager import TemplateManager
                    logger.info(f"Template manager type: {type(_template_manager)}")
                    if isinstance(_template_manager, TemplateManager):
                        logger.info("Attempting to render templates with deep_render...")
                        
                        # Debug: Check what's actually in the template manager context
                        logger.info(f"Template manager has {len(_template_manager.context)} context items")
                        
                        # Log key context items for debugging
                        important_keys = ['$item', 'item', 'input_text', 'translate', 'validate_translation']
                        for key in important_keys:
                            if key in _template_manager.context:
                                value = _template_manager.context[key]
                                if isinstance(value, str):
                                    logger.info(f"  ✓ {key} = '{value[:50]}...' (str)")
                                else:
                                    logger.info(f"  ✓ {key} = {type(value).__name__}")
                            else:
                                logger.warning(f"  ✗ {key} NOT in context!")
                        
                        # Check first few template variables
                        import re
                        template_vars = re.findall(r'{{\s*(\w+(?:\.\w+)*)', content[:500])
                        logger.info(f"First few template variables found: {template_vars[:5]}")
                        
                        # Use deep_render to handle complex nested templates
                        rendered = _template_manager.deep_render(content)
                        logger.info(f"Template rendering successful. Original had templates: {('{{' in content)}, Rendered has templates: {('{{' in rendered)}")
                        
                        # Check if rendering actually worked
                        if rendered == content and '{{' in content:
                            logger.warning("Template rendering returned original content unchanged!")
                            logger.warning(f"First 200 chars of content: {content[:200]}")
                        else:
                            logger.info(f"Templates rendered: {len(content)} chars -> {len(rendered)} chars")
                        
                        content = rendered
                    else:
                        logger.warning(f"Template manager is not TemplateManager type: {type(_template_manager)}")
                except Exception as e:
                    # If rendering fails, log but continue with original content
                    logger.error(f"Failed to render templates in content: {e}", exc_info=True)

        path_obj.write_text(content, encoding="utf-8")

        return {"action": "write", "path": path, "size": len(content), "success": True}

    async def _copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy file or directory."""
        source_path = Path(source)
        dest_path = Path(destination)

        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        if source_path.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        else:
            shutil.copytree(source, destination, dirs_exist_ok=True)

        return {
            "action": "copy",
            "source": source,
            "destination": destination,
            "success": True,
        }

    async def _move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move file or directory."""
        shutil.move(source, destination)

        return {
            "action": "move",
            "source": source,
            "destination": destination,
            "success": True,
        }

    async def _delete_file(self, path: str) -> Dict[str, Any]:
        """Delete file or directory."""
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path_obj.is_file():
            path_obj.unlink()
        else:
            shutil.rmtree(path)

        return {"action": "delete", "path": path, "success": True}

    async def _list_directory(self, path: str) -> Dict[str, Any]:
        """List directory contents."""
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path_obj.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        items = []
        for item in path_obj.iterdir():
            items.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )

        return {
            "action": "list",
            "path": path,
            "items": items,
            "count": len(items),
            "success": True,
        }
