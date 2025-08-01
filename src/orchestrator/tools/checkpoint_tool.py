"""Checkpoint inspection and extraction tool."""

import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import yaml


class CheckpointTool:
    """Tool for inspecting and extracting information from checkpoint files."""
    
    def __init__(self):
        """Initialize the checkpoint tool."""
        self.name = "checkpoint"
        self.description = "Inspect and extract information from pipeline execution checkpoints"
    
    async def execute(
        self,
        action: str,
        checkpoint_file: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        output_format: str = "markdown",
        output_file: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute checkpoint operations.
        
        Args:
            action: Action to perform (inspect, extract, list)
            checkpoint_file: Specific checkpoint file to analyze
            pipeline_name: Filter checkpoints by pipeline name
            output_format: Output format (markdown, yaml, json)
            output_file: Optional file to save output to
            
        Returns:
            Operation result
        """
        if action == "list":
            return self._list_checkpoints(pipeline_name)
        elif action == "inspect":
            return await self._inspect_checkpoint(checkpoint_file, pipeline_name)
        elif action == "extract":
            return await self._extract_checkpoint(
                checkpoint_file, pipeline_name, output_format, output_file
            )
        else:
            return {
                "error": f"Unknown action: {action}",
                "supported_actions": ["list", "inspect", "extract"]
            }
    
    def _list_checkpoints(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """List available checkpoints."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return {"error": "No checkpoints directory found", "checkpoints": []}
        
        checkpoints = list(checkpoint_dir.glob("*.json"))
        if pipeline_name:
            checkpoints = [c for c in checkpoints if pipeline_name in c.name]
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        checkpoint_list = []
        for cp in checkpoints[:20]:  # Limit to 20 most recent
            parts = cp.stem.split("_")
            pipeline_id = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
            timestamp = parts[-1] if parts else "unknown"
            
            checkpoint_list.append({
                "file": cp.name,
                "pipeline": pipeline_id,
                "timestamp": datetime.fromtimestamp(int(timestamp)).isoformat() if timestamp.isdigit() else timestamp,
                "size": cp.stat().st_size
            })
        
        return {
            "total": len(checkpoints),
            "showing": len(checkpoint_list),
            "checkpoints": checkpoint_list
        }
    
    def _decompress_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress checkpoint data if needed."""
        if checkpoint_data.get('state', {}).get('compressed'):
            compressed_data = checkpoint_data['state']['data']
            compressed_bytes = bytes.fromhex(compressed_data)
            decompressed_bytes = gzip.decompress(compressed_bytes)
            state_data = json.loads(decompressed_bytes.decode('utf-8'))
            checkpoint_data['state']['data'] = state_data
            checkpoint_data['state']['compressed'] = False
        
        return checkpoint_data
    
    async def _inspect_checkpoint(
        self, 
        checkpoint_file: Optional[str] = None,
        pipeline_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Inspect a checkpoint file and return summary information."""
        # Find checkpoint file
        if not checkpoint_file:
            checkpoint_list = self._list_checkpoints(pipeline_name)
            if not checkpoint_list.get("checkpoints"):
                return {"error": "No checkpoints found"}
            checkpoint_file = checkpoint_list["checkpoints"][0]["file"]
        
        checkpoint_path = Path("checkpoints") / checkpoint_file
        if not checkpoint_path.exists():
            return {"error": f"Checkpoint file not found: {checkpoint_file}"}
        
        # Load and decompress
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        checkpoint = self._decompress_checkpoint(checkpoint)
        
        # Extract summary information
        state = checkpoint.get('state', {}).get('data', {})
        tasks = state.get('tasks', {})
        
        task_summary = {}
        for task_id, task_data in tasks.items():
            status = task_data.get('status', 'unknown')
            task_summary[task_id] = {
                "status": status,
                "has_result": 'result' in task_data,
                "has_error": 'error' in task_data
            }
        
        metadata = checkpoint.get('metadata', {})
        
        return {
            "checkpoint_file": checkpoint_file,
            "pipeline_id": metadata.get('pipeline_id', 'unknown'),
            "execution_id": checkpoint.get('execution_id', 'unknown'),
            "timestamp": datetime.fromtimestamp(checkpoint.get('timestamp', 0)).isoformat(),
            "total_tasks": len(tasks),
            "task_summary": task_summary,
            "pipeline_context": metadata.get('pipeline_context', {}),
            "has_previous_results": 'previous_results' in metadata,
            "compressed": checkpoint.get('state', {}).get('compressed', False),
            "size": {
                "original": checkpoint.get('state', {}).get('original_size', 0),
                "compressed": checkpoint.get('state', {}).get('compressed_size', 0)
            }
        }
    
    async def _extract_checkpoint(
        self,
        checkpoint_file: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        output_format: str = "markdown",
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract full checkpoint information in human-readable format."""
        # Find checkpoint file
        if not checkpoint_file:
            checkpoint_list = self._list_checkpoints(pipeline_name)
            if not checkpoint_list.get("checkpoints"):
                return {"error": "No checkpoints found"}
            checkpoint_file = checkpoint_list["checkpoints"][0]["file"]
        
        checkpoint_path = Path("checkpoints") / checkpoint_file
        if not checkpoint_path.exists():
            return {"error": f"Checkpoint file not found: {checkpoint_file}"}
        
        # Load and decompress
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        checkpoint = self._decompress_checkpoint(checkpoint)
        
        # Generate output based on format
        if output_format == "markdown":
            content = self._generate_markdown_report(checkpoint)
        elif output_format == "yaml":
            content = self._generate_yaml_report(checkpoint)
        elif output_format == "json":
            content = json.dumps(checkpoint, indent=2, default=str)
        else:
            return {"error": f"Unsupported format: {output_format}"}
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            return {
                "success": True,
                "output_file": str(output_path),
                "format": output_format,
                "size": len(content)
            }
        else:
            return {
                "success": True,
                "format": output_format,
                "content": content
            }
    
    def _generate_markdown_report(self, checkpoint: Dict[str, Any]) -> str:
        """Generate a markdown report from checkpoint data."""
        lines = []
        
        # Header
        metadata = checkpoint.get('metadata', {})
        lines.append(f"# Pipeline Execution Report")
        lines.append(f"\n**Pipeline:** {metadata.get('pipeline_id', 'Unknown')}")
        lines.append(f"**Execution ID:** {checkpoint.get('execution_id', 'Unknown')}")
        lines.append(f"**Timestamp:** {datetime.fromtimestamp(checkpoint.get('timestamp', 0))}")
        
        # Pipeline Context
        if 'pipeline_context' in metadata:
            lines.append(f"\n## Pipeline Context")
            for key, value in metadata['pipeline_context'].items():
                if key != 'all_step_ids' and not isinstance(value, (dict, list)):
                    lines.append(f"- **{key}:** {value}")
        
        # Task Execution Summary
        state = checkpoint.get('state', {}).get('data', {})
        tasks = state.get('tasks', {})
        
        lines.append(f"\n## Task Execution Summary")
        lines.append(f"\nTotal tasks: {len(tasks)}")
        
        # Group by status
        status_groups = {}
        for task_id, task_data in tasks.items():
            status = task_data.get('status', 'unknown')
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(task_id)
        
        for status, task_ids in status_groups.items():
            lines.append(f"\n### {status.title()} ({len(task_ids)})")
            for task_id in task_ids:
                lines.append(f"- `{task_id}`")
        
        # Detailed Task Results
        lines.append(f"\n## Task Details")
        
        for task_id, task_data in tasks.items():
            lines.append(f"\n### {task_id}")
            lines.append(f"**Status:** {task_data.get('status', 'unknown')}")
            lines.append(f"**Action:** {task_data.get('action', 'unknown')}")
            
            if 'error' in task_data:
                lines.append(f"**Error:** {task_data['error']}")
            
            if 'result' in task_data:
                result = task_data['result']
                if isinstance(result, dict):
                    lines.append(f"**Result:**")
                    for key, value in result.items():
                        if key == 'content' and isinstance(value, str) and len(value) > 200:
                            lines.append(f"  - {key}: [content with {len(value)} characters]")
                        elif isinstance(value, (list, dict)):
                            lines.append(f"  - {key}: [{type(value).__name__} with {len(value)} items]")
                        else:
                            lines.append(f"  - {key}: {value}")
                elif isinstance(result, str):
                    if len(result) > 200:
                        lines.append(f"**Result:** {result[:200]}...")
                    else:
                        lines.append(f"**Result:** {result}")
                else:
                    lines.append(f"**Result:** {result}")
            
            # Show parameters if they contain templates
            params = task_data.get('parameters', {})
            if params:
                has_templates = any('{{' in str(v) for v in params.values() if isinstance(v, str))
                if has_templates:
                    lines.append(f"**Parameters with templates:**")
                    for key, value in params.items():
                        if isinstance(value, str) and '{{' in value:
                            preview = value[:100] + "..." if len(value) > 100 else value
                            lines.append(f"  - {key}: `{preview}`")
        
        # Previous Results Summary
        if 'previous_results' in metadata:
            lines.append(f"\n## Available Step Results")
            for step_id, result in metadata['previous_results'].items():
                if isinstance(result, dict):
                    lines.append(f"- **{step_id}:** {list(result.keys())}")
                else:
                    lines.append(f"- **{step_id}:** [{type(result).__name__}]")
        
        return "\n".join(lines)
    
    def _generate_yaml_report(self, checkpoint: Dict[str, Any]) -> str:
        """Generate a YAML report from checkpoint data."""
        # Create a simplified structure for YAML
        report = {
            "pipeline_execution": {
                "pipeline_id": checkpoint.get('metadata', {}).get('pipeline_id'),
                "execution_id": checkpoint.get('execution_id'),
                "timestamp": datetime.fromtimestamp(checkpoint.get('timestamp', 0)).isoformat(),
                "context": checkpoint.get('metadata', {}).get('pipeline_context', {})
            },
            "tasks": {}
        }
        
        state = checkpoint.get('state', {}).get('data', {})
        tasks = state.get('tasks', {})
        
        for task_id, task_data in tasks.items():
            task_info = {
                "status": task_data.get('status'),
                "action": task_data.get('action')
            }
            
            if 'error' in task_data:
                task_info['error'] = task_data['error']
            
            if 'result' in task_data:
                result = task_data['result']
                if isinstance(result, dict):
                    # Simplify large content
                    simplified_result = {}
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 100:
                            simplified_result[key] = f"[{len(value)} chars]"
                        elif isinstance(value, list) and len(value) > 5:
                            simplified_result[key] = f"[{len(value)} items]"
                        else:
                            simplified_result[key] = value
                    task_info['result'] = simplified_result
                elif isinstance(result, str) and len(result) > 100:
                    task_info['result'] = f"[{len(result)} chars]"
                else:
                    task_info['result'] = result
            
            report['tasks'][task_id] = task_info
        
        return yaml.dump(report, default_flow_style=False, sort_keys=False)


# Register the tool
def register_tool():
    """Register the checkpoint tool with the tool registry."""
    from ..core.tool_registry import get_tool_registry
    
    tool = CheckpointTool()
    registry = get_tool_registry()
    registry.register("checkpoint", tool)