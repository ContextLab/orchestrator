"""
Output visualization and analysis tools for pipeline execution.
Provides visual representation of output dependencies and metadata.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.output_tracker import OutputTracker


class OutputVisualizer:
    """
    Advanced visualization tools for output tracking and dependencies.
    
    Provides various visualization formats including:
    - Dependency graphs
    - Output flow diagrams
    - Validation reports
    - Interactive HTML dashboards
    """
    
    def __init__(self, output_tracker: OutputTracker):
        """Initialize with output tracker."""
        self.output_tracker = output_tracker
    
    def generate_dependency_graph(self, format: str = "mermaid") -> str:
        """Generate dependency graph in specified format."""
        if format == "mermaid":
            return self._generate_mermaid_graph()
        elif format == "dot":
            return self._generate_dot_graph()
        elif format == "json":
            return self._generate_json_graph()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_mermaid_graph(self) -> str:
        """Generate Mermaid diagram for output dependencies."""
        lines = ["graph TD"]
        
        # Add nodes
        for task_id in set(self.output_tracker.metadata.keys()) | set(self.output_tracker.outputs.keys()):
            metadata = self.output_tracker.metadata.get(task_id)
            output_info = self.output_tracker.outputs.get(task_id)
            
            # Determine node style based on status
            if output_info:
                if output_info.validation_status == "valid":
                    style = ":::success"
                elif output_info.validation_status == "invalid":
                    style = ":::error"
                else:
                    style = ":::pending"
            else:
                style = ":::pending"
            
            # Create node label
            if metadata and metadata.produces:
                label = f"{task_id}\\n[{metadata.produces}]"
            else:
                label = task_id
            
            lines.append(f'  {task_id}["{label}"]{style}')
        
        # Add edges (dependencies)
        for task_id, refs in self.output_tracker.references.items():
            for ref in refs:
                lines.append(f"  {ref.task_id} --> {task_id}")
        
        # Add styles
        lines.extend([
            "  classDef success fill:#d4edda,stroke:#28a745,stroke-width:2px",
            "  classDef error fill:#f8d7da,stroke:#dc3545,stroke-width:2px",
            "  classDef pending fill:#fff3cd,stroke:#ffc107,stroke-width:2px"
        ])
        
        return "\\n".join(lines)
    
    def _generate_dot_graph(self) -> str:
        """Generate DOT/Graphviz format for output dependencies."""
        lines = ["digraph OutputDependencies {"]
        lines.append("  rankdir=TD;")
        lines.append("  node [shape=box, style=rounded];")
        
        # Add nodes
        for task_id in set(self.output_tracker.metadata.keys()) | set(self.output_tracker.outputs.keys()):
            metadata = self.output_tracker.metadata.get(task_id)
            output_info = self.output_tracker.outputs.get(task_id)
            
            # Determine color based on status
            if output_info:
                if output_info.validation_status == "valid":
                    color = "lightgreen"
                elif output_info.validation_status == "invalid":
                    color = "lightcoral"
                else:
                    color = "lightyellow"
            else:
                color = "lightgray"
            
            # Create label
            label = task_id
            if metadata and metadata.produces:
                label += f"\\n[{metadata.produces}]"
            if output_info and output_info.location:
                label += f"\\n{os.path.basename(output_info.location)}"
            
            lines.append(f'  "{task_id}" [label="{label}", fillcolor="{color}", style="filled"];')
        
        # Add edges
        for task_id, refs in self.output_tracker.references.items():
            for ref in refs:
                edge_label = ref.field if ref.field else "result"
                lines.append(f'  "{ref.task_id}" -> "{task_id}" [label="{edge_label}"];')
        
        lines.append("}")
        return "\\n".join(lines)
    
    def _generate_json_graph(self) -> str:
        """Generate JSON representation of output dependencies."""
        graph = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "total_tasks": len(set(self.output_tracker.metadata.keys()) | set(self.output_tracker.outputs.keys())),
                "total_outputs": len(self.output_tracker.outputs),
                "total_references": sum(len(refs) for refs in self.output_tracker.references.values()),
                "pipeline_id": self.output_tracker.pipeline_id
            }
        }
        
        # Add nodes
        for task_id in set(self.output_tracker.metadata.keys()) | set(self.output_tracker.outputs.keys()):
            metadata = self.output_tracker.metadata.get(task_id)
            output_info = self.output_tracker.outputs.get(task_id)
            
            node = {
                "id": task_id,
                "has_metadata": metadata is not None,
                "has_output": output_info is not None
            }
            
            if metadata:
                node["produces"] = metadata.produces
                node["location_template"] = metadata.location
                node["format"] = metadata.format
            
            if output_info:
                node["actual_location"] = output_info.location
                node["validation_status"] = output_info.validation_status
                node["file_size"] = output_info.file_size
                node["created_at"] = output_info.created_at.isoformat() if output_info.created_at else None
            
            graph["nodes"].append(node)
        
        # Add edges
        for task_id, refs in self.output_tracker.references.items():
            for ref in refs:
                edge = {
                    "source": ref.task_id,
                    "target": task_id,
                    "field": ref.field,
                    "has_default": ref.default_value is not None
                }
                graph["edges"].append(edge)
        
        return json.dumps(graph, indent=2)
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "summary": {
                "total_tasks": len(set(self.output_tracker.metadata.keys()) | set(self.output_tracker.outputs.keys())),
                "tasks_with_metadata": len(self.output_tracker.metadata),
                "tasks_with_outputs": len(self.output_tracker.outputs),
                "validation_errors": len(self.output_tracker.validation_errors),
                "consistency_score": self._calculate_consistency_score()
            },
            "validation_details": {
                "errors": self.output_tracker.validation_errors,
                "warnings": [],
                "recommendations": []
            },
            "task_analysis": {},
            "dependency_analysis": self._analyze_dependencies(),
            "file_analysis": self._analyze_file_outputs()
        }
        
        # Analyze each task
        for task_id in set(self.output_tracker.metadata.keys()) | set(self.output_tracker.outputs.keys()):
            analysis = self._analyze_task(task_id)
            report["task_analysis"][task_id] = analysis
            
            # Add warnings and recommendations
            if analysis.get("warnings"):
                report["validation_details"]["warnings"].extend(analysis["warnings"])
            if analysis.get("recommendations"):
                report["validation_details"]["recommendations"].extend(analysis["recommendations"])
        
        return report
    
    def _calculate_consistency_score(self) -> float:
        """Calculate overall consistency score (0-100)."""
        if not self.output_tracker.metadata:
            return 100.0
        
        total_checks = len(self.output_tracker.metadata)
        passed_checks = sum(1 for status in self.output_tracker.consistency_checks.values() if status)
        
        return (passed_checks / total_checks) * 100.0 if total_checks > 0 else 100.0
    
    def _analyze_task(self, task_id: str) -> Dict[str, Any]:
        """Analyze individual task for validation and recommendations."""
        metadata = self.output_tracker.metadata.get(task_id)
        output_info = self.output_tracker.outputs.get(task_id)
        
        analysis = {
            "task_id": task_id,
            "has_metadata": metadata is not None,
            "has_output": output_info is not None,
            "warnings": [],
            "recommendations": []
        }
        
        if metadata and not output_info:
            analysis["warnings"].append(f"Task {task_id} has metadata but no actual output")
            analysis["recommendations"].append(f"Ensure task {task_id} executes and produces output")
        
        if output_info and not metadata:
            analysis["warnings"].append(f"Task {task_id} has output but no metadata specification")
            analysis["recommendations"].append(f"Add output metadata specification for task {task_id}")
        
        if metadata and output_info:
            # Check consistency
            if metadata.produces and not output_info.output_type:
                analysis["warnings"].append(f"Task {task_id} metadata specifies produces but output has no type")
            
            if metadata.location and not output_info.location:
                analysis["warnings"].append(f"Task {task_id} metadata specifies location but output has no location")
            
            if output_info.validation_errors:
                analysis["warnings"].extend([f"Task {task_id}: {error}" for error in output_info.validation_errors])
        
        return analysis
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency structure."""
        dependency_analysis = {
            "total_references": sum(len(refs) for refs in self.output_tracker.references.values()),
            "circular_dependencies": [],
            "missing_dependencies": [],
            "dependency_depth": {}
        }
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            # Check dependencies
            for ref in self.output_tracker.references.get(task_id, []):
                if has_cycle(ref.task_id):
                    dependency_analysis["circular_dependencies"].append((ref.task_id, task_id))
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.output_tracker.references.keys():
            if task_id not in visited:
                has_cycle(task_id)
        
        # Check for missing dependencies
        all_tasks = set(self.output_tracker.metadata.keys()) | set(self.output_tracker.outputs.keys())
        for task_id, refs in self.output_tracker.references.items():
            for ref in refs:
                if ref.task_id not in all_tasks:
                    dependency_analysis["missing_dependencies"].append((task_id, ref.task_id))
        
        return dependency_analysis
    
    def _analyze_file_outputs(self) -> Dict[str, Any]:
        """Analyze file outputs."""
        file_analysis = {
            "total_files": 0,
            "existing_files": 0,
            "missing_files": 0,
            "total_size": 0,
            "by_format": {},
            "files": []
        }
        
        for task_id, output_info in self.output_tracker.outputs.items():
            if output_info.location and output_info.location.startswith(('.', '/')):
                file_info = {
                    "task_id": task_id,
                    "location": output_info.location,
                    "format": output_info.format,
                    "exists": os.path.exists(output_info.location) if output_info.location else False,
                    "size": None
                }
                
                file_analysis["total_files"] += 1
                
                if file_info["exists"]:
                    file_analysis["existing_files"] += 1
                    try:
                        size = os.path.getsize(output_info.location)
                        file_info["size"] = size
                        file_analysis["total_size"] += size
                    except OSError:
                        pass
                else:
                    file_analysis["missing_files"] += 1
                
                # Track by format
                format_key = output_info.format or "unknown"
                if format_key not in file_analysis["by_format"]:
                    file_analysis["by_format"][format_key] = {"count": 0, "size": 0}
                file_analysis["by_format"][format_key]["count"] += 1
                if file_info["size"]:
                    file_analysis["by_format"][format_key]["size"] += file_info["size"]
                
                file_analysis["files"].append(file_info)
        
        return file_analysis
    
    def generate_html_dashboard(self, output_path: str) -> str:
        """Generate interactive HTML dashboard."""
        validation_report = self.generate_validation_report()
        mermaid_graph = self._generate_mermaid_graph()
        
        html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Output Tracking Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .metric { background: white; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .section { margin-bottom: 30px; }
        .section h2 { border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .error { color: #dc3545; }
        .warning { color: #ffc107; }
        .success { color: #28a745; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .mermaid { text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Output Tracking Dashboard</h1>
        <p>Pipeline: {pipeline_id}</p>
        <p>Generated: {timestamp}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{total_tasks}</div>
            <div>Total Tasks</div>
        </div>
        <div class="metric">
            <div class="metric-value">{tasks_with_outputs}</div>
            <div>With Outputs</div>
        </div>
        <div class="metric">
            <div class="metric-value">{validation_errors}</div>
            <div>Validation Errors</div>
        </div>
        <div class="metric">
            <div class="metric-value">{consistency_score:.1f}%</div>
            <div>Consistency Score</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Dependency Graph</h2>
        <div class="mermaid">
            {mermaid_graph}
        </div>
    </div>
    
    <div class="section">
        <h2>Validation Report</h2>
        <pre>{validation_report}</pre>
    </div>
    
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>
        '''
        
        from datetime import datetime
        
        html_content = html_template.format(
            pipeline_id=self.output_tracker.pipeline_id or "Unknown",
            timestamp=datetime.now().isoformat(),
            total_tasks=validation_report["summary"]["total_tasks"],
            tasks_with_outputs=validation_report["summary"]["tasks_with_outputs"],
            validation_errors=validation_report["summary"]["validation_errors"],
            consistency_score=validation_report["summary"]["consistency_score"],
            mermaid_graph=mermaid_graph,
            validation_report=json.dumps(validation_report, indent=2)
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        metrics = {
            "timestamp": self.output_tracker.created_at.isoformat(),
            "pipeline_id": self.output_tracker.pipeline_id,
            "execution_id": self.output_tracker.execution_id,
            "summary": self.generate_validation_report()["summary"],
            "file_metrics": self._analyze_file_outputs(),
            "dependency_metrics": self._analyze_dependencies()
        }
        
        if format == "json":
            return json.dumps(metrics, indent=2)
        elif format == "csv":
            return self._metrics_to_csv(metrics)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _metrics_to_csv(self, metrics: Dict[str, Any]) -> str:
        """Convert metrics to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write summary metrics
        writer.writerow(["Metric", "Value"])
        for key, value in metrics["summary"].items():
            writer.writerow([key, value])
        
        return output.getvalue()