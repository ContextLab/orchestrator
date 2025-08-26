"""Comprehensive pipeline validation and quality assessment."""

import logging
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PipelineValidator:
    """
    Comprehensive pipeline validation system.
    
    Provides detailed validation of:
    - YAML structure and syntax
    - Task definitions and dependencies
    - Template syntax and variables
    - Model specifications
    - Input/output definitions
    - Best practices compliance
    """
    
    def __init__(self):
        """Initialize pipeline validator."""
        self.validation_rules = self._init_validation_rules()
        self.best_practices = self._init_best_practices()
        
    def validate_pipeline_file(self, pipeline_path: Path) -> Dict[str, Any]:
        """
        Validate a pipeline YAML file comprehensively.
        
        Args:
            pipeline_path: Path to pipeline YAML file
            
        Returns:
            Dict[str, Any]: Validation results with scores and issues
        """
        results = {
            "valid": False,
            "score": 0.0,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "structure_valid": False,
            "syntax_valid": False,
            "templates_valid": False,
            "dependencies_valid": False,
            "best_practices_score": 0.0
        }
        
        try:
            # Read and parse YAML
            with open(pipeline_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pipeline_data = yaml.safe_load(content)
            
            # Run validation checks
            structure_results = self._validate_structure(pipeline_data)
            syntax_results = self._validate_syntax(content, pipeline_data)
            template_results = self._validate_templates(content, pipeline_data)
            dependency_results = self._validate_dependencies(pipeline_data)
            best_practices_results = self._validate_best_practices(content, pipeline_data)
            
            # Consolidate results
            results.update({
                "structure_valid": structure_results["valid"],
                "syntax_valid": syntax_results["valid"],
                "templates_valid": template_results["valid"],
                "dependencies_valid": dependency_results["valid"],
                "best_practices_score": best_practices_results["score"]
            })
            
            # Collect all issues
            for check_results in [structure_results, syntax_results, template_results, 
                                dependency_results, best_practices_results]:
                results["issues"].extend(check_results.get("issues", []))
                results["warnings"].extend(check_results.get("warnings", []))
                results["suggestions"].extend(check_results.get("suggestions", []))
            
            # Calculate overall validity and score
            results["valid"] = all([
                results["structure_valid"],
                results["syntax_valid"], 
                results["templates_valid"],
                results["dependencies_valid"]
            ])
            
            # Calculate composite score (0-100)
            component_scores = [
                40 if results["structure_valid"] else 0,
                20 if results["syntax_valid"] else 0,
                20 if results["templates_valid"] else 0,
                10 if results["dependencies_valid"] else 0,
                results["best_practices_score"] * 10  # Convert 0-1 to 0-10
            ]
            results["score"] = sum(component_scores)
            
        except yaml.YAMLError as e:
            results["issues"].append(f"YAML parsing error: {e}")
        except FileNotFoundError:
            results["issues"].append(f"Pipeline file not found: {pipeline_path}")
        except Exception as e:
            results["issues"].append(f"Validation error: {e}")
            
        return results
    
    def _validate_structure(self, data: Any) -> Dict[str, Any]:
        """Validate pipeline structure."""
        results = {"valid": True, "issues": [], "warnings": [], "suggestions": []}
        
        if not isinstance(data, dict):
            results["issues"].append("Pipeline must be a YAML dictionary")
            results["valid"] = False
            return results
        
        # Check required top-level fields
        required_fields = ["tasks"]
        for field in required_fields:
            if field not in data:
                results["issues"].append(f"Missing required field: {field}")
                results["valid"] = False
        
        # Check recommended fields
        recommended_fields = ["name", "description"]
        for field in recommended_fields:
            if field not in data:
                results["suggestions"].append(f"Consider adding {field} for better documentation")
        
        # Validate tasks structure
        if "tasks" in data:
            tasks_result = self._validate_tasks_structure(data["tasks"])
            results["issues"].extend(tasks_result["issues"])
            results["warnings"].extend(tasks_result["warnings"])
            if not tasks_result["valid"]:
                results["valid"] = False
        
        # Validate outputs if present
        if "outputs" in data:
            outputs_result = self._validate_outputs_structure(data["outputs"])
            results["issues"].extend(outputs_result["issues"])
            results["warnings"].extend(outputs_result["warnings"])
        
        return results
    
    def _validate_tasks_structure(self, tasks: Any) -> Dict[str, Any]:
        """Validate tasks structure."""
        results = {"valid": True, "issues": [], "warnings": []}
        
        if not isinstance(tasks, list):
            results["issues"].append("Tasks must be a list")
            results["valid"] = False
            return results
        
        if len(tasks) == 0:
            results["issues"].append("Pipeline must have at least one task")
            results["valid"] = False
            return results
        
        task_names = set()
        
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                results["issues"].append(f"Task {i} must be a dictionary")
                results["valid"] = False
                continue
            
            # Check required task fields
            required_task_fields = ["name", "type"]
            for field in required_task_fields:
                if field not in task:
                    results["issues"].append(f"Task {i} missing required field: {field}")
                    results["valid"] = False
            
            # Check for duplicate task names
            if "name" in task:
                name = task["name"]
                if name in task_names:
                    results["issues"].append(f"Duplicate task name: {name}")
                    results["valid"] = False
                task_names.add(name)
            
            # Validate task type
            if "type" in task:
                task_type = task["type"]
                valid_types = [
                    "llm", "chat", "completion", "image", "vision", "audio",
                    "web_search", "file_read", "file_write", "save", "load",
                    "for_each", "while", "until", "if", "condition", "parallel"
                ]
                if task_type not in valid_types:
                    results["warnings"].append(f"Task {i} has uncommon type: {task_type}")
            
            # Check for model specification in LLM tasks
            if task.get("type") in ["llm", "chat", "completion"]:
                if "model" not in task:
                    results["warnings"].append(f"LLM task '{task.get('name', i)}' should specify a model")
        
        return results
    
    def _validate_outputs_structure(self, outputs: Any) -> Dict[str, Any]:
        """Validate outputs structure."""
        results = {"valid": True, "issues": [], "warnings": []}
        
        if not isinstance(outputs, dict):
            results["issues"].append("Outputs must be a dictionary")
            results["valid"] = False
            return results
        
        if len(outputs) == 0:
            results["warnings"].append("Pipeline has no defined outputs")
        
        return results
    
    def _validate_syntax(self, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate YAML syntax and formatting."""
        results = {"valid": True, "issues": [], "warnings": [], "suggestions": []}
        
        # Check for common YAML issues
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for tabs (should use spaces)
            if '\t' in line:
                results["warnings"].append(f"Line {i}: Use spaces instead of tabs for indentation")
            
            # Check for trailing whitespace
            if line.endswith(' '):
                results["suggestions"].append(f"Line {i}: Remove trailing whitespace")
            
            # Check for very long lines
            if len(line) > 120:
                results["suggestions"].append(f"Line {i}: Consider breaking long line ({len(line)} chars)")
        
        # Check overall formatting
        if content.count('\n\n\n') > 0:
            results["suggestions"].append("Remove excessive blank lines")
        
        return results
    
    def _validate_templates(self, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template usage and syntax."""
        results = {"valid": True, "issues": [], "warnings": [], "suggestions": []}
        
        # Find all template expressions
        template_pattern = r'\{\{([^}]+)\}\}'
        templates = re.findall(template_pattern, content)
        
        # Check for balanced braces
        open_braces = content.count('{{')
        close_braces = content.count('}}')
        
        if open_braces != close_braces:
            results["issues"].append(f"Unbalanced template braces: {open_braces} {{ vs {close_braces} }}")
            results["valid"] = False
        
        # Analyze template expressions
        used_variables = set()
        
        for template in templates:
            template_clean = template.strip()
            used_variables.add(template_clean)
            
            # Check for problematic template syntax
            if template_clean == '':
                results["issues"].append("Empty template expression: {{}}")
                results["valid"] = False
            
            # Check for nested braces
            if '{{' in template_clean or '}}' in template_clean:
                results["issues"].append(f"Nested template braces: {{{template}}}")
                results["valid"] = False
            
            # Check for common Jinja2 filters and functions
            if '|' in template_clean:
                # This is likely a Jinja2 filter, which is valid
                pass
            elif '.' in template_clean:
                # This is likely accessing object properties
                pass
            elif template_clean in ['item', 'index']:
                # Common loop variables
                pass
        
        # Check for unused common variables
        common_vars = self._extract_likely_inputs(data)
        unused_vars = common_vars - used_variables
        
        if unused_vars:
            results["suggestions"].append(f"Potentially unused input variables: {', '.join(unused_vars)}")
        
        return results
    
    def _validate_dependencies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task dependencies."""
        results = {"valid": True, "issues": [], "warnings": []}
        
        tasks = data.get("tasks", [])
        task_names = {task.get("name") for task in tasks if isinstance(task, dict) and "name" in task}
        
        # Check dependency references
        for task in tasks:
            if not isinstance(task, dict):
                continue
            
            dependencies = task.get("dependencies", [])
            
            if dependencies:
                if not isinstance(dependencies, list):
                    results["issues"].append(f"Task '{task.get('name')}' dependencies must be a list")
                    results["valid"] = False
                    continue
                
                for dep in dependencies:
                    if dep not in task_names:
                        results["issues"].append(
                            f"Task '{task.get('name')}' depends on unknown task: {dep}"
                        )
                        results["valid"] = False
        
        # Check for circular dependencies (basic check)
        if len(task_names) > 1:
            circular_result = self._check_circular_dependencies(tasks)
            results["issues"].extend(circular_result["issues"])
            if not circular_result["valid"]:
                results["valid"] = False
        
        return results
    
    def _validate_best_practices(self, content: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate best practices compliance."""
        results = {"valid": True, "issues": [], "warnings": [], "suggestions": [], "score": 1.0}
        
        score_deductions = 0.0
        
        # Check for documentation
        if "description" not in data:
            results["suggestions"].append("Add pipeline description for better documentation")
            score_deductions += 0.1
        
        if "name" not in data:
            results["suggestions"].append("Add pipeline name for identification")
            score_deductions += 0.1
        
        # Check task naming
        tasks = data.get("tasks", [])
        for task in tasks:
            if isinstance(task, dict) and "name" in task:
                name = task["name"]
                if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
                    results["suggestions"].append(f"Task name '{name}' should use valid identifier format")
                    score_deductions += 0.05
        
        # Check for error handling
        has_error_handling = any(
            isinstance(task, dict) and (
                "on_error" in task or 
                "try_catch" in task or
                "error_action" in task
            )
            for task in tasks
        )
        
        if len(tasks) > 2 and not has_error_handling:
            results["suggestions"].append("Consider adding error handling for robust pipelines")
            score_deductions += 0.1
        
        # Check for appropriate model selection
        llm_tasks = [
            task for task in tasks 
            if isinstance(task, dict) and task.get("type") in ["llm", "chat", "completion"]
        ]
        
        for task in llm_tasks:
            if "model" in task:
                model = task["model"]
                if isinstance(model, str) and "gpt-4" in model.lower():
                    results["suggestions"].append(
                        f"Task '{task.get('name')}' uses expensive GPT-4 - consider if necessary"
                    )
                    score_deductions += 0.05
        
        # Check for reasonable pipeline complexity
        if len(tasks) > 20:
            results["warnings"].append("Very large pipeline - consider breaking into smaller modules")
            score_deductions += 0.1
        
        # Check for output definition
        if "outputs" not in data or not data["outputs"]:
            results["suggestions"].append("Define pipeline outputs for better usability")
            score_deductions += 0.1
        
        results["score"] = max(0.0, 1.0 - score_deductions)
        
        return results
    
    def _extract_likely_inputs(self, data: Dict[str, Any]) -> Set[str]:
        """Extract likely input variable names from pipeline data."""
        inputs = set()
        
        # Look for inputs section
        if "inputs" in data:
            inputs_section = data["inputs"]
            if isinstance(inputs_section, dict):
                inputs.update(inputs_section.keys())
        
        # Common input variable names
        common_inputs = {
            'input_file', 'data_file', 'topic', 'query', 'model', 
            'threshold', 'items', 'description', 'prompt'
        }
        inputs.update(common_inputs)
        
        return inputs
    
    def _check_circular_dependencies(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for circular dependencies in tasks."""
        results = {"valid": True, "issues": []}
        
        # Build dependency graph
        dependencies = {}
        for task in tasks:
            if isinstance(task, dict) and "name" in task:
                name = task["name"]
                deps = task.get("dependencies", [])
                dependencies[name] = deps
        
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_name in dependencies:
            if task_name not in visited:
                if has_cycle(task_name):
                    results["issues"].append("Circular dependency detected in task dependencies")
                    results["valid"] = False
                    break
        
        return results
    
    def _init_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules."""
        return {
            "required_fields": ["tasks"],
            "recommended_fields": ["name", "description", "outputs"],
            "valid_task_types": [
                "llm", "chat", "completion", "image", "vision", "audio",
                "web_search", "file_read", "file_write", "save", "load",
                "for_each", "while", "until", "if", "condition", "parallel"
            ],
            "max_line_length": 120,
            "max_tasks": 50
        }
    
    def _init_best_practices(self) -> Dict[str, Any]:
        """Initialize best practices rules."""
        return {
            "require_documentation": True,
            "require_error_handling": True,
            "prefer_specific_models": True,
            "limit_pipeline_complexity": True,
            "require_outputs": True
        }