#!/usr/bin/env python3
"""Audit all pipeline YAML files for common issues."""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.orchestrator.models import ModelRegistry
from src.orchestrator.tools.base import default_registry, register_default_tools

def load_pipeline(filepath: Path) -> Dict[str, Any]:
    """Load a pipeline YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def get_available_tools() -> Set[str]:
    """Get all available tool names from the registry."""
    # Make sure tools are registered
    register_default_tools()
    return set(default_registry.list_tools())

def get_available_models() -> Set[str]:
    """Get all available model names from the registry."""
    registry = ModelRegistry()
    return set(registry.list_models())

def check_tool_references(pipeline: Dict[str, Any], available_tools: Set[str]) -> List[str]:
    """Check for invalid tool references in a pipeline."""
    issues = []
    
    def check_step(step: Dict[str, Any], step_path: str = ""):
        if 'tool' in step:
            tool_name = step['tool']
            if tool_name not in available_tools:
                issues.append(f"{step_path}: Unknown tool '{tool_name}'")
        
        # Check nested steps
        if 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step(substep, f"{step_path}.steps[{i}]")
        
        # Check for_each steps
        if 'for_each' in step and 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step(substep, f"{step_path}.for_each.steps[{i}]")
        
        # Check while_loop steps
        if 'while' in step and 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step(substep, f"{step_path}.while.steps[{i}]")
    
    # Check all steps
    if 'steps' in pipeline:
        for i, step in enumerate(pipeline['steps']):
            check_step(step, f"steps[{i}]")
    
    return issues

def check_context_usage(pipeline: Dict[str, Any]) -> List[str]:
    """Check for proper context usage in templates."""
    issues = []
    
    def check_value(value: Any, path: str) -> None:
        if isinstance(value, str):
            # Check for common template issues
            if '{{' in value or '{%' in value:
                # Check for undefined common variables
                undefined_patterns = [
                    'execution.timestamp',  # Should be available
                    'pipeline.',  # Pipeline context should be available
                    'inputs.',  # Input parameters should be defined
                ]
                
                # Check for loop variables outside loops
                if '$item' in value and '.for_each' not in path:
                    issues.append(f"{path}: Using $item outside of for_each loop")
                if '$index' in value and '.for_each' not in path:
                    issues.append(f"{path}: Using $index outside of for_each loop")
                
                # Check for step references
                import re
                step_refs = re.findall(r'{{\s*(\w+)\.\w+\s*}}', value)
                # Note: We'd need to track defined step IDs to validate these properly
                
        elif isinstance(value, dict):
            for k, v in value.items():
                check_value(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                check_value(v, f"{path}[{i}]")
    
    check_value(pipeline, "pipeline")
    return issues

def check_action_references(pipeline: Dict[str, Any]) -> List[str]:
    """Check for undefined actions in steps."""
    issues = []
    
    def check_step(step: Dict[str, Any], step_path: str = ""):
        if 'action' in step and 'tool' not in step:
            # Action without tool implies it's a model action
            action = step['action']
            known_model_actions = {
                'generate_text', 'analyze_text', 'classify_text', 
                'extract_entities', 'summarize_text', 'translate_text',
                'answer_question', 'generate_code', 'explain_code'
            }
            if action not in known_model_actions:
                issues.append(f"{step_path}: Unknown action '{action}' (no tool specified)")
        
        # Check nested steps
        if 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step(substep, f"{step_path}.steps[{i}]")
        
        # Check for_each steps
        if 'for_each' in step and 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step(substep, f"{step_path}.for_each.steps[{i}]")
        
        # Check while_loop steps
        if 'while' in step and 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step(substep, f"{step_path}.while.steps[{i}]")
    
    # Check all steps
    if 'steps' in pipeline:
        for i, step in enumerate(pipeline['steps']):
            check_step(step, f"steps[{i}]")
    
    return issues

def check_dependencies(pipeline: Dict[str, Any]) -> List[str]:
    """Check for invalid dependency references."""
    issues = []
    defined_steps = set()
    
    def collect_step_ids(step: Dict[str, Any]):
        if 'id' in step:
            defined_steps.add(step['id'])
        
        # Check nested steps
        if 'steps' in step:
            for substep in step['steps']:
                collect_step_ids(substep)
        
        # Check for_each steps
        if 'for_each' in step and 'steps' in step:
            for substep in step['steps']:
                collect_step_ids(substep)
        
        # Check while_loop steps
        if 'while' in step and 'steps' in step:
            for substep in step['steps']:
                collect_step_ids(substep)
    
    def check_step_deps(step: Dict[str, Any], step_path: str = ""):
        if 'dependencies' in step:
            for dep in step['dependencies']:
                if dep not in defined_steps:
                    issues.append(f"{step_path}: Unknown dependency '{dep}'")
        
        # Check nested steps
        if 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step_deps(substep, f"{step_path}.steps[{i}]")
        
        # Check for_each steps
        if 'for_each' in step and 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step_deps(substep, f"{step_path}.for_each.steps[{i}]")
        
        # Check while_loop steps
        if 'while' in step and 'steps' in step:
            for i, substep in enumerate(step['steps']):
                check_step_deps(substep, f"{step_path}.while.steps[{i}]")
    
    # First collect all step IDs
    if 'steps' in pipeline:
        for step in pipeline['steps']:
            collect_step_ids(step)
    
    # Then check dependencies
    if 'steps' in pipeline:
        for i, step in enumerate(pipeline['steps']):
            check_step_deps(step, f"steps[{i}]")
    
    return issues

def audit_pipeline(filepath: Path) -> Dict[str, List[str]]:
    """Audit a single pipeline file."""
    try:
        pipeline = load_pipeline(filepath)
        available_tools = get_available_tools()
        
        return {
            'tool_issues': check_tool_references(pipeline, available_tools),
            'context_issues': check_context_usage(pipeline),
            'action_issues': check_action_references(pipeline),
            'dependency_issues': check_dependencies(pipeline),
        }
    except Exception as e:
        return {
            'parse_error': [str(e)]
        }

def main():
    """Audit all pipelines in the examples directory."""
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    print("Pipeline Audit Report")
    print("=" * 80)
    print()
    
    all_issues = defaultdict(lambda: defaultdict(list))
    
    for yaml_file in sorted(examples_dir.glob('*.yaml')):
        issues = audit_pipeline(yaml_file)
        
        has_issues = any(issues.values())
        if has_issues:
            print(f"Pipeline: {yaml_file.name}")
            print("-" * 40)
            
            for issue_type, issue_list in issues.items():
                if issue_list:
                    print(f"\n{issue_type.replace('_', ' ').title()}:")
                    for issue in issue_list:
                        print(f"  - {issue}")
                        all_issues[issue_type][yaml_file.name].append(issue)
            
            print()
    
    # Summary
    print("\nSummary")
    print("=" * 80)
    
    total_pipelines = len(list(examples_dir.glob('*.yaml')))
    pipelines_with_issues = len(set(
        pipeline for issues in all_issues.values() 
        for pipeline in issues.keys()
    ))
    
    print(f"Total pipelines: {total_pipelines}")
    print(f"Pipelines with issues: {pipelines_with_issues}")
    print(f"Pipelines without issues: {total_pipelines - pipelines_with_issues}")
    
    print("\nIssue counts by type:")
    for issue_type, pipelines in all_issues.items():
        total = sum(len(issues) for issues in pipelines.values())
        print(f"  - {issue_type.replace('_', ' ').title()}: {total} issues in {len(pipelines)} pipelines")

if __name__ == '__main__':
    main()