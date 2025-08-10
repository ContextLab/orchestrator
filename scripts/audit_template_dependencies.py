#!/usr/bin/env python3
"""
Pipeline Template Dependency Auditor

This script audits all pipelines to identify template rendering issues caused by
missing dependency declarations.

Usage:
    python scripts/audit_template_dependencies.py [--fix] [--verbose]
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class PipelineAuditor:
    """Audit pipelines for template dependency issues."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues_found = []
        self.pipelines_checked = 0
        
    def find_template_variables(self, text: str) -> Set[str]:
        """Find all template variables in text."""
        if not isinstance(text, str):
            return set()
        
        # Find {{ variable.attribute }} patterns
        pattern = r'{{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*}}'
        matches = re.findall(pattern, text)
        
        variables = set()
        for match in matches:
            # Skip execution metadata and pipeline parameters
            if match.startswith('execution.') or match in ['topic', 'query', 'max_results', 'output_path']:
                continue
            variables.add(match)
        
        return variables
    
    def extract_step_references(self, variables: Set[str]) -> Set[str]:
        """Extract step references from template variables."""
        step_refs = set()
        for var in variables:
            if '.' in var:
                # This looks like step_id.result or step_id.attribute
                step_id = var.split('.')[0]
                step_refs.add(step_id)
        return step_refs
    
    def analyze_pipeline_content(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze a single pipeline for template dependency issues."""
        try:
            pipeline_data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            return {
                'file': file_path,
                'error': f'YAML parsing error: {e}',
                'issues': []
            }
        
        if not isinstance(pipeline_data, dict) or 'steps' not in pipeline_data:
            return {
                'file': file_path,
                'error': 'No steps found in pipeline',
                'issues': []
            }
        
        steps = pipeline_data['steps']
        if not isinstance(steps, list):
            return {
                'file': file_path,
                'error': 'Steps is not a list',
                'issues': []
            }
        
        # Build step dependency map
        step_ids = set()
        step_dependencies = {}
        step_templates = {}
        
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
                
            step_id = step.get('id', f'step_{i}')
            step_ids.add(step_id)
            
            # Get declared dependencies
            dependencies = step.get('dependencies', [])
            if isinstance(dependencies, str):
                dependencies = [dependencies]
            step_dependencies[step_id] = set(dependencies) if dependencies else set()
            
            # Find all template variables in step
            step_content = yaml.dump(step, default_flow_style=False)
            template_vars = self.find_template_variables(step_content)
            step_refs = self.extract_step_references(template_vars)
            step_templates[step_id] = {
                'template_vars': template_vars,
                'step_refs': step_refs
            }
        
        # Find dependency issues
        issues = []
        
        for step_id, template_info in step_templates.items():
            step_refs = template_info['step_refs']
            declared_deps = step_dependencies[step_id]
            
            # Check for missing dependencies
            missing_deps = step_refs - declared_deps
            
            if missing_deps:
                # Filter out references that aren't actually step IDs
                real_missing_deps = missing_deps.intersection(step_ids)
                
                if real_missing_deps:
                    issues.append({
                        'type': 'missing_dependency',
                        'step_id': step_id,
                        'missing_dependencies': list(real_missing_deps),
                        'template_vars': list(template_info['template_vars']),
                        'declared_deps': list(declared_deps)
                    })
        
        return {
            'file': file_path,
            'step_count': len(steps),
            'step_ids': list(step_ids),
            'issues': issues
        }
    
    def audit_pipeline_file(self, file_path: Path) -> Dict[str, Any]:
        """Audit a single pipeline file."""
        self.pipelines_checked += 1
        
        try:
            content = file_path.read_text()
        except Exception as e:
            return {
                'file': str(file_path),
                'error': f'Failed to read file: {e}',
                'issues': []
            }
        
        result = self.analyze_pipeline_content(content, str(file_path))
        
        if result['issues']:
            self.issues_found.extend(result['issues'])
        
        return result
    
    def audit_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Audit all YAML files in a directory."""
        results = []
        
        # Find all YAML pipeline files
        yaml_files = []
        for pattern in ['*.yaml', '*.yml']:
            yaml_files.extend(directory.glob(pattern))
        
        yaml_files.sort()
        
        for yaml_file in yaml_files:
            if self.verbose:
                print(f"Auditing: {yaml_file}")
            
            result = self.audit_pipeline_file(yaml_file)
            results.append(result)
        
        return results
    
    def generate_fix_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Generate fix suggestions for a pipeline."""
        suggestions = []
        
        for issue in result['issues']:
            if issue['type'] == 'missing_dependency':
                step_id = issue['step_id']
                missing_deps = issue['missing_dependencies']
                
                suggestion = f"""
Step '{step_id}' uses template variables that reference other steps but doesn't declare dependencies.

Missing dependencies: {missing_deps}

Fix by adding dependencies to the step:
```yaml
- id: {step_id}
  # ... existing configuration ...
  dependencies:
{chr(10).join(f'    - {dep}' for dep in missing_deps)}
```

Template variables found: {issue['template_vars']}
"""
                suggestions.append(suggestion.strip())
        
        return suggestions
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print audit summary."""
        print("\n" + "="*60)
        print("PIPELINE TEMPLATE DEPENDENCY AUDIT SUMMARY")
        print("="*60)
        
        total_issues = sum(len(r['issues']) for r in results)
        pipelines_with_issues = sum(1 for r in results if r['issues'])
        
        print(f"Pipelines checked: {self.pipelines_checked}")
        print(f"Pipelines with issues: {pipelines_with_issues}")
        print(f"Total dependency issues found: {total_issues}")
        
        if total_issues == 0:
            print("\n✅ No template dependency issues found!")
            return
        
        print(f"\n📋 ISSUES BY PIPELINE:")
        print("-" * 40)
        
        for result in results:
            if not result['issues']:
                continue
                
            print(f"\n📁 {Path(result['file']).name}")
            print(f"   Steps: {result.get('step_count', 'unknown')}")
            print(f"   Issues: {len(result['issues'])}")
            
            for issue in result['issues']:
                if issue['type'] == 'missing_dependency':
                    print(f"   ❌ Step '{issue['step_id']}' missing deps: {issue['missing_dependencies']}")
        
        print(f"\n📝 DETAILED ANALYSIS:")
        print("-" * 40)
        
        for result in results:
            if not result['issues']:
                continue
                
            print(f"\n🔍 {result['file']}")
            
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
                continue
            
            suggestions = self.generate_fix_suggestions(result)
            for suggestion in suggestions:
                print(f"   {suggestion}")


def main():
    parser = argparse.ArgumentParser(description='Audit pipelines for template dependency issues')
    parser.add_argument('--directory', '-d', default='examples', help='Directory to audit (default: examples)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fix', action='store_true', help='Generate fix suggestions')
    
    args = parser.parse_args()
    
    # Check if directory exists
    audit_dir = Path(args.directory)
    if not audit_dir.exists():
        print(f"Error: Directory {audit_dir} not found")
        return 1
    
    # Create auditor and run audit
    auditor = PipelineAuditor(verbose=args.verbose)
    results = auditor.audit_directory(audit_dir)
    
    # Print results
    auditor.print_summary(results)
    
    # Return error code if issues found
    total_issues = sum(len(r['issues']) for r in results)
    return 1 if total_issues > 0 else 0


if __name__ == '__main__':
    import sys
    sys.exit(main())