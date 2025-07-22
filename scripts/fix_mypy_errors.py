#!/usr/bin/env python3
"""Fix common mypy errors in the codebase."""

import re
from pathlib import Path


def fix_missing_return_type(content: str) -> str:
    """Add -> None to functions missing return type annotations."""
    # Pattern to find function definitions without return type
    pattern = r'(\s*def\s+\w+\s*\([^)]*\)\s*)(?!->)(\s*:)'
    
    def check_and_fix(match):
        func_def = match.group(0)
        indent = match.group(1)
        
        # Check if function has a return statement
        func_name_match = re.search(r'def\s+(\w+)', func_def)
        if func_name_match:
            func_name = func_name_match.group(1)
            # Skip __init__ methods
            if func_name == '__init__':
                return indent + ' -> None' + match.group(2)
        
        return match.group(0)
    
    # First pass: Add -> None to functions that definitely don't return
    content = re.sub(pattern, lambda m: m.group(1) + ' -> None' + m.group(2), content)
    
    return content


def fix_dict_annotations(content: str) -> str:
    """Fix dict type annotations."""
    # Fix "Need type annotation" for dicts
    content = re.sub(
        r'(\s+)(\w+)\s*=\s*\{\}',
        r'\1\2: Dict[str, Any] = {}',
        content
    )
    
    # Fix specific cases
    content = re.sub(
        r'graph = nx\.DiGraph\(\)',
        'graph: nx.DiGraph = nx.DiGraph()',
        content
    )
    
    content = re.sub(
        r'_responses = \{\}',
        '_responses: Dict[str, Any] = {}',
        content
    )
    
    content = re.sub(
        r'status_counts = \{\}',
        'status_counts: Dict[str, int] = {}',
        content
    )
    
    content = re.sub(
        r'memo = \{\}',
        'memo: Dict[str, Tuple[int, List[str]]] = {}',
        content
    )
    
    return content


def fix_optional_parameters(content: str) -> str:
    """Fix implicit Optional parameters."""
    # Pattern to find parameters with None default
    pattern = r'(\w+):\s*([^=,\)]+)\s*=\s*None'
    
    def fix_optional(match):
        param_name = match.group(1)
        type_annotation = match.group(2).strip()
        
        # Skip if already Optional or Union
        if 'Optional' in type_annotation or 'Union' in type_annotation or '|' in type_annotation:
            return match.group(0)
        
        return f'{param_name}: Optional[{type_annotation}] = None'
    
    content = re.sub(pattern, fix_optional, content)
    
    return content


def fix_imports(content: str) -> str:
    """Ensure necessary imports are present."""
    # Check if Optional is needed but not imported
    if 'Optional[' in content and 'from typing import' in content:
        # Find the typing import line
        import_match = re.search(r'from typing import (.+)', content)
        if import_match:
            imports = import_match.group(1)
            if 'Optional' not in imports:
                # Add Optional to imports
                new_imports = imports.rstrip() + ', Optional'
                content = content.replace(
                    f'from typing import {imports}',
                    f'from typing import {new_imports}'
                )
    
    # Ensure Dict, List, Tuple are imported if used
    needed_imports = []
    if 'Dict[' in content and 'Dict' not in content:
        needed_imports.append('Dict')
    if 'List[' in content and 'List' not in content:
        needed_imports.append('List')
    if 'Tuple[' in content and 'Tuple' not in content:
        needed_imports.append('Tuple')
    
    if needed_imports and 'from typing import' in content:
        import_match = re.search(r'from typing import (.+)', content)
        if import_match:
            imports = import_match.group(1)
            for imp in needed_imports:
                if imp not in imports:
                    imports += f', {imp}'
            content = content.replace(
                f'from typing import {import_match.group(1)}',
                f'from typing import {imports}'
            )
    
    return content


def fix_file(filepath: Path) -> bool:
    """Fix mypy errors in a single file."""
    try:
        content = filepath.read_text()
        original_content = content
        
        # Apply fixes
        content = fix_imports(content)
        content = fix_missing_return_type(content)
        content = fix_dict_annotations(content)
        content = fix_optional_parameters(content)
        
        # Write back if changed
        if content != original_content:
            filepath.write_text(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    """Fix mypy errors in specific files."""
    files_to_fix = [
        "src/orchestrator/engine/pipeline_spec.py",
        "src/orchestrator/core/model.py",
        "src/orchestrator/state/simple_state_manager.py",
        "src/orchestrator/core/pipeline.py",
        "src/orchestrator/compiler/template_renderer.py",
        "src/orchestrator/engine/auto_resolver.py",
        "src/orchestrator/core/control_system.py",
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        path = Path(file_path)
        if path.exists():
            if fix_file(path):
                print(f"✅ Fixed {file_path}")
                fixed_count += 1
            else:
                print(f"⏭️  No changes needed for {file_path}")
        else:
            print(f"❌ File not found: {file_path}")
    
    print(f"\n✨ Fixed {fixed_count} files")


if __name__ == "__main__":
    main()