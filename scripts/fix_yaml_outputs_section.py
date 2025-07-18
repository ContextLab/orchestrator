#!/usr/bin/env python3
"""Fix missing outputs: section in YAML files."""

from pathlib import Path

def fix_creative_writing_yaml():
    """Fix creative_writing_assistant.yaml outputs section."""
    filepath = Path("examples/creative_writing_assistant.yaml")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        # Look for the line right before the outputs
        if 'depends_on: [complete_story]' in line:
            # Add outputs: section
            fixed_lines.append('\n')
            fixed_lines.append('outputs:\n')
    
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {filepath.name}")

def fix_financial_analysis_yaml():
    """Fix financial_analysis_bot.yaml outputs section."""
    filepath = Path("examples/financial_analysis_bot.yaml")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        # Look for the line right before the outputs
        if 'depends_on: [generate_report]' in line and i < len(lines) - 1 and 'symbols_analyzed:' in lines[i + 1]:
            # Add outputs: section
            fixed_lines.append('\n')
            fixed_lines.append('outputs:\n')
    
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {filepath.name}")

def main():
    """Fix outputs section in problematic YAML files."""
    fix_creative_writing_yaml()
    fix_financial_analysis_yaml()

if __name__ == "__main__":
    main()