#!/usr/bin/env python3
"""
Fix all YAML pipeline syntax errors and ensure consistent high-quality models.
"""

import re
from pathlib import Path
import yaml

# Best models for consistent high quality
BEST_MODELS = {
    "default": "anthropic/claude-sonnet-4-20250514",  # Best overall
    "analysis": "openai/gpt-4.1",  # Best for analysis
    "creative": "anthropic/claude-sonnet-4-20250514",  # Best for creative
    "coding": "openai/gpt-4.1",  # Best for coding
    "research": "google/gemini-2.5-flash",  # Good for research
    "data": "openai/gpt-4.1",  # Best for data processing
}

def fix_yaml_indentation(content: str) -> str:
    """Fix common YAML indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    in_action_block = False
    action_indent = 0
    
    for i, line in enumerate(lines):
        # Detect action blocks
        if re.match(r'^\s*action:\s*\|', line):
            in_action_block = True
            action_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
            continue
        
        # Handle end of action block
        if in_action_block and line.strip() and not line.startswith(' ' * (action_indent + 2)):
            in_action_block = False
        
        # Fix missing spaces after colons in flow mappings
        if not in_action_block:
            line = re.sub(r'(\w+):(\w+)', r'\1: \2', line)
        
        # Fix depends_on with unquoted template variables
        if 'depends_on:' in line and '{{' in line:
            # Extract the depends_on part
            match = re.search(r'depends_on:\s*\[(.*?)\]', line)
            if match:
                deps = match.group(1)
                # Quote any template variables
                deps = re.sub(r'(\{\{[^}]+\}\})', r'"\1"', deps)
                line = line[:match.start()] + f'depends_on: [{deps}]' + line[match.end():]
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def fix_action_blocks(content: str) -> str:
    """Fix malformed action blocks."""
    # Fix action blocks that have text on the same line as the pipe
    content = re.sub(
        r'action:\s*\|\s*([^\n]+)',
        r'action: |\n      \1',
        content
    )
    
    # Fix action blocks with improper indentation
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for action: | pattern
        if re.match(r'^\s*action:\s*\|', line):
            fixed_lines.append(line)
            base_indent = len(line) - len(line.lstrip())
            i += 1
            
            # Process action block content
            while i < len(lines):
                next_line = lines[i]
                
                # Check if we've exited the action block
                if next_line.strip() and not next_line.startswith(' ' * (base_indent + 2)):
                    # Check if this is a field that should be outside the action
                    if re.match(r'^\s*(depends_on|condition|timeout|cache_results|on_error|tags):', next_line):
                        break
                    # Otherwise, this line should be indented
                    fixed_lines.append(' ' * (base_indent + 6) + next_line.strip())
                else:
                    fixed_lines.append(next_line)
                i += 1
            i -= 1  # Back up one since we'll increment at the end
        else:
            fixed_lines.append(line)
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_model_references(content: str, pipeline_type: str = "default") -> str:
    """Update model references to best models."""
    # Determine best model based on pipeline type
    if "analysis" in pipeline_type or "code" in pipeline_type:
        best_model = BEST_MODELS["analysis"]
    elif "creative" in pipeline_type or "writing" in pipeline_type:
        best_model = BEST_MODELS["creative"]
    elif "research" in pipeline_type:
        best_model = BEST_MODELS["research"]
    elif "data" in pipeline_type:
        best_model = BEST_MODELS["data"]
    else:
        best_model = BEST_MODELS["default"]
    
    # Update main model reference
    content = re.sub(
        r'model:\s*"[^"]*"',
        f'model: "{best_model}"',
        content,
        count=1
    )
    
    # Update step-specific models to complementary models
    # Use different models for variety but maintain quality
    secondary_models = {
        "openai/gpt-4.1": "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-sonnet-4-20250514": "openai/gpt-4.1",
        "google/gemini-2.5-flash": "openai/gpt-4.1",
    }
    
    # Update model references in steps
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'model:' in line and 'openai' in line.lower():
            if best_model.startswith("openai"):
                lines[i] = re.sub(r'model:\s*"[^"]*"', f'model: "{secondary_models.get(best_model, best_model)}"', line)
    
    return '\n'.join(lines)

def add_proper_save_step(content: str, filename: str) -> str:
    """Add a proper save step that extracts content correctly."""
    pipeline_name = Path(filename).stem
    
    # Check if save step exists
    if 'save_output' in content or 'save_to_file' in content:
        return content
    
    # Find the main output step
    main_outputs = {
        "research": "generate_report",
        "content": "final_output",
        "creative": "complete_story",
        "analysis": "generate_report",
        "data": "generate_insights",
        "chat": "format_conversation",
        "financial": "generate_report",
        "customer": "final_response",
        "document": "final_analysis",
        "multi_agent": "final_consensus",
        "test": "test_report",
    }
    
    main_output_step = None
    for key, step in main_outputs.items():
        if key in pipeline_name.lower():
            main_output_step = step
            break
    
    if not main_output_step:
        # Try to find the last substantial step
        steps = re.findall(r'- id: (\w+)', content)
        if steps:
            main_output_step = steps[-1]
    
    # Create save step
    save_step = f'''
  - id: save_output
    action: |
      Save the pipeline output to markdown file.
      
      Create a file at: examples/output/{pipeline_name}.md
      
      Content to save:
      # {pipeline_name.replace('_', ' ').title()}
      
      Generated on: {{{{execution.timestamp}}}}
      
      ## Pipeline Output
      
      {{{{{main_output_step}.result}}}}
      
      ---
      Generated by Orchestrator Pipeline'''
    
    if main_output_step:
        save_step += f"\n    depends_on: [{main_output_step}]"
    
    # Add before outputs section
    if 'outputs:' in content:
        content = content.replace('outputs:', save_step + '\n\noutputs:')
    else:
        content += '\n' + save_step + '\n'
    
    return content

def fix_template_variables(content: str) -> str:
    """Ensure template variables are properly handled."""
    # Fix unquoted template variables in lists
    content = re.sub(
        r'\[(.*?{{[^}]+}}.*?)\]',
        lambda m: '[' + ', '.join(f'"{item.strip()}"' if '{{' in item else item.strip() 
                                  for item in m.group(1).split(',')) + ']',
        content
    )
    
    return content

def fix_yaml_file(filepath: Path) -> None:
    """Fix all issues in a YAML file."""
    print(f"Fixing {filepath.name}...")
    
    try:
        content = filepath.read_text()
        original_content = content
        
        # Determine pipeline type from filename
        pipeline_type = filepath.stem.lower()
        
        # Apply fixes in order
        content = fix_yaml_indentation(content)
        content = fix_action_blocks(content)
        content = fix_model_references(content, pipeline_type)
        content = fix_template_variables(content)
        content = add_proper_save_step(content, filepath.name)
        
        # Validate YAML
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            print(f"  ⚠️ YAML validation failed after fixes: {e}")
            # Try to fix the specific error
            if "found undefined alias" in str(e):
                # Remove asterisks that might be interpreted as aliases
                content = re.sub(r'^\*', '#', content, flags=re.MULTILINE)
        
        # Write back only if changed
        if content != original_content:
            filepath.write_text(content)
            print(f"  ✅ Fixed {filepath.name}")
        else:
            print(f"  ⏭️ No changes needed for {filepath.name}")
            
    except Exception as e:
        print(f"  ❌ Error fixing {filepath.name}: {e}")

def main():
    """Fix all YAML pipeline files."""
    print("🔧 Fixing All YAML Pipeline Files")
    print("="*60)
    
    # Get all YAML files
    yaml_files = list(Path("examples").glob("*.yaml"))
    
    print(f"\nFound {len(yaml_files)} YAML files to process\n")
    
    # Fix each file
    for yaml_file in sorted(yaml_files):
        fix_yaml_file(yaml_file)
    
    print("\n✨ All YAML files processed!")

if __name__ == "__main__":
    main()