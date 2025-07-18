#!/usr/bin/env python3
"""Fix the content_creation_pipeline.yaml file specifically."""

from pathlib import Path

def fix_content_creation_yaml():
    """Fix the broken content creation YAML file."""
    filepath = Path("examples/content_creation_pipeline.yaml")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for the broken section around line 121-127
        if i == 121 and "1. Hero image for blog post" in line:
            # This content belongs to a new step - generate_visuals
            # First close the previous step properly
            fixed_lines.append("    tags: [\"social\", \"content\"]\n")
            fixed_lines.append("\n")
            fixed_lines.append("  # Step 6: Generate visual content\n")
            fixed_lines.append("  - id: generate_visuals\n") 
            fixed_lines.append("    action: |\n")
            fixed_lines.append("      <AUTO>create visual assets for the content:\n")
            fixed_lines.append("      1. Hero image for blog post (1200x628px)\n")
            fixed_lines.append("      2. Social media cards for each platform\n")
            fixed_lines.append("      3. Infographic summarizing key points\n")
            fixed_lines.append("      4. Quote cards with key insights\n")
            fixed_lines.append("      \n")
            fixed_lines.append("      Include alt text and captions for accessibility</AUTO>\n")
            # Skip the lines with this content
            while i < len(lines) and "Include alt text and captions for accessibility</AUTO>" not in lines[i]:
                i += 1
            i += 1  # Skip the line with </AUTO>
            continue
        
        # Look for references to final_output that doesn't exist
        if "final_output" in line:
            # Skip the save_output step that references non-existent final_output
            if "- id: save_output" in line:
                # Skip this entire step
                while i < len(lines) and not (lines[i].strip() and lines[i][0] not in ' \t'):
                    i += 1
                i -= 1  # Back up one
            else:
                # Replace final_output references with save_content_to_file
                line = line.replace("final_output", "save_content_to_file")
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed {filepath}")

if __name__ == "__main__":
    fix_content_creation_yaml()