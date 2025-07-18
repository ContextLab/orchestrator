#!/usr/bin/env python3
"""Organize examples to match documentation."""

from pathlib import Path
import shutil

# Documented examples from docs/tutorials/examples.rst
DOCUMENTED_EXAMPLES = {
    "research_assistant.yaml",
    "code_analysis_suite.yaml", 
    "content_creation_pipeline.yaml",
    "data_processing_workflow.yaml",
    "multi_agent_collaboration.yaml",
    "automated_testing_system.yaml",
    "document_intelligence.yaml",
    "creative_writing_assistant.yaml",
    "financial_analysis_bot.yaml",
    "interactive_chat_bot.yaml",
    "scalable_customer_service_agent.yaml",
    "customer_support_automation.yaml",
}

def main():
    """Organize examples."""
    examples_dir = Path("examples")
    
    # Get all YAML files
    all_yamls = list(examples_dir.glob("*.yaml"))
    
    print("📋 Analyzing examples...")
    print(f"Found {len(all_yamls)} YAML files")
    print(f"Documented: {len(DOCUMENTED_EXAMPLES)} examples")
    
    # Categorize files
    documented = []
    undocumented = []
    
    for yaml_file in all_yamls:
        if yaml_file.name in DOCUMENTED_EXAMPLES:
            documented.append(yaml_file)
        else:
            undocumented.append(yaml_file)
    
    print(f"\n✅ Documented examples ({len(documented)}):")
    for f in sorted(documented):
        print(f"  - {f.name}")
    
    print(f"\n❌ Undocumented examples to remove ({len(undocumented)}):")
    for f in sorted(undocumented):
        print(f"  - {f.name}")
    
    # Create backup directory
    backup_dir = Path("scripts/undocumented_examples_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Move undocumented files
    print(f"\n🗂️  Moving undocumented examples to {backup_dir}")
    for f in undocumented:
        dest = backup_dir / f.name
        shutil.move(str(f), str(dest))
        print(f"  Moved: {f.name}")
    
    print("\n✨ Examples organized!")


if __name__ == "__main__":
    main()