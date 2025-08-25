#!/usr/bin/env python3
"""
Better test script to understand POML SDK for template system integration.
"""

import poml
from poml import Prompt
import json
import tempfile
import os

def test_simple_poml_rendering():
    """Test simple POML rendering without variables."""
    print("=== Simple POML Rendering ===")
    
    with Prompt() as p:
        with p.role():
            p.text("You are a helpful data analyst")
        
        with p.task():
            p.text("Analyze the provided data and generate insights")
    
    # Test different render modes
    result_chat = p.render(chat=True)
    print("Chat mode result:", type(result_chat))
    print(json.dumps(result_chat, indent=2))
    
    result_plain = p.render(chat=False)
    print("\nPlain mode result:", type(result_plain))
    print(result_plain)
    print()

def test_poml_xml_to_text():
    """Test POML XML structure and conversion to text."""
    print("=== POML XML to Text Conversion ===")
    
    with Prompt() as p:
        with p.role():
            p.text("Data Processing Assistant")
            
        with p.task():
            p.text("Process the input data file and extract key metrics")
            
        with p.output_format():
            p.text("Provide results in JSON format with metrics and insights")
    
    # Get XML structure
    xml_structure = p.dump_xml()
    print("XML structure:")
    print(xml_structure)
    
    # Get rendered output
    result = p.render(chat=False)
    print("\nRendered text:")
    print(result)
    print()

def test_poml_with_simple_data():
    """Test POML with data content (without external files)."""
    print("=== POML with Simple Data ===")
    
    with Prompt() as p:
        with p.role():
            p.text("Data Analysis Expert")
            
        with p.task():
            p.text("Analyze the following dataset")
            
        # Use table or object components with inline data
        with p.paragraph():
            p.text("Dataset: sales_q1_2024.csv")
            
        with p.object(syntax="json", data={"records": 1000, "revenue": 50000, "region": "north"}):
            p.text("Sample data structure")
    
    try:
        result = p.render(chat=False)
        print("Data integration result:")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        # Try XML dump to see structure
        xml = p.dump_xml()
        print("XML structure:")
        print(xml)
    print()

def test_format_detection_patterns():
    """Test patterns for detecting POML vs Jinja2 templates."""
    print("=== Template Format Detection Patterns ===")
    
    # Test various template strings
    test_templates = [
        # Pure Jinja2
        "Hello {{ user_name }}, your task is {{ task }}",
        "{% for item in items %}{{ item }}{% endfor %}",
        
        # Pure POML (as strings)
        "<role>Data Analyst</role><task>Process {{ data_file }}</task>",
        "<poml><role>Assistant</role><task>Help with analysis</task></poml>",
        
        # Mixed/Hybrid
        "<role>{{ role_type }}</role><task>{{ task_description }}</task>",
        "Regular text with {{ variable }} and <role>Assistant</role>",
        
        # Edge cases
        "",
        "Just plain text",
        "HTML-like <div>content</div> but not POML",
        "{{ var }} <span>not poml</span>",
    ]
    
    for template in test_templates:
        has_jinja = bool('{{' in template and '}}' in template) or bool('{%' in template and '%}' in template)
        has_poml = bool('<role>' in template or '<task>' in template or '<poml>' in template)
        
        format_type = "unknown"
        if has_poml and has_jinja:
            format_type = "hybrid"
        elif has_poml:
            format_type = "poml"
        elif has_jinja:
            format_type = "jinja2"
        else:
            format_type = "plain"
            
        print(f"Template: '{template[:50]}{'...' if len(template) > 50 else ''}'")
        print(f"  -> Format: {format_type}")
        print()

def create_sample_data_file():
    """Create a temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Sample Data Content\n")
        f.write("Record 1: Value A\n")
        f.write("Record 2: Value B\n")
        f.write("Record 3: Value C\n")
        return f.name

def test_poml_file_integration():
    """Test POML with actual file integration."""
    print("=== POML File Integration Test ===")
    
    # Create a temporary file
    test_file = create_sample_data_file()
    
    try:
        with Prompt() as p:
            with p.role():
                p.text("Document Processor")
                
            with p.task():
                p.text("Process the document and summarize its contents")
            
            # Use document component with actual file
            with p.document(src=test_file, parser="txt"):
                pass  # File content will be automatically included
        
        result = p.render(chat=False)
        print("File integration result:")
        print(result)
        
    except Exception as e:
        print(f"File integration failed: {e}")
        # Show XML structure instead
        xml = p.dump_xml()
        print("XML structure:")
        print(xml)
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
    print()

def understand_context_integration():
    """Understand how POML context works for integration with our system."""
    print("=== Understanding POML Context Integration ===")
    
    # Test what context structure POML expects
    test_context = {
        "user_name": "Alice",
        "data_file": "analysis_data.csv",
        "task_type": "data_analysis"
    }
    
    # Create POML template programmatically (not using context in template content)
    with Prompt() as p:
        with p.role():
            p.text(f"You are a {test_context['task_type']} assistant")
            
        with p.task():
            p.text(f"Help {test_context['user_name']} process {test_context['data_file']}")
    
    result = p.render(chat=False)
    print("Programmatic context integration:")
    print(result)
    print()
    
    # Now test how we could integrate with our existing context system
    print("Integration approach for existing template system:")
    print("1. Our system populates context dict")
    print("2. We programmatically build POML structure using context values")
    print("3. We render POML to get final template")
    print("4. We can then pass that to LLM or further process")

if __name__ == "__main__":
    test_simple_poml_rendering()
    test_poml_xml_to_text()
    test_poml_with_simple_data()
    test_format_detection_patterns()
    test_poml_file_integration()
    understand_context_integration()