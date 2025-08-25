#!/usr/bin/env python3
"""
Test script to explore POML SDK capabilities and understand the integration points.
"""

import poml
from poml import Prompt
import json

def test_basic_poml_template():
    """Test basic POML template creation and rendering."""
    print("=== Basic POML Template Test ===")
    
    # Create a basic POML prompt
    with Prompt() as p:
        with p.role():
            p.text("You are a helpful data analyst")
        
        with p.task():
            p.text("Analyze the following data and provide insights")
            
        with p.example():
            with p.example_input():
                p.text("Dataset: sales_data.csv with 1000 records")
            with p.example_output():
                p.text("Key insights: Revenue increased 25% year-over-year")
    
    # Render the template
    result = p.render()
    print("Template type:", type(result))
    print("Rendered content:")
    print(result)
    print()

def test_poml_with_data():
    """Test POML with data integration components."""
    print("=== POML Data Integration Test ===")
    
    with Prompt() as p:
        with p.role():
            p.text("You are a data processing assistant")
            
        with p.task():
            p.text("Process the provided document and extract key information")
            
        # Test document component (will fail gracefully if file doesn't exist)
        try:
            with p.document(src="test_data.txt"):
                p.text("Sample document content")
        except Exception as e:
            print(f"Document component failed as expected: {e}")
            # Fallback to regular content
            with p.paragraph():
                p.text("Sample data content for processing")
    
    try:
        result = p.render()
        print("Data integration template:")
        print(result)
    except Exception as e:
        print(f"Template rendering failed: {e}")
    print()

def test_poml_with_variables():
    """Test POML with variable substitution."""
    print("=== POML Variable Test ===")
    
    context_data = {
        "task_name": "Data Analysis",
        "dataset": "customer_data.csv",
        "user_name": "Alice"
    }
    
    with Prompt() as p:
        with p.role():
            p.text("You are an expert {{role_type}} assistant")
            
        with p.task():
            p.text("Help {{user_name}} with {{task_name}} using {{dataset}}")
    
    try:
        # Try rendering with context
        result = p.render(context=context_data)
        print("Template with variables:")
        print(result)
    except Exception as e:
        print(f"Variable rendering failed: {e}")
        # Try basic rendering
        try:
            result = p.render()
            print("Basic template:")
            print(result)
        except Exception as e2:
            print(f"Basic rendering also failed: {e2}")
    print()

def test_poml_xml_output():
    """Test POML XML dump for understanding structure."""
    print("=== POML XML Structure Test ===")
    
    with Prompt() as p:
        with p.role():
            p.text("Data Analyst")
            
        with p.task():
            p.text("Analyze data patterns")
            
        with p.hint():
            p.text("Focus on trends and anomalies")
    
    # Get XML structure
    xml_output = p.dump_xml()
    print("XML structure:")
    print(xml_output)
    print()

def explore_poml_api():
    """Explore available POML API functions."""
    print("=== POML API Exploration ===")
    
    print("POML module attributes:", [attr for attr in dir(poml) if not attr.startswith('_')])
    print()
    
    # Check if there are utility functions
    if hasattr(poml, 'api'):
        print("POML API attributes:", [attr for attr in dir(poml.api) if not attr.startswith('_')])
        print()

if __name__ == "__main__":
    explore_poml_api()
    test_basic_poml_template()
    test_poml_with_data()
    test_poml_with_variables()
    test_poml_xml_output()