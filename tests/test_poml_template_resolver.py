#!/usr/bin/env python3
"""
Test the POML integration in the enhanced TemplateResolver.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator.core.template_resolver import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    TemplateResolver, 
    TemplateFormat, 
    TemplateFormatDetector, 
    POMLTemplateProcessor,
    POMLIntegrationError,
    POML_AVAILABLE
)
from src.orchestrator.core.output_tracker import OutputTracker
from src.orchestrator.core.output_metadata import OutputReference

def test_format_detection():
    """Test automatic template format detection."""
    print("=== Template Format Detection Test ===")
    
    detector = TemplateFormatDetector()
    
    test_cases = [
        # Jinja2 templates
        ("Hello {{ user_name }}", TemplateFormat.JINJA2),
        ("{% for item in items %}{{ item }}{% endfor %}", TemplateFormat.JINJA2),
        
        # POML templates
        ("<role>Assistant</role><task>Help user</task>", TemplateFormat.POML),
        ("<poml><role>Data Analyst</role></poml>", TemplateFormat.POML),
        
        # Hybrid templates
        ("<role>{{ role_type }}</role><task>Process {{ data }}</task>", TemplateFormat.HYBRID),
        ("Regular {{ var }} text <role>Assistant</role>", TemplateFormat.HYBRID),
        
        # Plain text
        ("Just plain text", TemplateFormat.PLAIN),
        ("", TemplateFormat.PLAIN),
        ("HTML-like <div>content</div>", TemplateFormat.PLAIN),
    ]
    
    for template, expected_format in test_cases:
        detected_format = detector.detect_format(template)
        status = "✓" if detected_format == expected_format else "✗"
        print(f"{status} '{template[:50]}...' -> {detected_format.value} (expected: {expected_format.value})")
    
    print()

def test_backward_compatibility():
    """Test that existing Jinja2 templates still work unchanged."""
    print("=== Backward Compatibility Test ===")
    
    # Create mock output tracker with test data
    output_tracker = OutputTracker()
    
    # Add some test outputs using register_output
    output_tracker.register_output("task1", "Analysis complete")
    output_tracker.register_output("task2", {"count": 100, "status": "success"})
    
    # Create enhanced template resolver
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    # Test existing Jinja2 templates
    test_templates = [
        "The result is: {{ task1 }}",
        "Status: {{ task2.status }}, Count: {{ task2.count }}",
        "Task {{ task1 }} completed with {{ task2.count }} items",
    ]
    
    for template in test_templates:
        resolved = resolver.resolve_template(template)
        print(f"Template: {template}")
        print(f"Resolved: {resolved}")
        print()

def test_poml_integration():
    """Test POML template processing."""
    print("=== POML Integration Test ===")
    
    if not POML_AVAILABLE:
        print("POML SDK not available, skipping integration tests")
        return
    
    # Create test data
    output_tracker = OutputTracker()
    output_tracker.register_output("data_analysis", "Customer satisfaction increased by 15%")
    output_tracker.register_output("file_processor", "/data/processed_results.csv")
    
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    if not resolver.is_poml_available():
        print("POML processor not available")
        return
    
    # Test basic POML template
    poml_template = """
    <role>Data Analysis Expert</role>
    <task>Analyze the processed data and provide insights about customer satisfaction trends</task>
    <hint>Focus on quantitative metrics and actionable recommendations</hint>
    <output-format>Provide insights in a structured format with key metrics and recommendations</output-format>
    """
    
    resolved = resolver.resolve_template(poml_template)
    print("POML Template Resolved:")
    print(resolved)
    print()

def test_hybrid_template():
    """Test hybrid templates with both Jinja2 and POML syntax."""
    print("=== Hybrid Template Test ===")
    
    if not POML_AVAILABLE:
        print("POML SDK not available, skipping hybrid tests")
        return
    
    output_tracker = OutputTracker()
    output_tracker.register_output("analysis_task", "Sales data processed")
    output_tracker.register_output("user_context", {"name": "Alice"})
    
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    if not resolver.is_poml_available():
        print("POML processor not available")
        return
    
    # Hybrid template with both syntaxes
    hybrid_template = """
    <role>{{ user_context.name }}'s Personal Assistant</role>
    <task>Help analyze the results from {{ analysis_task }}</task>
    <example>
        <input>What are the key findings from the analysis?</input>
        <output>The analysis shows {{ analysis_task }} with positive trends</output>
    </example>
    """
    
    try:
        resolved = resolver.resolve_template(hybrid_template)
        print("Hybrid Template Resolved:")
        print(resolved)
    except Exception as e:
        print(f"Hybrid template failed: {e}")
    print()

def test_template_validation():
    """Test POML template validation features."""
    print("=== Template Validation Test ===")
    
    output_tracker = OutputTracker()
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    # Test various templates for validation
    test_templates = [
        # Valid POML
        "<role>Assistant</role><task>Help user</task>",
        
        # Invalid POML (missing required elements)
        "<hint>Just a hint without role or task</hint>",
        
        # Valid Jinja2
        "Hello {{ user_name }}",
        
        # Mixed formats
        "<role>{{ role_type }}</role><task>{{ task_desc }}</task>",
    ]
    
    for template in test_templates:
        format_detected = resolver.get_template_format(template)
        validation_issues = resolver.validate_poml_template(template)
        
        print(f"Template: '{template}'")
        print(f"Format: {format_detected.value}")
        print(f"Issues: {validation_issues if validation_issues else 'None'}")
        print()

def test_programmatic_creation():
    """Test programmatic POML template creation."""
    print("=== Programmatic Template Creation Test ===")
    
    output_tracker = OutputTracker()
    resolver = TemplateResolver(output_tracker, enable_poml=True)
    
    # Create POML template programmatically
    poml_template = resolver.create_poml_template_from_components(
        role="Data Processing Assistant",
        task="Process the uploaded CSV file and extract key metrics",
        examples=[
            {
                "input": "Please analyze sales_data.csv",
                "output": "Found 1,250 records with total revenue of $45,000"
            }
        ],
        hints=["Focus on numerical summaries", "Check for data quality issues"],
        output_format="Provide results in JSON format with metrics and insights"
    )
    
    print("Programmatically created POML template:")
    print(poml_template)
    print()
    
    # Test that it's recognized as POML
    format_detected = resolver.get_template_format(poml_template)
    print(f"Detected format: {format_detected.value}")
    
    if resolver.is_poml_available():
        try:
            resolved = resolver.resolve_template(poml_template)
            print("Resolved template:")
            print(resolved)
        except Exception as e:
            print(f"Template resolution failed: {e}")
    print()

def run_all_tests():
    """Run all POML integration tests."""
    print("Testing POML Integration in Template Resolver")
    print("=" * 60)
    print()
    
    test_format_detection()
    test_backward_compatibility()
    test_poml_integration()
    test_hybrid_template()
    test_template_validation()
    test_programmatic_creation()
    
    print("=" * 60)
    print("POML Integration Tests Complete")

if __name__ == "__main__":
    run_all_tests()