#!/usr/bin/env python3
"""
Test POML migration tools for converting existing templates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator.core.template_migration_tools import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    TemplateMigrationAnalyzer,
    TemplateMigrationEngine,
    MigrationStrategy,
    analyze_template,
    migrate_template,
    batch_analyze_templates,
    batch_migrate_templates
)

def test_template_analysis():
    """Test template analysis functionality."""
    print("=== Template Analysis Test ===")
    
    analyzer = TemplateMigrationAnalyzer()
    
    # Test different types of templates
    test_templates = {
        "simple_jinja": "Hello {{ user_name }}, please analyze {{ data_file }}",
        
        "structured_jinja": """
        Role: You are a data analyst
        Task: Process the uploaded file {{ filename }}
        Please provide insights in JSON format.
        """,
        
        "complex_jinja": """
        {% if user_type == 'admin' %}
        You are an advanced {{ role | default('analyst') }}.
        {% else %}
        You are a basic analyst.
        {% endif %}
        
        Task: Process these files:
        {% for file in files %}
        - {{ file.name }}: {{ file.description }}
        {% endfor %}
        
        Output format: {{ output_format | upper }}
        """,
        
        "plain_text": "Please analyze the data and provide insights.",
        
        "existing_poml": "<role>Data Analyst</role><task>Process data</task>"
    }
    
    for name, template in test_templates.items():
        analysis = analyzer.analyze_template(template)
        
        print(f"\n--- {name.upper()} ---")
        print(f"Format: {analysis.original_format.value}")
        print(f"Complexity: {analysis.complexity_score:.2f}")
        print(f"Features: {', '.join(analysis.jinja2_features) if analysis.jinja2_features else 'None'}")
        print(f"Strategy: {analysis.suggested_strategy.value}")
        print(f"Effort: {analysis.migration_effort}")
        print(f"Benefits: {len(analysis.poml_benefits)} identified")
        print(f"Notes: {len(analysis.migration_notes)} migration notes")
        
        # Show first few notes and benefits
        if analysis.migration_notes:
            print(f"  Sample note: {analysis.migration_notes[0]}")
        if analysis.poml_benefits:
            print(f"  Sample benefit: {analysis.poml_benefits[0]}")
    
    print()

def test_template_migration():
    """Test actual template migration."""
    print("=== Template Migration Test ===")
    
    engine = TemplateMigrationEngine()
    
    # Test templates to migrate
    migration_tests = [
        {
            "name": "Simple Role-Task",
            "template": "You are a helpful assistant. Please analyze the data file and provide insights.",
            "strategy": MigrationStrategy.FULL_POML
        },
        {
            "name": "Structured Content", 
            "template": """
            Role: Data Analyst
            Task: Process customer survey data
            Example: Input: survey.csv -> Output: satisfaction: 85%
            Format: JSON with metrics and insights
            """,
            "strategy": MigrationStrategy.FULL_POML
        },
        {
            "name": "Jinja2 with Variables",
            "template": "You are {{ role_type }}. Analyze {{ data_source }} for {{ stakeholder }}.",
            "strategy": MigrationStrategy.HYBRID
        },
        {
            "name": "Complex Template",
            "template": """
            Role: {{ user_role | default('Analyst') }}
            {% if include_examples %}
            Example: 
            Input: {{ sample_input }}
            Output: {{ sample_output }}
            {% endif %}
            Process data from {{ data_files | join(', ') }}.
            """,
            "strategy": MigrationStrategy.ENHANCED_JINJA2
        }
    ]
    
    for test in migration_tests:
        print(f"\n--- {test['name'].upper()} ---")
        print(f"Strategy: {test['strategy'].value}")
        print("Original:")
        print(f"  {test['template'].strip()}")
        
        result = engine.migrate_template(test['template'], test['strategy'])
        
        print(f"Success: {result.success}")
        print(f"Changes: {', '.join(result.changes_made) if result.changes_made else 'None'}")
        
        if result.validation_issues:
            print(f"Issues: {', '.join(result.validation_issues)}")
        
        print("Migrated:")
        for line in result.migrated_template.split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
        print()

def test_batch_operations():
    """Test batch analysis and migration."""
    print("=== Batch Operations Test ===")
    
    # Sample template collection
    template_collection = {
        "customer_analysis": "You are a customer service analyst. Review {{ tickets }} and summarize issues.",
        "data_processor": """
        Task: Process {{ input_file }}
        Output: Generate report in {{ format }}
        """,
        "report_generator": """
        Role: Report Writer
        Task: Create executive summary
        Example: Input: metrics data -> Output: Executive dashboard
        Hint: Focus on key business metrics
        """,
        "simple_query": "What are the trends in {{ dataset }}?",
    }
    
    # Batch analysis
    print("Batch Analysis Results:")
    analyses = batch_analyze_templates(template_collection)
    
    for name, analysis in analyses.items():
        print(f"  {name}: {analysis.original_format.value} -> {analysis.suggested_strategy.value} (effort: {analysis.migration_effort})")
    
    print()
    
    # Batch migration using auto-detected strategies
    print("Batch Migration Results:")
    migrations = batch_migrate_templates(template_collection)
    
    for name, result in migrations.items():
        status = "✓" if result.success else "✗"
        print(f"  {status} {name}: {result.strategy_used.value}")
        if result.changes_made:
            print(f"    Changes: {result.changes_made[0]}")
    
    print()

def test_convenience_functions():
    """Test convenience functions."""
    print("=== Convenience Functions Test ===")
    
    # Test single template analysis
    template = "You are a {{ role }}. Process {{ data }} and provide {{ output_type }}."
    
    analysis = analyze_template(template)
    print(f"Quick Analysis: {analysis.original_format.value} -> {analysis.suggested_strategy.value}")
    
    # Test single template migration
    result = migrate_template(template)
    print(f"Quick Migration: {result.strategy_used.value} ({'success' if result.success else 'failed'})")
    
    if result.success:
        print("Result preview:")
        lines = result.migrated_template.split('\n')[:3]
        for line in lines:
            if line.strip():
                print(f"  {line.strip()}")
        if len(result.migrated_template.split('\n')) > 3:
            print("  ...")
    
    print()

def test_real_world_examples():
    """Test with realistic template examples."""
    print("=== Real-World Examples Test ===")
    
    examples = {
        "api_documentation": """
        You are a technical writer. 
        
        Task: Generate API documentation for {{ service_name }}
        
        Include:
        - Endpoint descriptions  
        - Request/response examples
        - Error codes and messages
        
        Format: Markdown with code blocks
        """,
        
        "customer_support": """
        You are a {{ support_level | default('senior') }} customer support specialist.
        
        {% if priority == 'high' %}
        This is a HIGH PRIORITY ticket. Respond immediately.
        {% endif %}
        
        Customer issue: {{ issue_description }}
        Account type: {{ account.type }}
        Previous interactions: {{ history | length }}
        
        Provide a professional response addressing the customer's concerns.
        """,
        
        "data_analysis": """
        # Data Analysis Request
        
        **Analyst Role:** {{ analyst.name }} ({{ analyst.department }})
        **Dataset:** {{ dataset.name }} ({{ dataset.size }} records)
        **Analysis Type:** {{ analysis_type }}
        
        ## Instructions
        1. Load data from {{ dataset.path }}
        2. Apply {{ preprocessing_steps | join(', ') }}
        3. Generate insights for {{ stakeholders | join(', ') }}
        
        ## Output Requirements
        - Executive summary
        - Key findings with statistics  
        - Actionable recommendations
        - Data quality assessment
        
        Format: {{ output_format | upper }}
        """
    }
    
    for name, template in examples.items():
        print(f"\n--- {name.upper()} ---")
        
        # Analyze
        analysis = analyze_template(template)
        print(f"Analysis: {analysis.complexity_score:.2f} complexity, {analysis.suggested_strategy.value} strategy")
        
        # Migrate with recommended strategy  
        result = migrate_template(template, analysis.suggested_strategy)
        print(f"Migration: {'Success' if result.success else 'Failed'}")
        
        if result.success:
            # Show a preview of the migrated template
            lines = [line for line in result.migrated_template.split('\n') if line.strip()][:5]
            print("Preview:")
            for line in lines:
                print(f"  {line}")
            if len([line for line in result.migrated_template.split('\n') if line.strip()]) > 5:
                print("  ...")
    
    print()

def run_migration_tests():
    """Run all migration tool tests."""
    print("Testing POML Template Migration Tools")
    print("=" * 60)
    print()
    
    test_template_analysis()
    test_template_migration() 
    test_batch_operations()
    test_convenience_functions()
    test_real_world_examples()
    
    print("=" * 60)
    print("Migration Tools Tests Complete")

if __name__ == "__main__":
    run_migration_tests()