#!/usr/bin/env python3
"""Final fix for YAML files with broken step structures."""

from pathlib import Path
import re

def fix_automated_testing_yaml():
    """Fix automated_testing_system.yaml specifically."""
    filepath = Path("examples/automated_testing_system.yaml")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the broken sections
    # First issue: content after line 117 needs to be a new step
    content = content.replace(
        '''    condition: "'unit' in {{test_types}}"
      
      Create tests for:''',
        '''    condition: "'unit' in {{test_types}}"
    tags: ["unit-tests", "testing"]
    
  # Step 5: Generate integration tests
  - id: generate_integration_tests
    action: |
      <AUTO>generate integration tests:
      Framework: {{test_framework}}
      Create tests for:'''
    )
    
    # Second issue: content after "Include test data setup and cleanup</AUTO>" needs proper structure
    content = content.replace(
        '''      Include test data setup and cleanup</AUTO>
    depends_on: [generate_test_plan]
      
      Generate:''',
        '''      Include test data setup and cleanup</AUTO>
    depends_on: [generate_test_plan]
    condition: "'integration' in {{test_types}}"
    tags: ["integration-tests", "testing"]
    
  # Step 6: Generate test fixtures
  - id: generate_fixtures
    action: |
      <AUTO>generate test fixtures and mocks:
      Generate:'''
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath.name}")

def fix_document_intelligence_yaml():
    """Fix document_intelligence.yaml specifically."""
    filepath = Path("examples/document_intelligence.yaml")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix: content after line 81 needs to be a new step
    content = content.replace(
        '''      Return classification with confidence
    depends_on: [discover_documents]
      Extract:''',
        '''      Return classification with confidence
    depends_on: [discover_documents]
    tags: ["classification", "ml"]
    
  # Step 3: Extract text content
  - id: extract_text
    action: |
      <AUTO>extract text from document {{loop_item}}:
      Document type: {{loop_item.classification}}
      Extract:'''
    )
    
    # Fix other sections similarly
    content = content.replace(
        '''      Return extracted text with metadata</AUTO>
    depends_on: [classify_documents]
      Identify:''',
        '''      Return extracted text with metadata</AUTO>
    depends_on: [classify_documents]
    tags: ["extraction", "ocr"]
    
  # Step 4: Analyze document structure
  - id: analyze_structure
    action: |
      <AUTO>analyze document structure:
      Text content: {{extract_text.result}}
      Identify:'''
    )
    
    content = content.replace(
        '''      Return document structure analysis</AUTO>
    depends_on: [extract_text]
      For each table:''',
        '''      Return document structure analysis</AUTO>
    depends_on: [extract_text]
    tags: ["structure", "layout"]
    
  # Step 5: Extract tables
  - id: extract_tables
    action: |
      <AUTO>extract and parse tables:
      Document structure: {{analyze_structure.result}}
      For each table:'''
    )
    
    content = content.replace(
        '''      Return structured table data</AUTO>
    depends_on: [analyze_structure]
      
      Extract:''',
        '''      Return structured table data</AUTO>
    depends_on: [analyze_structure]
    condition: "{{analyze_structure.result.table_count}} > 0"
    tags: ["tables", "data"]
    
  # Step 6: Extract entities
  - id: extract_entities
    action: |
      <AUTO>extract named entities from text:
      Text content: {{extract_text.result}}
      Extract:'''
    )
    
    content = content.replace(
        '''      Return entities with metadata</AUTO>
    depends_on: [extract_text]
      Detect:''',
        '''      Return entities with metadata</AUTO>
    depends_on: [extract_text]
    tags: ["entities", "nlp"]
    
  # Step 7: Detect PII
  - id: detect_pii
    action: |
      <AUTO>detect personally identifiable information:
      Text content: {{extract_text.result}}
      Detect:'''
    )
    
    content = content.replace(
        '''      Return PII findings and compliance report</AUTO>
    depends_on: [extract_text]
      Analyze:''',
        '''      Return PII findings and compliance report</AUTO>
    depends_on: [extract_text]
    tags: ["pii", "compliance"]
    
  # Step 8: Analyze content
  - id: analyze_content
    action: |
      <AUTO>analyze document content and meaning:
      Text: {{extract_text.result}}
      Entities: {{extract_entities.result}}
      Analyze:'''
    )
    
    content = content.replace(
        '''      Return comprehensive content analysis</AUTO>
    depends_on: [extract_entities]
      Create summary that:''',
        '''      Return comprehensive content analysis</AUTO>
    depends_on: [extract_entities]
    tags: ["analysis", "insights"]
    
  # Step 9: Generate summary
  - id: generate_summary
    action: |
      <AUTO>generate document summary:
      Content analysis: {{analyze_content.result}}
      Create summary that:'''
    )
    
    content = content.replace(
        '''      Return summary with metadata</AUTO>
    depends_on: [analyze_content]
      Find relationships:''',
        '''      Return summary with metadata</AUTO>
    depends_on: [analyze_content]
    tags: ["summary", "abstract"]
    
  # Step 10: Extract relationships
  - id: extract_relationships
    action: |
      <AUTO>extract entity relationships:
      Entities: {{extract_entities.result}}
      Content: {{analyze_content.result}}
      Find relationships:'''
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath.name}")

def fix_creative_writing_yaml():
    """Fix creative_writing_assistant.yaml specifically."""
    filepath = Path("examples/creative_writing_assistant.yaml")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix: content after line 204 needs to be a new step
    content = content.replace(
        '''      Maintain consistent voice
    depends_on: [outline_chapters]
      For each conversation:''',
        '''      Maintain consistent voice
    depends_on: [outline_chapters]
    condition: "{{write_detailed_chapters}} == true"
    tags: ["writing", "scenes"]
    
  # Step 9: Generate dialogue
  - id: generate_dialogue
    action: |
      <AUTO>write authentic dialogue:
      Characters: {{develop_characters.result}}
      Scene context: {{write_key_scenes.result}}
      For each conversation:'''
    )
    
    content = content.replace(
        '''      Make each character distinct</AUTO>
    depends_on: [write_key_scenes]
      Add:''',
        '''      Make each character distinct</AUTO>
    depends_on: [write_key_scenes]
    tags: ["dialogue", "character-voice"]
    
  # Step 10: Enhance descriptions
  - id: enhance_descriptions
    action: |
      <AUTO>enhance scene descriptions:
      Current scenes: {{write_key_scenes.result}}
      Dialogue: {{generate_dialogue.result}}
      Add:'''
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath.name}")

def fix_financial_analysis_yaml():
    """Fix financial_analysis_bot.yaml specifically."""
    filepath = Path("examples/financial_analysis_bot.yaml")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix: content after line 57 needs to be a new step
    content = content.replace(
        '''      Return indicators and patterns with signals</AUTO>
    depends_on: [collect_market_data]
      Calculate metrics:''',
        '''      Return indicators and patterns with signals</AUTO>
    depends_on: [collect_market_data]
    tags: ["technical", "indicators"]
    
  # Step 3: Fundamental analysis
  - id: fundamental_analysis
    action: |
      <AUTO>perform fundamental analysis for {{loop_item}}:
      Financial data: {{collect_market_data.result[loop_item].fundamentals}}
      Calculate metrics:'''
    )
    
    content = content.replace(
        '''      Return fundamental scores and analysis</AUTO>
    depends_on: [collect_market_data]
      Sources to analyze:''',
        '''      Return fundamental scores and analysis</AUTO>
    depends_on: [collect_market_data]
    condition: "{{include_fundamentals}} == true"
    tags: ["fundamental", "valuation"]
    
  # Step 4: Sentiment analysis
  - id: sentiment_analysis
    action: |
      <AUTO>analyze market sentiment for {{loop_item}}:
      Ticker: {{loop_item}}
      Sources to analyze:'''
    )
    
    content = content.replace(
        '''      Return sentiment analysis with key drivers</AUTO>
    depends_on: [collect_market_data]
      Calculate:''',
        '''      Return sentiment analysis with key drivers</AUTO>
    depends_on: [collect_market_data]
    tags: ["sentiment", "news"]
    
  # Step 5: Risk assessment
  - id: risk_assessment
    action: |
      <AUTO>assess portfolio risk metrics:
      Holdings: {{symbols}}
      Market data: {{collect_market_data.result}}
      Calculate:'''
    )
    
    content = content.replace(
        '''      Return comprehensive risk profile</AUTO>
    depends_on: [collect_market_data]
    tags: ["risk", "portfolio"]
    
  # Step 6: Generate price predictions
  - id: predictive_modeling''',
        '''      Return comprehensive risk profile</AUTO>
    depends_on: [collect_market_data]
    tags: ["risk", "portfolio"]
    
  # Step 6: Generate price predictions
  - id: predictive_modeling'''
    )
    
    content = content.replace(
        '''    condition: "{{include_predictions}} == true"
      
      Optimize for:''',
        '''    condition: "{{include_predictions}} == true"
    tags: ["prediction", "ml"]
    
  # Step 7: Portfolio optimization
  - id: portfolio_optimization
    action: |
      <AUTO>optimize portfolio allocation:
      Assets: {{symbols}}
      Risk profiles: {{risk_assessment.result}}
      Optimize for:'''
    )
    
    content = content.replace(
        '''      Return optimal weights and metrics</AUTO>
    depends_on: [risk_assessment]
    tags: ["portfolio", "optimization"]
    
  # Step 8: Generate trading signals
  - id: generate_signals''',
        '''      Return optimal weights and metrics</AUTO>
    depends_on: [risk_assessment]
    condition: "{{symbols|length}} > 1"
    tags: ["portfolio", "optimization"]
    
  # Step 8: Generate trading signals
  - id: generate_signals'''
    )
    
    content = content.replace(
        '''    depends_on: [technical_analysis, sentiment_analysis, risk_assessment]
      Test parameters:''',
        '''    depends_on: [technical_analysis, sentiment_analysis, risk_assessment]
    tags: ["signals", "trading"]
    
  # Step 9: Backtest strategy
  - id: backtest_strategy
    action: |
      <AUTO>backtest trading strategy:
      Signals: {{generate_signals.result}}
      Historical data: {{collect_market_data.result}}
      Test parameters:'''
    )
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed {filepath.name}")

def main():
    """Fix all remaining YAML files."""
    fix_automated_testing_yaml()
    fix_document_intelligence_yaml()
    fix_creative_writing_yaml()
    fix_financial_analysis_yaml()

if __name__ == "__main__":
    main()