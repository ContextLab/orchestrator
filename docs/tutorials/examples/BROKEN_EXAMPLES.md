# Broken Examples Report

## Overview
This document lists examples that are currently broken and need to be fixed or removed.

## Broken Examples

### 1. simple_pipeline.yaml
**Issues:**
- Uses non-existent actions: `generate`, `analyze`, `transform`
- Template variable `input_topic` is not properly passed from context
- Version format was incorrect (fixed to "1.0.0")

**Status:** BROKEN - Uses fictional actions that don't map to real tools

### 2. data_processing.yaml  
**Issues:**
- Uses non-existent actions: `ingest`, `clean`, `quality_check`, `export`, `report`
- These actions don't correspond to any real tools in the system

**Status:** BROKEN - Completely fictional pipeline

### 3. code_optimization.yaml
**Status:** NOT TESTED - Likely broken based on pattern

### 4. research-report-template.yaml
**Status:** NOT TESTED - May work if it uses real tools

### 5. research_assistant_with_pdf.py
**Status:** NOT TESTED - Python example, not YAML

## Root Causes

1. **Fictional Actions**: Most examples use made-up actions that don't exist in the real tool system
2. **Template Context Issues**: Context variables aren't being properly passed to templates
3. **Orchestrator Initialization**: The orchestrator takes too long to initialize (timeout issues)

## Recommendations

1. **Remove or Update Broken Examples**: 
   - `simple_pipeline.yaml` - Replace with real tool usage
   - `data_processing.yaml` - Replace with real data processing using actual tools

2. **Create Working Examples**:
   - Web search and summarization (using web-search and report-generator)
   - File processing (using file-system and data-processing)
   - Validation pipeline (using the new validation tool)

3. **Fix Core Issues**:
   - Investigate orchestrator initialization timeout
   - Fix template context passing
   - Ensure all examples use real, tested tools