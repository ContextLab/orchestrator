# Examples Status Report

## Overview
This document tracks the status of example pipelines and their fixes.

**Last Updated:** 2025-01-22

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

## Fixed Examples

The following working examples have been created to replace the broken ones:

### 1. working_web_search.yaml ✓
**Purpose:** Web search and summary pipeline
**Tools Used:** web-search, report-generator
**Status:** WORKING

### 2. simple_data_processing.yaml ✓
**Purpose:** Read CSV, process data, save results
**Tools Used:** filesystem, data-processing
**Status:** CREATED - Replaces broken data_processing.yaml

### 3. validation_pipeline.yaml ✓
**Purpose:** Validate data and extract structured information
**Tools Used:** filesystem, validation
**Status:** CREATED - Demonstrates validation tool usage

### 4. research_pipeline.yaml ✓
**Purpose:** Complete research workflow with PDF generation
**Tools Used:** web-search, headless-browser, report-generator, filesystem, pdf-compiler
**Status:** CREATED - Advanced example with conditional steps

### 5. terminal_automation.yaml ✓
**Purpose:** System information gathering and automation
**Tools Used:** terminal, filesystem
**Status:** CREATED - Shows terminal command execution

## Recommendations

1. **Remove Old Broken Examples**: 
   - Delete `simple_pipeline.yaml` (uses fictional actions)
   - Delete `data_processing.yaml` (uses fictional actions)
   - Move broken examples to an archive folder

2. **Template Context Issues**:
   - Need to ensure template variables are properly passed
   - Use parameters section for user inputs
   - Use Jinja2 filters like `| from_json`, `| to_json`, `| slugify`

3. **Testing Strategy**:
   - Test each example with real tool execution
   - Verify all dependencies are satisfied
   - Document any required setup (e.g., data files, config files)