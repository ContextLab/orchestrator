# Phase 3: Data Structure Fixes

## Objective
Fix data access patterns and template resolution issues identified in test_data_processing.py.

## Root Issues
- Result structure access changes (result.steps → result["steps"])
- Template variable resolution (load_data.content → load_data["content"])
- Pipeline context and metadata format changes
- Validation schema updates

## Action Plan

### 3.1: Result Structure Access Pattern (Priority: Critical)
**Issue**: Tests expect object attribute access but results are now dictionaries
**Pattern Already Applied**: `result.steps[key]` → `result["steps"][key]`
**Status**: ✅ Applied to test_data_processing.py via sed command
**Next**: Apply systematically to all remaining test files

**Command to Scale**:
```bash
find tests/ -name "*.py" -exec sed -i '' 's/result\.steps\[/result["steps"][/g' {} +
find tests/ -name "*.py" -exec sed -i '' 's/result\.metadata\./result["metadata"][/g' {} +
find tests/ -name "*.py" -exec sed -i '' 's/result\.context\./result["context"][/g' {} +
```

### 3.2: Template Variable Resolution (Priority: High)
**Issue**: Template variables like `{{ load_data.content }}` fail because load_data is dict not object
**Current Error**: `'dict object' has no attribute 'content'`
**Investigation Needed**: 
- Determine if templates should use `load_data["content"]` or `load_data.content`
- Check if TemplateManager needs to handle dict attribute access
- Verify pipeline result structure format

### 3.3: Pipeline Context Changes (Priority: Medium)
**Areas to Check**:
- Pipeline metadata format
- Execution context structure  
- Step result aggregation
- Error context information

### 3.4: Validation Schema Updates (Priority: Medium)
**Issues**:
- Data validation expecting different formats
- Schema validation rule changes
- Type checking updates

## Systematic Fix Commands

### 3.1: Apply Result Access Pattern to All Tests
```bash
# Fix result.steps pattern across all test files
find tests/ -name "*.py" -exec sed -i '' 's/result\.steps\[/result["steps"][/g' {} +

# Fix result.metadata pattern  
find tests/ -name "*.py" -exec sed -i '' 's/result\.metadata\[/result["metadata"][/g' {} +

# Fix result.context pattern
find tests/ -name "*.py" -exec sed -i '' 's/result\.context\[/result["context"][/g' {} +
```

### 3.2: Test Impact Analysis
```bash
# Run tests to see template variable failures
python -m pytest tests/ --tb=no -x | grep -E "(content|template|variable)"
```

## Success Criteria
- [ ] All result.attribute access patterns converted to result["key"] 
- [ ] Template variable resolution works consistently
- [ ] Pipeline context accessed correctly across all tests
- [ ] Validation schemas aligned with current data structures
- [ ] 95% of tests pass data structure compatibility

## Estimated Impact
- **Before**: 90-95% of tests fail on data access patterns
- **After**: 95-98% of tests access data correctly