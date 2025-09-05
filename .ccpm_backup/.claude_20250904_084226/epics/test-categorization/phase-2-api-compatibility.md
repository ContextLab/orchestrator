# Phase 2: API Compatibility Fixes

## Objective
Update tests to match current API interfaces and method signatures after refactor #307.

## Root Issues
- Method signature changes (execute_pipeline_from_dict, etc.)
- Return type modifications (strings vs objects)
- Interface updates in control systems
- Tool handler API changes

## Action Plan

### 2.1: Orchestrator API Updates (Priority: Critical)
**Common Issues**:
- `execute_pipeline_from_dict()` signature changes
- `execute_pipeline()` parameter updates  
- Result structure modifications
- Context passing changes

**Pattern to Apply**:
```python
# Old pattern
result = await orchestrator.execute_pipeline(pipeline_dict, **inputs)

# New pattern  
result = await orchestrator.execute_pipeline_from_dict(pipeline_dict, inputs)
```

### 2.2: Model Registry API Updates (Priority: High)
**Issues Fixed in Phase 1, verify propagation**:
- `select_model()` returns string, not Model object
- `get_model()` is async and must be awaited
- Provider registration changes

### 2.3: Control System Interface Updates (Priority: High)
**Files to Update**:
- Tests using HybridControlSystem directly
- ModelBasedControlSystem tests
- Custom control system tests

**Common Changes**:
- Constructor parameters
- Method signatures
- Return value structures

### 2.4: Tool Handler API Changes (Priority: Medium)
**Tool Categories**:
- File system tools
- Validation tools  
- Data processing tools
- External integration tools

**Common Issues**:
- Parameter naming changes
- Return format modifications
- Error handling updates

## Success Criteria
- [ ] All orchestrator API calls use correct signatures
- [ ] All model registry calls properly handle async patterns
- [ ] Control system tests use current interfaces
- [ ] Tool handler tests match current APIs
- [ ] 90% of executable tests now pass basic API compatibility

## Estimated Impact
- **Before**: 80-90% of tests execute but fail on API mismatches
- **After**: 90-95% of tests pass API compatibility checks