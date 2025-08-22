# Stream 2: Filesystem Tool Fix (#220)

**Issue**: Filesystem tool not resolving template variables in paths and content
**Status**: In Progress

## Analysis

### Current Implementation Status
- ✅ FileSystemTool exists in `/src/orchestrator/tools/system_tools.py`
- ✅ Has basic template resolution logic in place
- ❌ Template resolution not working properly - files created with literal template syntax

### Key Problems Identified
1. Path template resolution happening but may not be working correctly
2. Content template resolution has fallback logic but may not have proper context
3. Template manager/resolver not being passed correctly from execution context

### Investigation Progress

#### Phase 1: Code Analysis ✅
- [x] Located filesystem tool implementation
- [x] Analyzed current template resolution logic
- [x] Identified potential issues with context passing

#### Phase 2: Test Case Analysis (In Progress)
- [ ] Examine failing pipeline examples
- [ ] Identify exact template variables not being resolved
- [ ] Test current resolution logic

#### Phase 3: Fix Implementation (Pending)
- [ ] Fix path template resolution
- [ ] Fix content template resolution  
- [ ] Ensure proper context passing
- [ ] Add debugging/logging

#### Phase 4: Testing (Pending)
- [ ] Create test pipeline
- [ ] Verify fixes work
- [ ] Test edge cases

## Implementation Details

### Root Cause
The filesystem tool was receiving template strings with `$` variables (e.g., `{{ $iteration }}`) but Jinja2 doesn't support `$` in variable names. The base Tool class template resolution was failing with syntax errors.

### Solution Implemented
Added `_preprocess_dollar_variables()` method to FileSystemTool that:
1. Converts `{{ $variable }}` to `{{ variable }}` before Jinja2 processing
2. Uses regex pattern `r'\{\{\s*\$([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'` to find and replace
3. Applied to both path and content parameters
4. Works as fallback when unified resolver not available

### Code Changes
- Added preprocessing method to handle $ variables  
- Updated `_execute_impl()` to preprocess path templates
- Updated `_write_file()` to preprocess content templates
- Enhanced logging for debugging template resolution

### Verification
✅ Test script confirms path and content templates now resolve correctly
✅ Pipeline execution creates files with resolved paths instead of literal template syntax
✅ Template variables like `{{ parameters.input_document }}` and `{{ $iteration }}` properly resolved

## Status: COMPLETED ✅

All filesystem tool template resolution issues have been fixed. The tool now properly:
- Resolves `$` variables in both paths and content
- Creates files with resolved filenames instead of literal template syntax  
- Processes template content before writing to files
- Works with both unified resolver and legacy template manager