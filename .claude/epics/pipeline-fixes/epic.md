---
name: pipeline-fixes
status: backlog
created: 2025-08-22T13:27:39Z
progress: 0%
prd: .claude/prds/pipeline-fixes.md
github: https://github.com/ContextLab/orchestrator/issues/234
updated: 2025-08-22T13:39:11Z
---

# Epic: pipeline-fixes

## Overview
Technical implementation to fix critical pipeline execution issues affecting output quality and reliability. Leverages existing UnifiedTemplateResolver and TemplateValidator components already created, focusing on integration, cleanup, and validation of 25 example pipelines.

## Architecture Decisions

### Key Technical Decisions
- **Leverage Existing Components**: Use already-implemented UnifiedTemplateResolver and TemplateValidator instead of building new systems
- **Incremental Migration**: Update tools one-by-one to use unified resolver without breaking compatibility
- **Centralized Logging**: Replace all debug prints with proper Python logging at appropriate levels
- **Standardized Tool Interface**: Enforce consistent return format through base Tool class validation
- **Automated Testing**: Use pytest with real API calls (no mocks) to validate all pipelines

### Technology Choices
- **Template Engine**: Continue with Jinja2, extend existing TemplateManager
- **Logging**: Python standard logging with configurable levels
- **Testing**: pytest with real resources (following CLAUDE.md guidelines)
- **Validation**: JSON Schema for output validation

### Design Patterns
- **Decorator Pattern**: For output sanitization without modifying core logic
- **Strategy Pattern**: For different error recovery mechanisms
- **Chain of Responsibility**: For template context resolution hierarchy

## Technical Approach

### Core Components

1. **Template Resolution Integration**
   - Integrate UnifiedTemplateResolver into all control systems
   - Update FileSystemTool to use resolver before operations
   - Ensure loop context variables properly injected

2. **Debug Cleanup**
   - Global search/replace of print statements with logging
   - Add LOG_LEVEL environment variable support
   - Create debug mode flag for development

3. **Tool Standardization**
   - Update Tool base class with validation decorator
   - Modify generate-structured to return actual objects
   - Fix DataProcessingTool CSV handling
   - Implement ValidationTool quality_check schema

4. **Output Sanitization**
   - Create OutputSanitizer class with regex patterns
   - Remove conversational markers ("Certainly!", "Here is...")
   - Strip debug content from final outputs

### Backend Services
- No new API endpoints required
- Update existing tool execute() methods
- Enhance error handling in orchestrator.py
- Modify control systems for consistent template handling

### Infrastructure
- No deployment changes needed
- Logging configuration for production
- Test infrastructure for 25 pipelines
- CI/CD pipeline validation checks

## Implementation Strategy

### Development Phases
1. **Immediate Fixes**: Debug removal and critical bugs
2. **Tool Updates**: Standardize interfaces and fix data handling
3. **Pipeline Validation**: Test and fix all 25 example pipelines
4. **Documentation**: Update guides and references

### Risk Mitigation
- Run full test suite after each change
- Keep changes minimal and focused
- Maintain backward compatibility
- Test with real API calls per CLAUDE.md

### Testing Approach
- Real API calls for all tests (no mocks)
- Validate actual output files
- Check for unrendered templates
- Quality scoring on outputs

## Task Breakdown Preview

Simplified task structure (max 10 tasks):

- [ ] Task 1: Remove all debug output and implement proper logging
- [ ] Task 2: Integrate UnifiedTemplateResolver into remaining tools
- [ ] Task 3: Fix generate-structured to return objects instead of strings
- [ ] Task 4: Standardize tool return format (result/success/error)
- [ ] Task 5: Implement OutputSanitizer for clean outputs
- [ ] Task 6: Fix DataProcessingTool CSV handling and ValidationTool schemas
- [ ] Task 7: Add compile-time validation to YAMLCompiler
- [ ] Task 8: Create automated test suite for all 25 pipelines
- [ ] Task 9: Fix all pipeline-specific issues (#158-#182)
- [ ] Task 10: Update documentation and create migration guide

## Dependencies

### Internal Dependencies
- Template System Epic (#225) - Already partially completed
- UnifiedTemplateResolver - Already implemented
- TemplateValidator - Already implemented

### External Dependencies
- None - all fixes are internal to orchestrator

### Prerequisite Work
- Template system fixes (Issues #219, #220) - COMPLETED

## Success Criteria (Technical)

### Performance Benchmarks
- Pipeline execution time unchanged or improved
- Memory usage stable
- No performance regression

### Quality Gates
- Zero unrendered templates in outputs
- No debug statements in logs
- All 25 pipelines pass validation
- Test coverage >80%

### Acceptance Criteria
- All tools return consistent format
- Clean, professional outputs
- Meaningful error messages
- No breaking changes

## Estimated Effort

### Overall Timeline
- **Total Duration**: 2-3 weeks (reduced from 5 weeks in PRD)
- **Developer Resources**: 1-2 developers
- **Parallel Work**: Tasks 1-6 can be done in parallel

### Critical Path Items
1. Debug removal (affects all code)
2. Tool standardization (blocks pipeline fixes)
3. Pipeline validation (final verification)

### Resource Requirements
- Developer time for implementation
- API credits for testing (minimal)
- Review time for documentation

## Tasks Created
- [ ] #235 - Remove debug output and implement logging (parallel: true)
- [ ] #236 - Integrate UnifiedTemplateResolver into tools (parallel: true)
- [ ] #237 - Fix generate-structured return format (parallel: true)
- [ ] #238 - Standardize tool return format (parallel: true)
- [ ] #239 - Implement OutputSanitizer (parallel: true)
- [ ] #240 - Fix DataProcessingTool and ValidationTool (parallel: true)
- [ ] #241 - Add compile-time validation (parallel: false, depends on 235-240)
- [ ] #242 - Create automated test suite (parallel: false, depends on 241)
- [ ] #243 - Fix pipeline-specific issues (parallel: false, depends on 235-242)
- [ ] #244 - Update documentation (parallel: false, depends on 243)

Total tasks: 10
Parallel tasks: 6
Sequential tasks: 4
Estimated total effort: 76 hours
