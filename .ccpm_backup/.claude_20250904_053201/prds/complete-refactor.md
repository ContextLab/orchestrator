---
name: complete-refactor
description: Complete architectural overhaul of the orchestrator toolbox for improved maintainability, performance, and user experience
status: backlog
created: 2025-08-30T03:49:45Z
github_issue: 307
---

# PRD: Complete Orchestrator Refactor

> **Reference**: GitHub Issue #307 - "Major refactor"  
> **URL**: https://github.com/ContextLab/orchestrator/issues/307

## Executive Summary

The orchestrator toolbox requires a comprehensive architectural overhaul to address fundamental design misalignment, maintainability challenges, and broken core functionality as detailed in GitHub Issue #307. This refactor will create a streamlined, robust system centered around YAML-defined pipeline execution with LangGraph integration, simplified architecture, and comprehensive quality assurance.

**Value Proposition**: Transform the orchestrator from a complex, difficult-to-maintain system into a clean, intuitive, and reliable AI pipeline orchestration platform that aligns with its original design intent and supports multi-platform deployment (macOS, Ubuntu, Windows) on Python 3.9+.

## Problem Statement

### Current Pain Points

**Architectural Issues**:
- Code base has become clunky and difficult to debug/maintain
- Fundamental design has drifted from original toolbox intent
- Complex template resolution system with multiple overlapping layers
- Tight coupling and circular dependencies throughout the system

**Functional Issues**:
- Core functionality is missing or broken
- Inconsistent pipeline execution paths
- Complex control flow management that's hard to follow
- Legacy code patterns mixed with modern approaches

**Developer Experience Issues**:
- Difficult to debug and extend
- Inconsistent patterns across the codebase
- Poor separation of concerns
- Limited test coverage and quality assurance

### Why This Matters Now

The current system's complexity is hindering development velocity and preventing the toolbox from fulfilling its core mission of providing reliable AI pipeline orchestration. Without this refactor, technical debt will continue accumulating, making future improvements increasingly difficult.

## User Stories

### Primary User Personas

**Pipeline Developer**: Creates and maintains YAML pipelines for AI workflows
**System Integrator**: Integrates orchestrator into larger systems
**Maintenance Engineer**: Debugs, monitors, and maintains running systems

### Core User Journeys

**Pipeline Creation Flow**:
1. Developer writes intuitive YAML pipeline definition
2. System compiles pipeline with comprehensive validation
3. Pipeline runs reliably with clear error handling
4. Quality control provides actionable feedback

**Pipeline Execution Flow**:
1. User invokes compiled pipeline with parameters
2. System executes steps with proper dependency management
3. Real-time monitoring shows clear execution status
4. Results are delivered with comprehensive logging

**Debugging & Maintenance Flow**:
1. Issue occurs during pipeline execution
2. Clear error messages guide to root cause
3. Comprehensive logs provide execution context
4. Fix can be applied without architectural knowledge

## Requirements

### Functional Requirements

#### Core Pipeline System
- **YAML Pipeline Definition**: Support comprehensive pipeline specification with header, steps, dependencies, and control flow
- **LangGraph Integration**: Automatic StateGraph compilation from YAML definitions
- **Model Management**: Support for OpenAI, Anthropic, Gemini, HuggingFace, and Ollama models with intelligent selection
- **Tool Integration**: Comprehensive tool registry with automatic setup/installation
- **Dependency Analysis**: Automatic parallelization and sequential execution planning
- **Variable Management**: Structured variable handling with LangChain integration

#### Execution Engine  
- **Pipeline Compilation**: Validate, optimize, and prepare pipelines for execution
- **Runtime Execution**: Reliable step execution with proper error handling
- **Quality Control**: Automated output quality assessment and reporting
- **Logging System**: Comprehensive execution tracking with markdown export
- **Control Flow**: Support for conditional execution, success/failure routing

#### Developer Experience
- **Clear Error Messages**: Actionable feedback for compilation and runtime errors  
- **Progress Monitoring**: Real-time execution status with progress indicators
- **Debugging Tools**: Comprehensive logging and state inspection capabilities
- **Documentation**: Complete API documentation and usage examples

### Non-Functional Requirements

#### Performance
- **Compilation Speed**: Pipeline compilation with comprehensive validation and model setup (may take longer for large model downloads with progress bars)
- **Execution Efficiency**: Minimal overhead during pipeline execution
- **Resource Management**: Efficient model loading/unloading and memory management with progress indication
- **Parallel Execution**: Optimal parallelization of independent pipeline steps
- **Multi-Platform Support**: Full compatibility across macOS, Ubuntu, and Windows on Python 3.9+

#### Reliability
- **Error Recovery**: Graceful handling of model failures, network issues, and resource constraints
- **Data Integrity**: Reliable variable state management throughout execution
- **Resource Cleanup**: Proper cleanup of temporary resources and connections
- **Testing Coverage**: 100% unit test coverage for all user-facing functionality

#### Security
- **API Key Management**: Secure storage and handling of model API credentials
- **Sandbox Execution**: Safe execution of user-defined pipeline steps
- **Permission Management**: Appropriate access controls for file system and external resources

#### Usability
- **Intuitive YAML Syntax**: Clear, self-documenting pipeline definition format
- **Clear Documentation**: Comprehensive guides and API reference
- **Error Messages**: Human-readable error descriptions with suggested fixes
- **Migration Path**: Clear upgrade path from existing system

## Success Criteria

### Measurable Outcomes

**Development Velocity**:
- 50% reduction in time to create new pipelines
- 75% reduction in debugging time for pipeline issues
- 90% reduction in code complexity metrics

**System Reliability**:
- 99%+ pipeline compilation success rate
- 95%+ pipeline execution success rate for valid inputs
- Zero critical bugs in core execution engine

**Code Quality**:
- 100% unit test coverage
- All user-facing functions tested with real API calls
- Comprehensive documentation for all public APIs

### Key Performance Indicators

- Pipeline compilation time (target: <30s for complex pipelines)
- Error resolution time (target: <5 minutes for common issues)
- Developer onboarding time (target: <1 hour to create first pipeline)
- System maintainability score (target: 8/10 on standard metrics)

## Technical Architecture

### Pipeline Specification (from GitHub Issue #307)

The refactored system will implement the complete pipeline specification detailed in the GitHub issue, including:

#### YAML Pipeline Header Example
```yaml
id: research_report_pipeline
name: Research Report Generator
description: Generate comprehensive research reports on specified topics using multi-model collaboration
orchestrator: ollama:llama3.2-70b
default_model: openai:gpt-5
experts:
  - web_search: gemini:gemini-2.0-flash-thinking-exp
  - content_generation: anthropic:claude-3-5-sonnet-20241022
  - quality_review: openai:gpt-5
selection_schema: balanced
inputs:
  - output_dir: /examples/outputs/research_report_pipeline/
  - topic: None
  - style: "academic"
  - max_pages: 10
```

#### Pipeline Step Example
```yaml
- id: research_phase
  name: Research Phase
  description: Conduct comprehensive research on the specified topic using web search and analysis
  dependencies: []
  tools: [web_search, document_analyzer]
  model: gemini:gemini-2.0-flash-thinking-exp
  personality: "research_specialist"
  condition: "topic is not None"
  vars:
    - research_findings: "Structured research data with sources and key insights"
    - source_quality_score: "Float between 0-1 indicating research source reliability"
  products:
    - research_findings: "research_summary.md"
    - source_quality_score: "quality_metrics.json"
```

#### API Usage Examples
```python
import orchestrator as orc

# Compile pipeline with comprehensive validation
pipeline = orc.compile("research_report.yaml")

# Access pipeline metadata
print(pipeline.intention)  # LLM-generated intention summary
print(pipeline.architecture)  # LLM-generated architecture description

# Run with inputs
result = pipeline.run(
    topic="using agentic AI to solve society-scale problems in the modern era",
    style="professor",
    max_pages=15
)

# Access results
print(result.outputs)  # Dictionary of step outputs
orc.log.markdown(result.log)  # Human-readable execution log
result.qc()  # Automated quality control report
```

### Core Components

#### Pipeline Compiler
- **YAML Parser**: Convert YAML definitions to internal representation
- **Dependency Analyzer**: Build execution graph and identify parallelization opportunities
- **Model Resolver**: Select appropriate models based on requirements and preferences
- **Tool Manager**: Initialize and configure required tools and resources
- **Validation Engine**: Comprehensive compilation-time error detection

#### Execution Engine
- **LangGraph Runner**: Execute StateGraph with proper state management
- **Step Executor**: Individual step execution with error handling
- **Variable Manager**: Thread-safe variable state management
- **Progress Tracker**: Real-time execution monitoring and reporting

#### Quality Assurance System
- **Output Validator**: Automated quality assessment of pipeline results
- **Logging System**: Comprehensive execution tracking and reporting
- **Error Handler**: Intelligent error recovery and user feedback
- **Performance Monitor**: Execution metrics and optimization recommendations

### Data Flow

1. **Compilation Phase**: YAML → Internal Representation → LangGraph StateGraph → Compiled Pipeline
2. **Execution Phase**: Input Parameters → State Initialization → Step Execution → Output Collection → Quality Control
3. **Monitoring Phase**: Real-time Logging → Progress Tracking → Error Detection → User Notification

## Development Philosophy

### In-Place Development Approach
**Critical Requirement**: All changes must be made *in place* rather than creating backup files or redundant functions/files. This refactor will:

- **Complete Component Replacement**: Large components of the toolbox will be completely scrapped or rewritten
- **No Redundancy**: No backup functions, parallel implementations, or temporary compatibility layers
- **Clean Transitions**: Each component replaced entirely before moving to the next

### Repository Cleanliness Standards
**100% Clean Repository Requirement**: Throughout development and testing, the toolbox must be kept completely clean:

- **Structural Integrity**: Design appropriate repository structure and enforce it throughout development
- **Continuous Validation**: Automated checks to ensure adherence to clean structure at all times
- **No Temporary Files**: No development artifacts, backup files, or temporary implementations
- **Professional Standards**: Repository must maintain production-ready appearance throughout development

### Progress Indication Requirements
**User Experience During Long Operations**:
- **Model Downloads**: Progress bars for large model downloads (which may significantly exceed 30 seconds)
- **Compilation Status**: Clear indication of compilation phases and current operations
- **Resource Setup**: Progress feedback for tool initialization and dependency installation
- **Multi-Platform Compatibility**: Consistent progress indication across macOS, Ubuntu, and Windows

## Implementation Phases

### Phase 1: Core Architecture (Weeks 1-4)
- Design and implement new pipeline compiler
- Create LangGraph integration layer
- Build basic execution engine
- Establish testing framework

### Phase 2: Pipeline Features (Weeks 5-8)
- Implement comprehensive YAML specification
- Add model management and selection
- Build tool integration system
- Create variable management system

### Phase 3: Quality & Monitoring (Weeks 9-12)
- Implement quality control system
- Build comprehensive logging
- Add progress monitoring
- Create debugging tools

### Phase 4: Migration & Optimization (Weeks 13-16)
- Create migration utilities
- Performance optimization
- Comprehensive testing
- Documentation completion

## Constraints & Assumptions

### Technical Constraints
- Must maintain backward compatibility during transition period
- API keys stored in ~/.orchestrator/.env or GitHub secrets
- Dependencies installed on-demand to minimize setup overhead
- Support for existing model providers (OpenAI, Anthropic, etc.)

### Resource Constraints
- Development team availability
- Model API rate limits and costs
- Storage requirements for model downloads
- Testing infrastructure needs

### Timeline Constraints
- 16-week development timeline
- Phased rollout to minimize disruption
- User feedback integration periods

## Dependencies

### External Dependencies
- **Model Providers**: OpenAI, Anthropic, Google, HuggingFace, Ollama
- **Infrastructure**: Docker, Pandoc, various system utilities
- **Frameworks**: LangGraph, LangChain, StateGraph

### Internal Dependencies
- **Existing User Base**: Migration and training requirements
- **Current Pipelines**: Compatibility and migration needs
- **Team Coordination**: Multiple engineers working on different components

## Out of Scope

### Explicitly Excluded
- **UI Development**: Focus on API and command-line interface only
- **Multi-tenant System**: Single-user deployment model maintained
- **Real-time Collaboration**: No concurrent user support
- **Legacy Pipeline Support**: Full backward compatibility not guaranteed
- **Custom Model Training**: Only inference with existing models

### Future Considerations
- Web-based pipeline editor
- Multi-user deployment options
- Advanced analytics and monitoring dashboard
- Integration with workflow orchestration platforms

## Risk Mitigation

### Technical Risks
- **Model API Changes**: Abstract model interfaces to minimize impact
- **LangGraph Evolution**: Monitor upstream changes and maintain compatibility
- **Performance Degradation**: Comprehensive benchmarking and optimization

### Project Risks
- **Scope Creep**: Strict adherence to defined requirements and phases
- **Timeline Pressure**: Phased delivery with MVP at each stage
- **User Adoption**: Clear migration path and comprehensive documentation

### Quality Risks
- **Regression Issues**: 100% test coverage requirement with real API testing
- **User Experience**: Continuous feedback integration and usability testing
- **Documentation Gaps**: Documentation-driven development approach

## Validation & Testing Strategy

### Testing Requirements
- **Unit Tests**: 100% coverage for all user-facing functionality
- **Integration Tests**: Real API calls with actual model providers
- **End-to-End Tests**: Complete pipeline execution scenarios
- **Performance Tests**: Benchmarking against current system
- **User Acceptance Tests**: Validation with real-world use cases

### Quality Gates
- All functions tested with real resources before production
- Manual verification of outputs for user-facing features
- Human review required for critical functionality sign-off
- Comprehensive regression testing for each release

## Success Validation

Upon completion, this refactor will deliver:

1. **Streamlined Architecture**: Clean, maintainable codebase aligned with original design intent
2. **Reliable Execution**: Robust pipeline compilation and execution with comprehensive error handling
3. **Developer Experience**: Intuitive YAML syntax, clear documentation, and efficient debugging tools
4. **Quality Assurance**: Automated testing, real-world validation, and comprehensive monitoring

The refactored orchestrator will transform from a complex, difficult-to-maintain system into a professional-grade AI pipeline orchestration platform that enables rapid development and reliable execution of sophisticated AI workflows.