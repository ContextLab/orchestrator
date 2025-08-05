# 🚀 Issue 187 Implementation Plan: create_parallel_queue Action for Dynamic Parallel Task Execution

## Overview
This plan provides a comprehensive implementation of the `create_parallel_queue` action that dynamically generates and executes tasks in parallel with real API integration and NO MOCK TESTS. The implementation integrates with existing systems and the recently completed Issue 189 (until/while loop conditions).

## 📋 Analysis of Existing Codebase Integration Points

### Core Integration Requirements
Based on codebase analysis, the parallel queue implementation must integrate with:

1. **Task System** (`src/orchestrator/core/task.py`): 
   - Existing `Task` dataclass with status tracking, dependencies, and metadata
   - Task execution lifecycle management
   - Output tracking and template metadata

2. **Loop Context System** (`src/orchestrator/core/loop_context.py`):
   - Named loop system with unlimited nesting depth
   - Cross-step loop variable access
   - Template integration with `$item`, `$index`, `$items` variables

3. **Control Flow System** (`src/orchestrator/control_flow/loops.py`):
   - ForLoopHandler patterns for expanding loops into tasks
   - ControlFlowAutoResolver for AUTO tag resolution
   - Loop state management and context tracking

4. **Until/While Conditions** (Issue 189 components):
   - EnhancedConditionEvaluator for complex condition evaluation
   - LoopCondition structured evaluation with caching
   - Until conditions that can evaluate across parallel tasks

5. **YAML Compiler** (`src/orchestrator/compiler/yaml_compiler.py`):
   - Task definition parsing and validation
   - AUTO tag detection and resolution
   - Pipeline object construction

6. **Tool System** (`src/orchestrator/tools/base.py`):
   - Tool execution patterns with headless-browser, terminal, etc.
   - Parameter validation and template rendering
   - Resource sharing across parallel executions

## 🏗️ Implementation Architecture

### Phase 1: Core Parallel Queue Infrastructure

#### 1.1 ParallelQueueTask Model
```python
@dataclass
class ParallelQueueTask(Task):
    """Task for parallel queue execution with until condition support."""
    
    # Queue generation
    on: str  # Expression to generate queue items (AUTO tags supported)
    max_parallel: int = 10
    
    # Action template
    action_loop: List[Dict[str, Any]] = field(default_factory=list)
    tool: Optional[str] = None
    
    # Until condition integration (Issue 189)
    until_condition: Optional[str] = None
    while_condition: Optional[str] = None
    
    # Runtime state
    queue_items: List[Any] = field(default_factory=list)
    active_subtasks: Dict[str, Task] = field(default_factory=dict)
    completed_items: Set[int] = field(default_factory=set)
    failed_items: Set[int] = field(default_factory=set)
    
    # Performance tracking
    parallel_execution_stats: Dict[str, Any] = field(default_factory=dict)
```

#### 1.2 ParallelQueueHandler
```python
class ParallelQueueHandler:
    """Handles create_parallel_queue execution with real concurrency."""
    
    def __init__(self, 
                 auto_resolver: ControlFlowAutoResolver,
                 loop_context_manager: GlobalLoopContextManager,
                 condition_evaluator: EnhancedConditionEvaluator):
        self.auto_resolver = auto_resolver
        self.loop_context_manager = loop_context_manager
        self.condition_evaluator = condition_evaluator  # From Issue 189
        
    async def execute_parallel_queue(self, 
                                    task: ParallelQueueTask,
                                    context: Dict[str, Any]) -> TaskResult:
        # 1. Resolve 'on' expression using real AUTO tag resolution
        # 2. Create subtasks with proper loop context
        # 3. Execute with concurrency limits using asyncio.Semaphore
        # 4. Integrate until/while condition evaluation
        # 5. Handle tool resource sharing
        # 6. Return aggregated results
```

### Phase 2: Real API Integration & Testing Strategy

#### 2.1 Queue Generation with Real AUTO Tags
- **Integration**: Use existing `ControlFlowAutoResolver` with real model calls
- **Test Pattern**: Use actual OpenAI/Anthropic APIs to resolve expressions like:
  ```yaml
  on: "<AUTO>create a list of every source URL in {{ document }}</AUTO>"
  ```
- **Real Examples**: Process actual research documents with hundreds of real URLs

#### 2.2 Parallel Tool Execution
- **Headless Browser**: Real Selenium/Playwright instances with resource pooling
- **Terminal**: Real bash command execution with sandboxing
- **API Calls**: Actual HTTP requests to external services
- **File Operations**: Real filesystem I/O with proper cleanup

#### 2.3 Until/While Condition Integration
- **Cross-task Evaluation**: Until conditions that depend on results from multiple parallel tasks
- **Real Model Calls**: Actual LLM evaluation of complex conditions like:
  ```yaml
  until: "<AUTO>all sources have been verified (or removed, if incorrect)</AUTO>"
  ```
- **Synchronization**: Proper coordination between parallel workers and condition evaluation

### Phase 3: Advanced Features & Optimizations

#### 3.1 Resource Management
```python
class ParallelResourceManager:
    """Manages shared resources across parallel executions."""
    
    async def acquire_tool_instance(self, tool_name: str) -> Tool:
        # Real resource pooling for tools like headless-browser
        # Connection pooling for database/API tools
        # Rate limiting for external API calls
        
    async def release_tool_instance(self, tool_name: str, instance: Tool):
        # Proper cleanup and resource return
```

#### 3.2 Performance Optimization
- **Streaming Processing**: Handle large queues without memory overflow
- **Dynamic Concurrency**: Adjust parallelism based on system load
- **Intelligent Batching**: Group similar operations for efficiency
- **Resource Sharing**: Share expensive resources (browser instances, DB connections)

#### 3.3 Error Handling & Recovery
- **Individual Failure Isolation**: Failed items don't stop the queue
- **Retry Logic**: Per-item retry with exponential backoff
- **Partial Results**: Meaningful results even with some failures
- **Resource Cleanup**: Guaranteed cleanup even on failure

## 🧪 Comprehensive Testing Strategy (NO MOCKS)

### Test Infrastructure
```python
class RealParallelQueueTester:
    """Real-world testing framework for parallel queues."""
    
    async def setup_real_environment(self):
        # Start real headless browser instances
        # Set up test databases with real data
        # Configure actual API endpoints
        # Create temporary directories for file operations
```

### Test Categories

#### 1. Basic Parallel Execution Tests
```python
async def test_real_parallel_queue_basic():
    """Test basic parallel execution with real HTTP requests."""
    yaml_content = '''
    steps:
      - id: fetch_urls
        action: create_parallel_queue
        parameters:
          on: ["https://httpbin.org/delay/1", "https://httpbin.org/delay/2", "https://httpbin.org/delay/3"]
          max_parallel: 3
          tool: web
          action_loop:
            - action: "<AUTO>fetch {{ $item }} and return status code</AUTO>"
    '''
    # Execute with REAL HTTP requests, verify timing and results
```

#### 2. Large-Scale Real Data Tests
```python
async def test_parallel_queue_research_pipeline():
    """Test with actual research document containing 100+ URLs."""
    # Use real research paper with actual citations
    # Extract real URLs using AUTO tag resolution
    # Verify each URL with real browser automation
    # Measure performance with real network latency
```

#### 3. Tool Integration Tests
```python
async def test_parallel_headless_browser():
    """Test parallel browser automation with real websites."""
    yaml_content = '''
    steps:
      - id: verify_sources
        action: create_parallel_queue
        parameters:
          on: "<AUTO>extract all Wikipedia URLs from {{ research_doc }}</AUTO>"
          max_parallel: 5
          tool: headless-browser
          action_loop:
            - action: "<AUTO>navigate to {{ $item }} and verify it loads correctly</AUTO>"
            - action: "<AUTO>extract page title from {{ $item }}</AUTO>"
          until: "<AUTO>all URLs have been verified or marked as broken</AUTO>"
    '''
    # Execute with real Selenium/Playwright instances
    # Verify actual Wikipedia pages
    # Handle real network timeouts and errors
```

#### 4. Until/While Condition Integration Tests
```python
async def test_parallel_queue_with_until_conditions():
    """Test until conditions with parallel execution and real LLM evaluation."""
    # Create parallel queue that processes items until LLM-evaluated condition is met
    # Use real model API calls for condition evaluation
    # Test synchronization between parallel workers and condition checker
    # Verify proper termination when condition is satisfied
```

#### 5. Error Handling & Recovery Tests
```python
async def test_parallel_queue_error_recovery():
    """Test error handling with real network failures and timeouts."""
    # Mix of valid and invalid URLs
    # Simulate real network conditions (slow, timeout, DNS failures)
    # Verify partial results and proper error reporting
    # Test resource cleanup after failures
```

#### 6. Performance & Scale Tests
```python
async def test_parallel_queue_scale():
    """Test with 1000+ real items to verify performance."""
    # Generate queue of 1000+ real API endpoints
    # Test memory usage and execution time
    # Verify resource limits and throttling
    # Measure actual throughput with real network I/O
```

### Real-World Integration Examples

#### Example 1: Research Paper Verification
```yaml
name: "Research Paper Source Verification"
steps:
  - id: extract_sources
    action: create_parallel_queue
    parameters:
      on: "<AUTO>extract all academic paper URLs from {{ input_document }}</AUTO>"
      max_parallel: 10
      tool: headless-browser
      action_loop:
        - action: "<AUTO>verify {{ $item }} is accessible and contains academic content</AUTO>"
          name: verify_source
        - action: "<AUTO>extract DOI and publication info from {{ $item }}</AUTO>"
          name: extract_metadata
        - action: "<AUTO>check if citation in original document accurately represents {{ $item }}</AUTO>"
          name: verify_citation
      until: "<AUTO>all sources verified and citation accuracy checked</AUTO>"
```

#### Example 2: Website Health Check
```yaml
name: "Website Monitoring"
steps:
  - id: health_check
    action: create_parallel_queue
    parameters:
      on: "<AUTO>read URLs from {{ monitoring_config }}</AUTO>"
      max_parallel: 20
      tool: web
      action_loop:
        - action: "<AUTO>check HTTP status of {{ $item }}</AUTO>"
        - action: "<AUTO>measure response time for {{ $item }}</AUTO>"
        - action: "<AUTO>verify SSL certificate for {{ $item }}</AUTO>"
      while: "{{ $iteration }} < {{ max_checks }}"
      until: "<AUTO>all critical services are healthy</AUTO>"
```

## 📚 Integration with Issue 189 Components

### Until/While Condition Evaluation
- **Reuse EnhancedConditionEvaluator**: Leverage structured condition evaluation from Issue 189
- **Cross-Parallel Synchronization**: Until conditions that evaluate results from all parallel tasks
- **Performance Optimization**: Use condition caching across parallel executions

### Template Resolution Integration
- **Loop Variables**: `$item`, `$index`, `$items` available in parallel contexts
- **Cross-Task References**: Access results from other parallel executions
- **Dynamic Condition Updates**: Conditions can reference intermediate results

## 🔧 Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement ParallelQueueTask model
- [ ] Create ParallelQueueHandler with basic execution
- [ ] Integrate with existing loop context system
- [ ] Add YAML compiler support for create_parallel_queue syntax

### Phase 2: Real API Integration (Week 2)  
- [ ] Implement real AUTO tag resolution for queue generation
- [ ] Add tool resource management and sharing
- [ ] Integrate with Issue 189 condition evaluation
- [ ] Create comprehensive real-world test suite

### Phase 3: Advanced Features (Week 3)
- [ ] Add until/while condition support for parallel queues
- [ ] Implement performance optimizations and resource pooling
- [ ] Add error handling and recovery mechanisms
- [ ] Create debugging and monitoring tools

### Phase 4: Production Hardening (Week 4)
- [ ] Large-scale testing with 1000+ item queues
- [ ] Performance benchmarking and optimization
- [ ] Memory usage optimization for large queues
- [ ] Documentation and example creation

## 🎯 Success Criteria

### Functional Requirements
- ✅ Parse and execute create_parallel_queue YAML syntax
- ✅ Generate queues using real AUTO tag resolution  
- ✅ Execute items in parallel with configurable concurrency
- ✅ Support until/while conditions with real LLM evaluation
- ✅ Integrate with all existing tools (headless-browser, terminal, etc.)
- ✅ Handle errors gracefully with partial results

### Performance Requirements
- ✅ Handle 1000+ item queues efficiently
- ✅ Maintain memory usage < 1GB for large queues
- ✅ Support 50+ concurrent parallel executions
- ✅ Respond to until condition changes within 1 second

### Integration Requirements
- ✅ Full compatibility with Issue 189 until/while conditions
- ✅ Seamless integration with existing loop context system
- ✅ Support for all existing tools and resource sharing
- ✅ Proper template variable resolution across parallel tasks

### Testing Requirements
- ✅ 100+ test cases covering all scenarios
- ✅ Zero mock tests - all real API/service integration
- ✅ Performance benchmarks with real-world data
- ✅ Error recovery testing with actual failure conditions

## 🚀 Ready to Implement

This plan provides a comprehensive, production-ready implementation of parallel queue functionality with full integration into the existing orchestrator ecosystem. All testing will use real APIs, services, and data - NO MOCKS OR SIMULATIONS.

The implementation leverages the robust foundation from Issue 189 and extends it with powerful parallel execution capabilities that will enable sophisticated real-world automation pipelines.

**Next Step**: Please review this plan and I'll begin implementation with Phase 1!