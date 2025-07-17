# Orchestrator Framework Redesign Plan

**Goal:** Transform the framework into a fully declarative, self-configuring pipeline system where EVERYTHING is defined in YAML files and executed automatically.

## Current Problems

### 1. **Too Much Manual Setup**
- Requires custom control systems for each pipeline type
- Manual tool registration and configuration
- Complex Python code for simple workflows
- Users must understand framework internals

### 2. **Not Truly Declarative**
- YAML files are just templates, not executable specifications
- Business logic scattered across Python classes
- Tool execution hardcoded in control systems
- No automatic prompt generation for abstract tasks

### 3. **Poor Abstraction**
- Users write Python instead of describing what they want
- Framework doesn't leverage AI models for task interpretation
- Manual prompt engineering required
- Tools not automatically discoverable/composable

## Vision: Fully Declarative Framework

### Core Principle
```yaml
# This should be ALL the user needs to write:
name: "Research Assistant"
inputs:
  topic: {type: string, description: "Research topic"}
  
steps:
  - id: search
    action: <AUTO>search web for information about {{topic}}</AUTO>
    tools: [web-search]
    
  - id: analyze  
    action: <AUTO>analyze search results and extract key insights about {{topic}}</AUTO>
    depends_on: [search]
    inputs: {results: "{{search.results}}"}
    
  - id: report
    action: <AUTO>generate comprehensive report about {{topic}} based on analysis</AUTO>
    depends_on: [analyze]
    inputs: {insights: "{{analyze.insights}}", topic: "{{topic}}"}
    tools: [report-generator]
    
outputs:
  report: "{{report.content}}"
```

### Framework Should Handle Everything Else Automatically

## New Architecture Design

### 1. **Declarative Pipeline Engine**

```python
class DeclarativePipelineEngine:
    """Executes YAML pipelines with zero custom code required."""
    
    async def execute_pipeline(self, yaml_content: str, inputs: dict) -> dict:
        # 1. Parse YAML and resolve AUTO tags with AI models
        # 2. Automatically discover and configure required tools  
        # 3. Generate execution plan with dependency resolution
        # 4. Execute tasks with automatic prompt generation
        # 5. Return structured outputs
```

### 2. **AUTO Tag Resolution System**

```python
class AutoTagResolver:
    """Converts abstract task descriptions into executable prompts."""
    
    async def resolve_auto_tag(self, tag_content: str, context: dict) -> str:
        # Examples:
        # "<AUTO>search web for {{topic}}</AUTO>" 
        # → "Search the web for information about {topic}. Return results as JSON with title, url, snippet for each result."
        
        # "<AUTO>analyze data and extract insights</AUTO>"
        # → "Analyze the provided data: {data}. Extract key insights, patterns, and important findings. Return as structured analysis with insights, themes, and summary."
```

### 3. **Universal Tool Registry**

```python
class UniversalToolRegistry:
    """Automatically discovers and provides tools for any task."""
    
    def auto_discover_tools_for_action(self, action_description: str) -> List[Tool]:
        # Maps abstract actions to concrete tools
        # "search web" → WebSearchTool
        # "analyze data" → DataProcessingTool  
        # "generate report" → ReportGeneratorTool
        
    def execute_tool_for_action(self, action: str, inputs: dict, tools: List[Tool]) -> dict:
        # Automatically executes appropriate tool based on action description
```

### 4. **Intelligent Task Executor**

```python
class IntelligentTaskExecutor:
    """Executes individual pipeline tasks with AI assistance."""
    
    async def execute_task(self, task_spec: dict, context: dict) -> dict:
        # 1. Resolve AUTO tags in task specification
        # 2. Generate appropriate prompts for the task
        # 3. Select and configure required tools
        # 4. Execute with appropriate AI model
        # 5. Validate and structure outputs
```

## Implementation Plan

### Phase 1: Core Declarative Engine (Week 1-2)

#### 1.1 **New Pipeline Parser**
```python
# src/orchestrator/engine/declarative_parser.py
class DeclarativePipelineParser:
    """Parses YAML into executable pipeline specifications."""
    
    def parse_pipeline(self, yaml_content: str) -> PipelineSpec:
        # Parse YAML with validation
        # Identify all AUTO tags for resolution
        # Build dependency graph
        # Validate inputs/outputs
```

#### 1.2 **Enhanced AUTO Tag System**
```python
# src/orchestrator/engine/auto_resolver.py  
class EnhancedAutoResolver:
    """AI-powered resolution of abstract task descriptions."""
    
    async def resolve_action_to_prompt(self, auto_tag: str, context: dict) -> str:
        # Convert "<AUTO>search for {{topic}}</AUTO>" to executable prompt
        
    async def resolve_action_to_tools(self, auto_tag: str) -> List[str]:
        # Determine which tools are needed for abstract action
```

#### 1.3 **Universal Task Executor**
```python
# src/orchestrator/engine/task_executor.py
class UniversalTaskExecutor:
    """Executes any task defined declaratively."""
    
    async def execute(self, task_spec: TaskSpec, context: dict) -> dict:
        # 1. Resolve AUTO tags
        # 2. Generate prompts  
        # 3. Select tools
        # 4. Execute with AI model
        # 5. Return structured results
```

### Phase 2: Tool Integration Layer (Week 2-3)

#### 2.1 **Smart Tool Discovery**
```python
# src/orchestrator/tools/discovery.py
class ToolDiscoveryEngine:
    """Automatically maps actions to tools."""
    
    ACTION_PATTERNS = {
        r"search.*web|web.*search": ["web-search"],
        r"analyze.*data|data.*analysis": ["data-processing"],
        r"generate.*report|create.*report": ["report-generator"],
        r"extract.*content|scrape.*page": ["headless-browser"],
        # ... more patterns
    }
    
    def discover_tools_for_action(self, action_description: str) -> List[str]:
        # Use pattern matching + AI to determine required tools
```

#### 2.2 **Automatic Tool Execution**
```python
# src/orchestrator/tools/executor.py
class AutoToolExecutor:
    """Executes tools automatically based on task description."""
    
    async def execute_tools_for_task(self, 
                                   action: str, 
                                   inputs: dict, 
                                   available_tools: List[Tool]) -> dict:
        # Automatically configure and execute tools
```

### Phase 3: Advanced Pipeline Features (Week 3-4)

#### 3.1 **Conditional Execution**
```yaml
# Support for conditional logic in YAML
steps:
  - id: validate_input
    action: <AUTO>validate that {{input}} meets requirements</AUTO>
    
  - id: process_valid
    action: <AUTO>process validated input</AUTO>
    condition: "{{validate_input.valid}} == true"
    depends_on: [validate_input]
    
  - id: handle_invalid  
    action: <AUTO>handle invalid input with error message</AUTO>
    condition: "{{validate_input.valid}} == false"
    depends_on: [validate_input]
```

#### 3.2 **Loop Support**
```yaml
# Support for iterative processing
steps:
  - id: process_items
    action: <AUTO>process each item in {{items}}</AUTO>
    foreach: "{{inputs.items}}"
    tools: [data-processing]
```

#### 3.3 **Error Handling**
```yaml
steps:
  - id: risky_operation
    action: <AUTO>perform operation that might fail</AUTO>
    on_error:
      action: <AUTO>handle failure gracefully</AUTO>
      continue: true
```

## New User Experience

### Example 1: Simple Research Pipeline
```yaml
name: "Quick Research"
description: "Research any topic and generate summary"

inputs:
  topic: {type: string, description: "Topic to research"}

steps:
  - id: search
    action: <AUTO>search web for recent information about {{topic}}</AUTO>
    
  - id: summarize
    action: <AUTO>create concise summary of key findings about {{topic}}</AUTO>
    depends_on: [search]

outputs:
  summary: "{{summarize.content}}"
```

**Usage:**
```bash
orchestrator run research.yaml --topic "quantum computing 2024"
```

### Example 2: Data Analysis Pipeline  
```yaml
name: "Data Analysis"
description: "Analyze CSV data and generate insights"

inputs:
  data_file: {type: file, description: "CSV file to analyze"}
  focus: {type: string, description: "Analysis focus area"}

steps:
  - id: load_data
    action: <AUTO>load and validate CSV data from {{data_file}}</AUTO>
    
  - id: analyze
    action: <AUTO>analyze data focusing on {{focus}}, find patterns and insights</AUTO>
    depends_on: [load_data]
    
  - id: visualize
    action: <AUTO>create relevant charts and graphs for {{focus}} analysis</AUTO>
    depends_on: [analyze]
    
  - id: report
    action: <AUTO>generate comprehensive analysis report with insights and visualizations</AUTO>
    depends_on: [analyze, visualize]

outputs:
  insights: "{{analyze.insights}}"
  report: "{{report.content}}"
  charts: "{{visualize.charts}}"
```

### Example 3: Complex Multi-Stage Pipeline
```yaml
name: "Content Creation Pipeline"
description: "Research topic, create content, and optimize for publication"

inputs:
  topic: {type: string}
  target_audience: {type: string}
  content_type: {type: string, enum: [blog, article, report]}

steps:
  - id: research
    action: <AUTO>research {{topic}} thoroughly for {{target_audience}}</AUTO>
    
  - id: outline
    action: <AUTO>create detailed outline for {{content_type}} about {{topic}}</AUTO>
    depends_on: [research]
    
  - id: write
    action: <AUTO>write engaging {{content_type}} following outline, targeting {{target_audience}}</AUTO>
    depends_on: [outline]
    
  - id: edit
    action: <AUTO>edit and improve content for clarity, flow, and engagement</AUTO>
    depends_on: [write]
    
  - id: optimize
    action: <AUTO>optimize content for SEO and readability</AUTO>
    depends_on: [edit]
    
  - id: format
    action: <AUTO>format content appropriately for {{content_type}} publication</AUTO>
    depends_on: [optimize]

outputs:
  final_content: "{{format.content}}"
  seo_keywords: "{{optimize.keywords}}"
  readability_score: "{{optimize.score}}"
```

## Migration Strategy

### Phase 1: Backwards Compatibility
- Keep existing control system interface working
- Add new declarative engine alongside existing system
- Allow gradual migration of existing pipelines

### Phase 2: Framework Simplification  
- Deprecate custom control systems
- Migrate all examples to pure YAML
- Update documentation to focus on declarative approach

### Phase 3: Advanced Features
- Add conditional execution, loops, error handling
- Implement advanced AUTO tag patterns
- Add pipeline composition and reuse features

## Benefits of New Design

### For Users
✅ **Simple**: Write YAML, get results  
✅ **Powerful**: AI handles complex prompt generation  
✅ **Flexible**: Works for any workflow imaginable  
✅ **Fast**: No boilerplate code to write  

### For Framework
✅ **Maintainable**: Business logic in AI models, not code  
✅ **Extensible**: Easy to add new tools and capabilities  
✅ **Testable**: Clear input/output contracts  
✅ **Scalable**: Automatic optimization and parallelization  

## Success Metrics

1. **User Experience**: Can a non-programmer create a working pipeline in < 10 lines of YAML?
2. **Flexibility**: Can the framework handle 90% of AI workflow use cases declaratively?
3. **Performance**: Does AUTO tag resolution add < 2 seconds to pipeline execution?
4. **Adoption**: Do users prefer declarative pipelines over custom Python code?

## Implementation Timeline

- **Week 1**: Core declarative engine + AUTO resolver
- **Week 2**: Tool discovery and automatic execution  
- **Week 3**: Advanced pipeline features (conditions, loops)
- **Week 4**: Documentation, examples, and migration tools

This redesign transforms Orchestrator from a framework that requires programming into a true declarative AI pipeline platform where users describe what they want and the system figures out how to do it.