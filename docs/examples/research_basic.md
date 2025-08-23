# Basic Research Pipeline

**Pipeline**: `examples/research_basic.yaml`  
**Category**: Web & Research  
**Complexity**: Advanced  
**Key Features**: Multi-stage research, Web search integration, AUTO model selection, Structured analysis, Report generation

## Overview

The Basic Research Pipeline provides comprehensive research capabilities using standard LLM actions and web search tools. It performs multi-stage information gathering, analysis, and synthesis to generate professional research reports with configurable depth levels, making it ideal for automated research and content generation workflows.

## Key Features Demonstrated

### 1. Multi-Stage Web Search
```yaml
- id: initial_search
  tool: web-search
  parameters:
    query: "{{topic}} overview introduction basics"
    
- id: deep_search
  tool: web-search
  parameters:
    query: "{{topic}} advanced research latest developments 2024 2025"
```

### 2. Task-Specific AUTO Model Selection
```yaml
model: <AUTO task="analyze">Select model for information extraction</AUTO>
model: <AUTO task="generate">Select model for summary generation</AUTO>
```

### 3. Structured Content Analysis
```yaml
- id: extract_key_points
  action: analyze_text
  parameters:
    analysis_type: "key_points"
    prompt: |
      Extract and synthesize the key points about {{topic}}:
      1. Identify the most important facts, concepts, and developments
      2. Organize information thematically
      3. Note any conflicting information or debates
```

### 4. Configurable Research Depth
```yaml
parameters:
  depth:
    type: string
    default: comprehensive
    description: Research depth (basic, comprehensive, or expert)
```

## Pipeline Architecture

### Input Parameters
- **topic** (required): Research topic to investigate
- **depth** (optional): Research depth level - "basic", "comprehensive", or "expert" (default: "comprehensive")
- **output_path** (optional): Directory for output files (default: "examples/outputs/research_basic")

### Processing Flow

1. **Initial Search** - Broad overview and introduction search
2. **Deep Search** - Advanced and recent developments search
3. **Extract Key Points** - Analyze and synthesize search results
4. **Generate Summary** - Create executive summary
5. **Generate Analysis** - Produce detailed analysis with sections
6. **Generate Conclusion** - Synthesize findings into conclusion
7. **Compile Report** - Assemble complete research report
8. **Save Report** - Store final report to file

### Research Depth Levels

#### Basic Research
```yaml
depth: "basic"
# Focus on fundamental concepts and major trends
# Suitable for general audience
# Covers essential information without deep technical details
```

#### Comprehensive Research
```yaml
depth: "comprehensive"  # Default
# Include detailed analysis and multiple perspectives
# Balanced depth for professional use
# Covers technical details with context
```

#### Expert Research
```yaml
depth: "expert"
# Deep dive with technical details and nuanced insights
# Suitable for specialists and experts
# Includes advanced concepts and methodologies
```

## Usage Examples

### Basic Research Report
```bash
python scripts/run_pipeline.py examples/research_basic.yaml \
  -i topic="renewable energy technologies"
```

### Comprehensive Analysis
```bash
python scripts/run_pipeline.py examples/research_basic.yaml \
  -i topic="artificial intelligence in healthcare" \
  -i depth="comprehensive"
```

### Expert-Level Research
```bash
python scripts/run_pipeline.py examples/research_basic.yaml \
  -i topic="quantum computing algorithms" \
  -i depth="expert"
```

### Multiple Topic Research
```bash
# Research multiple related topics
for topic in "machine learning" "deep learning" "neural networks"; do
  python scripts/run_pipeline.py examples/research_basic.yaml \
    -i topic="$topic" \
    -i depth="basic"
done
```

## Search Strategy Details

### Initial Search Phase
```yaml
query: "{{topic}} overview introduction basics"
max_results: 10
# Purpose: Establish foundational understanding
# Target: General overview and basic concepts
# Sources: Educational content, introductions, overviews
```

### Deep Search Phase
```yaml
query: "{{topic}} advanced research latest developments 2024 2025"
max_results: 10
# Purpose: Capture recent developments and advanced concepts
# Target: Cutting-edge research and current trends
# Sources: Recent publications, research papers, news
```

### Search Result Processing
```yaml
# Template for result integration
{% for result in initial_search.results %}
- {{result.title}}: {{result.snippet}}
{% endfor %}

{% for result in deep_search.results %}
- {{result.title}}: {{result.snippet}}
{% endfor %}
```

## Analysis Framework

### Key Point Extraction
```yaml
analysis_tasks:
  1. "Identify the most important facts, concepts, and developments"
  2. "Organize information thematically" 
  3. "Note any conflicting information or debates"
  4. "Highlight recent developments or breakthroughs"
  5. "Include specific examples, statistics, or case studies"
```

### Analysis Structure
```yaml
sections:
  1. "Current State and Trends"
  2. "Key Challenges and Opportunities"
  3. "Future Directions and Implications"
```

### Content Requirements
```yaml
style_requirements:
  - "Direct, academic style suitable for research report"
  - "No conversational language or phrases"
  - "Specific examples and evidence from findings"
  - "Balanced perspective on controversies"
  - "Concrete predictions or recommendations"
```

## Sample Report Structure

### Executive Summary (250-400 words)
```markdown
## Executive Summary

Artificial intelligence in healthcare represents a transformative shift in medical practice, 
driven by advances in machine learning algorithms and the availability of large-scale 
health datasets. Current implementations span diagnostic imaging, drug discovery, 
clinical decision support, and personalized treatment planning.

Key developments include the FDA approval of over 100 AI-based medical devices since 2020, 
with diagnostic radiology leading adoption at 75% of major hospitals. Machine learning 
models now match or exceed human performance in detecting diabetic retinopathy, skin 
cancer, and certain cardiac abnormalities.

Significant challenges persist around data privacy, algorithmic bias, and integration 
with existing electronic health records. Regulatory frameworks are evolving rapidly, 
with new guidelines for AI validation and post-market surveillance emerging globally.

The market is projected to reach $102 billion by 2028, with venture capital investment 
exceeding $4.5 billion in 2024, indicating sustained confidence in healthcare AI 
applications and commercial viability.
```

### Detailed Analysis (800-1000 words)
```markdown
## Current State and Trends

Healthcare AI adoption has accelerated significantly, with implementation rates 
increasing 300% since 2021. Major health systems report deploying an average 
of 12 AI tools across clinical workflows...

## Key Challenges and Opportunities

Data interoperability remains the primary barrier, with 68% of hospitals citing 
integration difficulties as the main adoption obstacle. However, emerging standards 
like FHIR and HL7 are creating new opportunities for seamless AI deployment...

## Future Directions and Implications

The convergence of AI with emerging technologies like 5G, edge computing, and 
wearable devices is expected to enable real-time health monitoring and 
predictive intervention capabilities...
```

### Conclusion (150-250 words)
```markdown
## Conclusion

The integration of AI in healthcare has reached a critical inflection point where 
theoretical potential is becoming practical reality. The combination of regulatory 
approval acceleration, proven clinical outcomes, and substantial investment 
indicates sustainable growth trajectory.

Most significantly, AI is shifting healthcare from reactive treatment to proactive 
prevention, with predictive models identifying at-risk patients before symptom 
onset. This paradigm change promises to reduce costs while improving outcomes 
across population health.

Future success depends on addressing current limitations in data standardization, 
algorithmic transparency, and workforce adaptation. Healthcare organizations that 
develop comprehensive AI strategies now will gain substantial competitive advantages 
in the evolving medical landscape.
```

## AUTO Model Selection Strategy

### Analysis Tasks
```yaml
<AUTO task="analyze">Select model for information extraction</AUTO>
# Optimizes for: Information synthesis, pattern recognition, structured analysis
# Likely selection: Models with strong analytical and reasoning capabilities
```

### Generation Tasks  
```yaml
<AUTO task="generate">Select model for summary generation</AUTO>
# Optimizes for: Content creation, synthesis, professional writing
# Likely selection: Models with strong language generation capabilities
```

### Task-Specific Optimization
```yaml
analysis_type: "key_points"     # Hints for analysis optimization
max_tokens: 500                 # Summary length optimization
max_tokens: 1000               # Analysis depth optimization
```

## Professional Writing Standards

### Content Requirements
- **No Conversational Language**: Avoid "Certainly!", "I'd be happy to"
- **Direct Academic Style**: Professional, informative tone
- **Specific Evidence**: Include statistics, examples, case studies
- **Balanced Perspective**: Present multiple viewpoints on debates
- **Future-Focused**: Forward-looking insights and predictions

### Structural Standards
- **Clear Sections**: Organized thematic structure
- **Logical Flow**: Information builds progressively
- **Specific Details**: Concrete examples and data points
- **Substantive Conclusions**: Meaningful synthesis and implications

## Advanced Features

### Template Integration
```yaml
# Dynamic content insertion
Topic: {{topic}}
Depth: {{depth}}
Key Points: {{extract_key_points.result}}
Summary: {{generate_summary.result}}
```

### Conditional Processing
```yaml
# Depth-aware analysis
"{{depth}}" means:
- basic: Focus on fundamental concepts
- comprehensive: Detailed analysis and perspectives  
- expert: Deep dive with technical details
```

### Multi-Stage Synthesis
```yaml
# Progressive content building
initial_search → deep_search → extract_key_points → generate_summary → generate_analysis → generate_conclusion
```

## Best Practices Demonstrated

1. **Multi-Source Research**: Combine broad and deep search strategies
2. **Structured Analysis**: Systematic information extraction and synthesis
3. **Professional Standards**: Academic writing quality and formatting
4. **Configurable Depth**: Adaptable to different audience needs
5. **Progressive Synthesis**: Build complexity through sequential analysis
6. **Quality Control**: Explicit requirements for content and style

## Common Use Cases

- **Business Intelligence**: Market research and competitive analysis
- **Academic Research**: Literature reviews and topic exploration
- **Content Creation**: Research-backed article and report writing
- **Due Diligence**: Investment and partnership research
- **Policy Analysis**: Government and regulatory research
- **Technology Assessment**: Emerging technology evaluation

## Related Examples
- [research_advanced_tools.md](research_advanced_tools.md) - Advanced research with specialized tools
- [enhanced_research_pipeline.md](enhanced_research_pipeline.md) - Enhanced research capabilities
- [working_web_search.md](working_web_search.md) - Basic web search integration

## Technical Requirements

- **Web Search**: Internet connectivity and search API access
- **Multiple Models**: Access to different AI models for optimal task performance
- **Template Engine**: Advanced template processing for content synthesis
- **File System**: Write access for report generation
- **Text Analysis**: Sophisticated text processing and analysis capabilities

This pipeline provides enterprise-grade research capabilities suitable for professional research, content creation, and business intelligence applications requiring comprehensive analysis and high-quality written outputs.