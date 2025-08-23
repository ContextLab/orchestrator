# Research Pipeline with Advanced Tools

**Pipeline**: `examples/research_advanced_tools.yaml`  
**Category**: Web & Research  
**Complexity**: Expert  
**Key Features**: Advanced web search, Content extraction, Headless browser integration, PDF generation, Multi-source synthesis

## Overview

The Research Pipeline with Advanced Tools provides sophisticated research capabilities using specialized tools for web searching, content extraction, and document generation. It combines multiple search strategies, extracts full content from sources using headless browsing, and produces professional research reports with optional PDF compilation.

## Key Features Demonstrated

### 1. Multi-Backend Web Search
```yaml
- id: search_topic
  tool: web-search
  parameters:
    query: "{{ topic }} latest developments"
    backend: "duckduckgo"
    
- id: deep_search
  tool: web-search
  parameters:
    query: "{{ topic }} research papers technical details implementation"
```

### 2. Headless Browser Content Extraction
```yaml
- id: extract_content
  tool: headless-browser
  action: scrape
  parameters:
    url: "{{ search_topic.results[0].url if search_topic.results and search_topic.results|length > 0 else deep_search.results[0].url }}"
```

### 3. Conditional PDF Generation
```yaml
- id: compile_pdf
  tool: pdf-compiler
  action: compile
  parameters:
    markdown_content: "{{ create_report.content }}"
  condition: "{{ compile_to_pdf | bool }}"
```

### 4. Advanced Error Handling
```yaml
continue_on_error: true
condition: "{{ (search_topic.results | length > 0) or (deep_search.results | length > 0) }}"
```

## Pipeline Architecture

### Input Parameters
- **topic** (optional): Research topic to investigate (default: "quantum computing applications")
- **max_results** (optional): Maximum search results per query (default: 10)
- **compile_to_pdf** (optional): Whether to generate PDF output (default: true)
- **output_path** (optional): Directory for output files (default: "examples/outputs/research_advanced_tools")

### Processing Flow

1. **Search Topic** - Initial web search for latest developments
2. **Deep Search** - Technical and research-focused search
3. **Extract Content** - Full content extraction from top result using headless browser
4. **Analyze Findings** - Comprehensive analysis and synthesis
5. **Create Report** - Generate structured markdown report
6. **Compile PDF** - Convert to PDF format (conditional)
7. **Save Report** - Store final report with timestamped filename

### Advanced Tool Integration

#### Web Search with Backend Selection
```yaml
tool: web-search
backend: "duckduckgo"
# Supports multiple search backends
# Configurable result limits and query strategies
```

#### Headless Browser Scraping
```yaml
tool: headless-browser
action: scrape
# Extracts full page content including dynamic elements
# Provides title, text content, and metadata
# Handles JavaScript-rendered pages
```

#### PDF Compilation
```yaml
tool: pdf-compiler
action: compile
# Converts markdown to professional PDF
# Maintains formatting and structure
# Suitable for professional distribution
```

## Usage Examples

### Comprehensive Research Report
```bash
python scripts/run_pipeline.py examples/research_advanced_tools.yaml \
  -i topic="artificial intelligence in healthcare" \
  -i max_results=15
```

### Technical Research Without PDF
```bash
python scripts/run_pipeline.py examples/research_advanced_tools.yaml \
  -i topic="blockchain consensus mechanisms" \
  -i compile_to_pdf=false \
  -i max_results=8
```

### High-Volume Research
```bash
python scripts/run_pipeline.py examples/research_advanced_tools.yaml \
  -i topic="renewable energy storage systems" \
  -i max_results=20 \
  -i compile_to_pdf=true
```

### Batch Research Topics
```bash
# Research multiple related topics
topics=("machine learning algorithms" "deep learning architectures" "neural network optimization")
for topic in "${topics[@]}"; do
  python scripts/run_pipeline.py examples/research_advanced_tools.yaml \
    -i topic="$topic" \
    -i max_results=12
done
```

## Advanced Search Strategies

### Primary Search Strategy
```yaml
query: "{{ topic }} latest developments"
# Focus: Recent developments and current state
# Target: News, updates, current research
# Scope: Broad overview of topic
```

### Deep Search Strategy
```yaml
query: "{{ topic }} research papers technical details implementation"
# Focus: Technical depth and implementation
# Target: Academic papers, technical documentation
# Scope: Detailed technical information
```

### Search Result Processing
```yaml
# Comprehensive result integration
Primary search results ({{ search_topic.total_results }} total):
{% for result in search_topic.results[:max_results] %}
{{ loop.index }}. {{ result.title }}
   URL: {{ result.url }}
   Summary: {{ result.snippet }}
{% endfor %}

Technical search results ({{ deep_search.total_results }} total):
{% for result in deep_search.results %}
- {{ result.title }}: {{ result.snippet }}
{% endfor %}
```

## Content Extraction Features

### Headless Browser Capabilities
```yaml
extract_content:
  title: "Page title"
  text: "Full text content"
  word_count: 1500
  url: "Source URL"
  images: ["image1.jpg", "image2.png"]
  links: ["link1.com", "link2.com"]
```

### Dynamic Content Handling
- **JavaScript Execution**: Renders dynamic content
- **Full Page Processing**: Captures complete page content
- **Metadata Extraction**: Title, descriptions, structured data
- **Error Recovery**: Graceful handling of inaccessible pages

### Content Integration
```yaml
{% if extract_content is defined and 'word_count' in extract_content and extract_content.word_count > 0 %}
Extracted content from primary source:
Title: {{ extract_content.title }}
Key content: {{ extract_content.text | truncate(1500) }}...
{% endif %}
```

## Comprehensive Analysis Framework

### Analysis Structure
```yaml
analysis_components:
  1. "2-3 paragraph overview synthesizing key insights"
  2. "Key Findings - 5-8 specific, substantive points"
  3. "Technical Analysis - Deep dive into implementation"
  4. "Current Trends and Future Directions"
  5. "Critical Evaluation - Strengths, limitations, gaps"
```

### Analysis Requirements
```yaml
content_standards:
  - "Specific examples and data from sources"
  - "Professional, analytical style"
  - "Advanced research report quality"
  - "Avoid generic statements"
  - "Direct content start (no headers/metadata)"
```

### AUTO Model Selection
```yaml
model: <AUTO task="analyze">Select model for comprehensive analysis</AUTO>
analysis_type: "comprehensive"
max_tokens: 2000
# Optimized for deep analytical thinking and synthesis
```

## Sample Research Output

### Overview Section
```markdown
Quantum computing applications have reached a critical maturity threshold in 2024, with 
commercial implementations emerging across cryptography, optimization, and simulation domains. 
IBM's 1000-qubit processor and Google's logical qubit breakthrough represent significant 
hardware advances, while software frameworks like Qiskit and Cirq have democratized quantum 
algorithm development.

The field has shifted from theoretical exploration to practical implementation, with 
financial services, pharmaceutical companies, and logistics providers deploying quantum 
solutions for specific use cases. However, current systems remain limited by quantum 
decoherence and error rates, requiring sophisticated error correction protocols.

Current applications focus on problems where quantum advantage is most pronounced: 
cryptographic key generation, portfolio optimization, molecular simulation, and 
combinatorial optimization challenges that exceed classical computing capabilities.
```

### Key Findings Section
```markdown
## Key Findings

1. **Commercial Viability**: Quantum computing has achieved commercial deployment in 
   specialized applications, with companies like D-Wave reporting over 100 customer 
   implementations in optimization scenarios.

2. **Error Correction Progress**: IBM's demonstration of quantum error correction with 
   surface codes has reduced logical error rates by 65%, approaching thresholds needed 
   for practical applications.

3. **Industry Adoption**: Financial sector leads adoption with 40% of major banks 
   exploring quantum applications for risk analysis and portfolio optimization.

4. **Hardware Scaling**: Multiple vendors have achieved 100+ qubit systems, with 
   roadmaps targeting 10,000+ qubits by 2028 across different qubit technologies.

5. **Software Ecosystem**: Quantum development environments have matured significantly, 
   with cloud-based quantum computing services reporting 300% growth in developer usage.
```

## PDF Generation Features

### Document Formatting
```yaml
pdf_features:
  - "Professional typography and layout"
  - "Automatic table of contents generation" 
  - "Consistent heading hierarchy"
  - "Proper citation formatting"
  - "Page numbering and headers"
```

### Conditional Generation
```yaml
condition: "{{ compile_to_pdf | bool }}"
# Only generates PDF when explicitly requested
# Reduces processing time for markdown-only workflows
# Configurable via input parameters
```

## Error Handling and Resilience

### Content Extraction Fallbacks
```yaml
continue_on_error: true
# Continues pipeline execution even if content extraction fails
# Ensures research report generation with available data
# Graceful degradation of functionality
```

### Conditional Processing
```yaml
condition: "{{ (search_topic.results | length > 0) or (deep_search.results | length > 0) }}"
# Only attempts content extraction if search results exist
# Prevents errors from empty result sets
# Smart resource utilization
```

### Complex URL Selection
```yaml
url: "{{ search_topic.results[0].url if search_topic.results and search_topic.results|length > 0 else deep_search.results[0].url if deep_search.results and deep_search.results|length > 0 else '' }}"
# Intelligent fallback URL selection
# Primary search results preferred
# Secondary search as backup
```

## Best Practices Demonstrated

1. **Multi-Source Research**: Combine different search strategies and tools
2. **Content Depth**: Extract full content beyond search snippets
3. **Error Resilience**: Continue processing despite individual tool failures
4. **Professional Output**: Generate both markdown and PDF formats
5. **Configurable Processing**: Flexible parameters for different use cases
6. **Comprehensive Analysis**: Structured analytical framework
7. **Advanced Tool Integration**: Leverage specialized research tools

## Common Use Cases

- **Academic Research**: Literature reviews and technical analysis
- **Business Intelligence**: Market research and competitive analysis
- **Technology Assessment**: Emerging technology evaluation
- **Investment Research**: Due diligence and market analysis
- **Policy Research**: Regulatory and policy impact analysis
- **Scientific Research**: Research paper synthesis and analysis

## Troubleshooting

### Search Result Issues
- Verify web search tool configuration and API access
- Check query formatting and search backend availability
- Ensure network connectivity for external searches

### Content Extraction Problems
- Validate headless browser tool availability
- Check target URL accessibility and structure
- Verify JavaScript rendering requirements

### PDF Generation Failures
- Ensure PDF compilation tool is properly configured
- Check markdown content format compatibility
- Verify file system write permissions

## Related Examples
- [research_basic.md](research_basic.md) - Basic research without advanced tools
- [working_web_search.md](working_web_search.md) - Simple web search integration
- [mcp_simple_test.md](mcp_simple_test.md) - Alternative search tool integration

## Technical Requirements

- **Web Search**: Configured search backend (DuckDuckGo, etc.)
- **Headless Browser**: Browser automation capabilities (Chrome, Firefox)
- **PDF Compiler**: Markdown to PDF conversion (Pandoc, wkhtmltopdf)
- **Network Access**: Reliable internet connectivity
- **File System**: Read/write access for report generation
- **AI Models**: Advanced language models for comprehensive analysis

This pipeline represents the state-of-the-art in automated research capabilities, combining multiple specialized tools to produce professional-quality research reports suitable for academic, business, and technical applications.