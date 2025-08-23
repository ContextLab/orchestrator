# Intelligent Fact-Checker Pipeline

**Pipeline**: `examples/fact_checker.yaml`  
**Category**: Quality & Testing  
**Complexity**: Advanced  
**Key Features**: AUTO tags, Parallel processing, Structured data, For-each loops

## Overview

The Intelligent Fact-Checker Pipeline demonstrates advanced AUTO tag usage for dynamic list generation and parallel processing. It automatically extracts claims and sources from documents, then verifies them in parallel using sophisticated fact-checking techniques.

## Key Features Demonstrated

### 1. AUTO Tag List Resolution
```yaml
for_each: "<AUTO>list of sources to verify</AUTO>"
```

### 2. Structured Data Extraction
```yaml
action: generate-structured
schema:
  type: object
  properties:
    sources:
      type: array
      items:
        type: object
        properties:
          name: {type: string}
          url: {type: string}
          type: {type: string, enum: ["journal", "report", "website", "other"]}
```

### 3. Parallel Processing with Runtime Expansion
```yaml
for_each: "<AUTO>claims that need fact-checking</AUTO>"
max_parallel: 3
add_completion_task: true
```

## Pipeline Architecture

### Input Parameters
- **document_source** (required): Path to document file or URL to analyze
- **strictness** (optional): Fact-checking strictness level (lenient/moderate/strict, default: moderate)
- **output_path** (optional): Custom output path for the fact-check report

### Processing Flow

1. **Document Loading** - Reads source document using filesystem tool
2. **Source Extraction** - Uses structured generation to extract all citations and references
3. **Claim Extraction** - Identifies all verifiable factual claims in the document
4. **Parallel Source Verification** - Verifies each source's authenticity and credibility
5. **Parallel Claim Verification** - Fact-checks each claim against available evidence
6. **Report Generation** - Creates comprehensive fact-checking report
7. **Report Saving** - Saves the final report to specified location

### Advanced AUTO Tag Usage

#### Dynamic List Generation
The pipeline uses AUTO tags that resolve to arrays, enabling runtime for-each expansion:

```yaml
# Sources are extracted as structured data, then AUTO tag references them
- id: extract_sources_list
  action: generate-structured
  # ... produces list of sources ...

- id: verify_sources  
  for_each: "<AUTO>list of sources to verify</AUTO>"  # Resolves to extracted sources
```

#### Context-Aware Processing
AUTO tags can access complex nested data:
```yaml
for_each: "<AUTO>claims that need fact-checking</AUTO>"
# References the extracted claims from previous structured generation
```

## Usage Examples

### Basic Fact-Checking
```bash
python scripts/run_pipeline.py examples/fact_checker.yaml \
  -i document_source="examples/data/test_article.md"
```

### With Custom Settings
```bash
python scripts/run_pipeline.py examples/fact_checker.yaml \
  -i document_source="examples/data/research_paper.md" \
  -i strictness="strict" \
  -i output_path="my_fact_check_report.md"
```

### URL-Based Analysis
```bash
python scripts/run_pipeline.py examples/fact_checker.yaml \
  -i document_source="https://example.com/article.html" \
  -i strictness="lenient"
```

## Sample Output Structure

### Fact-Check Report Sections

1. **Executive Summary**
   - Brief overview of article topic and assertions
   - Overall credibility assessment (High/Medium/Low)
   - Verified vs unverified claims count

2. **Source Analysis**
   - Credibility assessment of each cited source
   - Missing or questionable citations
   - Balance between peer-reviewed and industry sources

3. **Claim Verification**
   - Each claim with verification status
   - Supporting or contradicting evidence
   - Confidence levels

4. **Red Flags and Concerns**
   - Misleading statements identified
   - Unsupported assertions
   - Potential biases detected

5. **Conclusion**
   - Overall accuracy assessment
   - Reader recommendations
   - Areas requiring further investigation

### Example Output File
Check the actual generated report: [fact_check_report.md](../../examples/outputs/fact_checker/fact_check_report.md)

## Parallel Processing Details

### Source Verification (Max 2 Parallel)
```yaml
- id: verify_sources
  for_each: "<AUTO>list of sources to verify</AUTO>"
  max_parallel: 2  # Conservative for API rate limits
  steps:
    - id: verify_source
      # Checks authenticity, accessibility, and credibility
```

### Claim Verification (Max 3 Parallel)  
```yaml
- id: verify_claims
  for_each: "<AUTO>claims that need fact-checking</AUTO>"
  max_parallel: 3  # Higher throughput for claim processing
  steps:
    - id: verify_claim
      # Analyzes claim support, reliability, and evidence
```

## Technical Implementation

### Structured Data Schema
The pipeline uses JSON schemas to ensure consistent data extraction:

```yaml
schema:
  type: object
  properties:
    sources:
      type: array
      items:
        type: object
        properties:
          name: {type: string}
          url: {type: string}  
          type: {type: string, enum: ["journal", "report", "website", "other"]}
    required: [sources]
```

### Template Variable Access
The for-each loops can access individual items and indices:
```yaml
parameters:
  prompt: |
    Verify this specific claim: {{ item }}
    Claim number: {{ index }}
```

### Dependency Management
Complex dependencies ensure proper execution order:
```yaml
dependencies:
  - verify_sources   # The entire ForEachTask
  - verify_claims    # The entire ForEachTask  
  - extract_sources_list
  - extract_claims_list
```

## Advanced Features

### Dynamic Strictness Adaptation
The pipeline can adapt its verification standards:
```yaml
inputs:
  strictness:
    type: string
    description: How strict should fact-checking be (lenient/moderate/strict)
    default: "moderate"
```

### Completion Tasks
For-each loops automatically include completion tasks:
```yaml
add_completion_task: true  # Aggregates results from all parallel executions
```

### Professional Output Format
Generates publication-ready fact-checking reports suitable for:
- Academic peer review
- Journalistic fact-checking
- Content verification workflows
- Regulatory compliance documentation

## Best Practices Demonstrated

1. **Structured Extraction**: Using schemas for consistent data parsing
2. **Parallel Processing**: Optimizing throughput while respecting API limits
3. **Professional Standards**: Following industry fact-checking methodologies
4. **Comprehensive Analysis**: Covering both sources and claims systematically
5. **Actionable Output**: Providing specific, useful recommendations

## Common Use Cases

- **News Article Verification**: Fact-check journalistic content
- **Academic Paper Review**: Verify research claims and citations
- **Content Moderation**: Identify false or misleading information
- **Due Diligence**: Verify claims in business or legal documents
- **Educational Content**: Check accuracy of instructional materials

## Troubleshooting

### Common Issues
- **Large Documents**: May hit token limits; consider document chunking
- **Network Sources**: URL access may fail; ensure accessibility
- **Rate Limiting**: Adjust max_parallel settings for API constraints
- **Complex Claims**: Some claims may be too subjective for verification

### Performance Optimization
- Reduce `max_parallel` if hitting API rate limits
- Increase token limits for complex documents
- Use caching for repeated source verification
- Consider document preprocessing for very long texts

## Related Examples
- [iterative_fact_checker.md](iterative_fact_checker.md) - Multi-pass fact-checking
- [iterative_fact_checker_simple.md](iterative_fact_checker_simple.md) - Simplified version
- [auto_tags_demo.md](auto_tags_demo.md) - Basic AUTO tag usage
- [control_flow_for_loop.md](control_flow_for_loop.md) - Parallel processing patterns

## Technical Requirements

- **Model**: Claude Sonnet 4 (for high-quality structured generation)
- **Tools**: filesystem (document reading), structured generation capability
- **API Access**: Sufficient rate limits for parallel processing
- **Memory**: Adequate for processing extracted claims and sources lists

This pipeline showcases enterprise-grade fact-checking capabilities with professional output quality and efficient parallel processing architecture.