# Simple Iterative Fact-Checking Pipeline

**Pipeline**: `examples/iterative_fact_checker_simple.yaml`  
**Category**: Quality & Testing  
**Complexity**: Intermediate  
**Key Features**: Structured generation, Citation addition, Document analysis, Reference validation

## Overview

The Simple Iterative Fact-Checking Pipeline provides a streamlined approach to document fact-checking without complex while loops. It analyzes documents for factual claims, identifies those lacking references, finds appropriate citations, and produces an improved document with proper sourcing.

## Key Features Demonstrated

### 1. Structured Claim Extraction
```yaml
action: generate-structured
schema:
  properties:
    claims:
      type: array
      items:
        properties:
          text: {type: string}
          has_reference: {type: boolean}
          reference_url: {type: string}
```

### 2. Citation Research and Addition
```yaml
prompt: |
  Find reliable sources and citations for these unreferenced claims:
  {% for claim in extract_claims.claims %}
  {% if not claim.has_reference %}
  - {{ claim.text }}
  {% endif %}
  {% endfor %}
```

### 3. Document Enhancement
```yaml
prompt: |
  Update this document by adding citations for all unreferenced claims.
  Instructions:
  1. Add inline citations [1], [2], etc. after each claim
  2. Add a "References" section at the end
  3. Keep the document structure unchanged
```

### 4. Comprehensive Reporting
```yaml
content: |
  ## Analysis Summary
  - **Total Claims**: {{ extract_claims.total_claims }}
  - **Claims with References (before)**: {{ extract_claims.claims_with_references }}
  - **Percentage Referenced (before)**: {{ extract_claims.percentage_referenced }}%
```

## Pipeline Architecture

### Input Parameters
- **input_document** (optional): Path to document for fact-checking (default: "test_climate_document.md")
- **quality_threshold** (optional): Minimum reference percentage (default: 0.95)

### Processing Flow

1. **Load Document** - Read source document from filesystem
2. **Extract Claims** - Use structured generation to identify all factual claims
3. **Find Citations** - Research appropriate sources for unreferenced claims
4. **Update Document** - Add inline citations and reference section
5. **Save Document** - Store enhanced version with citations
6. **Generate Report** - Create comprehensive fact-checking summary

### Single-Pass Processing

Unlike the complex iterative version, this pipeline performs fact-checking in a single pass:
- **Analyze once**: Extract all claims in one operation
- **Research once**: Find citations for all unreferenced claims
- **Update once**: Add all necessary references
- **Report once**: Provide complete analysis

## Usage Examples

### Basic Document Fact-Checking
```bash
python scripts/run_pipeline.py examples/iterative_fact_checker_simple.yaml \
  -i input_document="examples/data/test_article.md"
```

### Custom Quality Assessment
```bash
python scripts/run_pipeline.py examples/iterative_fact_checker_simple.yaml \
  -i input_document="research_paper.md" \
  -i quality_threshold=0.90
```

### Multiple Document Processing
```bash
# Process different document types
python scripts/run_pipeline.py examples/iterative_fact_checker_simple.yaml \
  -i input_document="blog_post.md"

python scripts/run_pipeline.py examples/iterative_fact_checker_simple.yaml \
  -i input_document="white_paper.md"
```

## Claim Analysis Process

### Structured Extraction Schema
```yaml
schema:
  type: object
  properties:
    claims:
      type: array
      items:
        type: object
        properties:
          text: {type: string}
          has_reference: {type: boolean}
          reference_url: {type: string}
    total_claims: {type: integer}
    claims_with_references: {type: integer}
    percentage_referenced: {type: number}
```

### Example Claim Extraction
```json
{
  "claims": [
    {
      "text": "Global temperatures have risen 1.1°C since 1880",
      "has_reference": false,
      "reference_url": ""
    },
    {
      "text": "Arctic sea ice is declining at 13% per decade",
      "has_reference": true,
      "reference_url": "https://climate.nasa.gov/evidence/"
    }
  ],
  "total_claims": 15,
  "claims_with_references": 8,
  "percentage_referenced": 53.3
}
```

## Citation Research Process

### Authoritative Source Guidelines
The pipeline prioritizes reliable sources:
- **Scientific**: NASA, NOAA, IPCC reports
- **Academic**: Nature, Science, peer-reviewed journals
- **Government**: Official agency publications
- **International**: WHO, UN, World Bank data

### Citation Format Requirements
```yaml
Format as a structured list with clear citations:
1. A reliable source URL
2. The source title
3. Brief explanation of claim support
```

### Example Citation Research Output
```
1. Global temperatures have risen 1.1°C since 1880
   - Source: NASA Goddard Institute for Space Studies
   - URL: https://climate.nasa.gov/evidence/
   - Support: NASA's temperature record shows clear warming trend

2. Arctic sea ice declining at 13% per decade
   - Source: National Snow and Ice Data Center
   - URL: https://nsidc.org/arctic-sea-ice-news
   - Support: Satellite measurements confirm decline rate
```

## Document Enhancement Process

### Citation Integration
```yaml
Instructions:
1. Add inline citations [1], [2], etc. after each claim
2. Add a "References" section at the end with all citations
3. Keep the document structure and content otherwise unchanged
4. Use consistent formatting throughout
```

### Before Enhancement
```markdown
# Climate Change Facts

Global temperatures have risen significantly since 1880.
Arctic sea ice continues to decline at an alarming rate.
```

### After Enhancement
```markdown
# Climate Change Facts

Global temperatures have risen significantly since 1880 [1].
Arctic sea ice continues to decline at an alarming rate [2].

## References

[1] NASA Goddard Institute for Space Studies. "Global Temperature Anomalies." 
    https://climate.nasa.gov/evidence/

[2] National Snow and Ice Data Center. "Arctic Sea Ice News & Analysis."
    https://nsidc.org/arctic-sea-ice-news
```

## Output Files Generated

### Enhanced Document
- **Location**: `{output_path}/{document_name}_fact_checked.md`
- **Content**: Original document with added citations and references
- **Format**: Maintains original structure with inline citations

### Fact-Checking Report
- **Location**: `{output_path}/fact_checking_report.md`
- **Content**: Comprehensive analysis and processing summary

### Sample Report Structure
```markdown
# Fact-Checking Report

## Document Information
- Source Document: test_climate_document.md
- Date Processed: 2024-08-23T10:30:00Z

## Analysis Summary
- Total Claims: 12
- Claims with References (before): 5
- Percentage Referenced (before): 41.7%
- Quality Threshold: 95%

## Claims Identified
1. Global temperatures have risen 1.1°C since 1880
   - Has reference: false
2. Arctic sea ice is declining at 13% per decade
   - Has reference: true

## Status
⚠️ Document below quality threshold - references added

## Output
- Updated document: test_climate_document_fact_checked.md
```

## Quality Assessment Metrics

### Reference Coverage
- **Total Claims**: All factual assertions identified
- **Referenced Claims**: Claims with existing citations
- **Coverage Percentage**: (Referenced / Total) × 100
- **Quality Status**: Meets/Below threshold comparison

### Processing Effectiveness
- **Claims Added**: New factual assertions discovered
- **References Added**: Citations provided for unreferenced claims
- **Source Quality**: Reliability of added references
- **Format Consistency**: Citation formatting standards

## Technical Implementation

### Template Logic
```yaml
{% if extract_claims.claims %}
{% for claim in extract_claims.claims %}
{% if not claim.has_reference %}
- {{ claim.text }}
{% endif %}
{% endfor %}
{% else %}
No claims data available.
{% endif %}
```

### File Path Construction
```yaml
path: "{{ output_path }}/{{ parameters.input_document | basename | regex_replace('\\.md$', '') }}_fact_checked.md"
```

### Dependency Management
```yaml
dependencies:
  - load_document
  - extract_claims
  - find_citations
```

## Advantages Over Complex Version

### 1. Simplicity
- No while loops or complex iteration logic
- Straightforward single-pass processing
- Easier to understand and maintain

### 2. Reliability
- Avoids potential infinite loop issues
- Predictable execution time
- Simpler error handling

### 3. Flexibility
- Can be run multiple times manually for iterative improvement
- Easy to integrate into larger workflows
- Clear input/output relationship

### 4. Debugging
- Each step produces clear intermediate outputs
- Easy to trace processing steps
- Simplified troubleshooting

## Common Use Cases

- **Content Review**: Quick fact-checking for blog posts and articles
- **Academic Writing**: Adding citations to research papers
- **Journalism**: Source verification for news articles
- **Educational Content**: Ensuring teaching materials are properly sourced
- **Business Documents**: Adding credibility to reports and presentations
- **Documentation**: Improving technical documentation with references

## Troubleshooting

### Low Quality Scores
- Check if document contains many subjective statements
- Verify claim extraction is identifying factual assertions correctly
- Ensure citation research is finding appropriate sources

### Citation Issues
- Confirm URLs are accessible and authoritative
- Check citation format matches document style
- Verify inline citations match reference list

### Processing Errors
- Ensure input document is readable and well-formatted
- Check for very long documents that may exceed token limits
- Verify file paths are correct and accessible

## Related Examples
- [iterative_fact_checker.md](iterative_fact_checker.md) - Complex iterative version
- [fact_checker.md](fact_checker.md) - Parallel processing fact-checker
- [validation_pipeline.md](validation_pipeline.md) - Data validation patterns

## Technical Requirements

- **Model**: Claude Sonnet 4 for high-quality structured generation
- **File System**: Read/write access for document processing
- **Template Engine**: Jinja2 for conditional logic
- **Structured Generation**: JSON schema validation
- **Token Limits**: Sufficient for document processing (up to 6000 tokens)

This pipeline provides an accessible entry point to fact-checking automation while maintaining professional quality and comprehensive reporting capabilities.