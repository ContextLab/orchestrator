# Iterative Fact-Checking Pipeline

**Pipeline**: `examples/iterative_fact_checker.yaml`  
**Category**: Quality & Testing  
**Complexity**: Expert  
**Key Features**: While loops, Iterative processing, Quality thresholds, Document verification, Citation management

## Overview

The Iterative Fact-Checking Pipeline demonstrates advanced while loop capabilities by continuously improving document quality through iterative fact-checking cycles. It automatically identifies unreferenced claims, finds appropriate citations, and verifies existing references until a quality threshold is met.

## Key Features Demonstrated

### 1. While Loop with Quality Threshold
```yaml
while: "{{ quality_score | default(0) < parameters.quality_threshold }}"
max_iterations: 3
```

### 2. Dynamic Document Loading
```yaml
path: |
  {% if $iteration == 1 %}
  {{ parameters.input_document }}
  {% else %}
  {{ output_path }}/iteration_{{ $iteration - 1 }}_document.md
  {% endif %}
```

### 3. Structured Claim Extraction
```yaml
schema:
  type: object
  properties:
    claims:
      type: array
      items:
        properties:
          text: {type: string}
          has_reference: {type: boolean}
          reference_url: {type: string}
```

### 4. Variable Production for Loop Control
```yaml
- id: update_score
  produces: quality_score
  prompt: |
    Output just the decimal percentage (e.g., 0.95 for 95%).
```

## Pipeline Architecture

### Input Parameters
- **input_document** (optional): Path to document for fact-checking (default: "test_climate_document.md")
- **quality_threshold** (optional): Minimum reference percentage required (default: 0.95)
- **max_iterations** (optional): Maximum improvement cycles (default: 5)

### Processing Flow

1. **Initialize Process** - Set up fact-checking workflow
2. **Load Initial Document** - Read source document
3. **Iterative Fact-Checking Loop**:
   - Load current document version
   - Extract claims and analyze references
   - Verify existing reference URLs
   - Find citations for unreferenced claims
   - Update document with new references
   - Calculate quality score
   - Continue if threshold not met
4. **Save Final Document** - Store verified version
5. **Generate Report** - Create comprehensive fact-checking summary

### While Loop Logic

#### Loop Condition
```yaml
while: "{{ quality_score | default(0) < parameters.quality_threshold }}"
```

#### Iteration Management
```yaml
{% if $iteration == 1 %}
  # First iteration: use original document
  {{ parameters.input_document }}
{% else %}
  # Subsequent iterations: use previous iteration's output
  {{ output_path }}/iteration_{{ $iteration - 1 }}_document.md
{% endif %}
```

#### Quality Score Calculation
```yaml
produces: quality_score
prompt: |
  The document now has {{ claims_with_references }} out of {{ total_claims }} claims with references.
  This is {{ percentage_referenced }}% referenced.
  Output just the decimal percentage (e.g., 0.95 for 95%).
```

## Usage Examples

### Basic Fact-Checking
```bash
python scripts/run_pipeline.py examples/iterative_fact_checker.yaml \
  -i input_document="examples/data/test_article.md"
```

### Custom Quality Threshold
```bash
python scripts/run_pipeline.py examples/iterative_fact_checker.yaml \
  -i input_document="research_paper.md" \
  -i quality_threshold=0.98 \
  -i max_iterations=3
```

### Lenient Fact-Checking
```bash
python scripts/run_pipeline.py examples/iterative_fact_checker.yaml \
  -i input_document="blog_post.md" \
  -i quality_threshold=0.80
```

## Iterative Process Details

### Iteration 1: Initial Analysis
1. **Load original document**
2. **Extract all factual claims** with structured analysis
3. **Identify unreferenced claims**
4. **Verify existing references** for validity
5. **Find new citations** for unreferenced claims
6. **Update document** with new references
7. **Calculate quality score** (% of claims with references)

### Iteration 2+: Refinement
1. **Load previous iteration's document**
2. **Re-analyze claims** (may find new ones in added content)
3. **Verify all references** including newly added ones
4. **Add missing citations** for any remaining unreferenced claims
5. **Improve existing citations** with better sources
6. **Recalculate quality score**

### Loop Termination
The loop terminates when either:
- Quality threshold is reached (e.g., 95% of claims referenced)
- Maximum iterations limit is hit

## Sample Output Structure

### Generated Files
```
outputs/iterative_fact_checker/
├── iteration_1_document.md      # First improvement cycle
├── iteration_2_document.md      # Second improvement cycle  
├── iteration_3_document.md      # Third improvement cycle
├── test_climate_document_verified.md  # Final verified document
└── fact_checking_report.md      # Comprehensive report
```

### Fact-Checking Report Contents
```markdown
# Fact-Checking Report

## Processing Summary
- Total Iterations: 3
- Quality Threshold: 95%
- Final Quality Score: 98%

## Iteration Details
### Iteration 1
- Claims analyzed: 15
- Claims with references: 8
- Percentage referenced: 53%

### Iteration 2  
- Claims analyzed: 17
- Claims with references: 14
- Percentage referenced: 82%

### Iteration 3
- Claims analyzed: 18
- Claims with references: 18
- Percentage referenced: 100%
```

## Advanced Features

### Claim Analysis Schema
```yaml
schema:
  properties:
    claims:
      items:
        properties:
          text: {type: string}
          has_reference: {type: boolean}
          reference_url: {type: string}
    total_claims: {type: integer}
    claims_with_references: {type: integer}
    percentage_referenced: {type: number}
```

### Dynamic Path Resolution
```yaml
path: "{{ output_path }}/{{ parameters.input_document | basename | regex_replace('\\.md$', '') }}_verified.md"
```

### Reference Verification
```yaml
prompt: |
  Verify which of these reference URLs are valid and accessible:
  {% for claim in extract_claims.claims %}
  {% if claim.has_reference and claim.reference_url %}
  - {{ claim.reference_url }}
  {% endif %}
  {% endfor %}
```

## Quality Metrics

### Reference Coverage
- **Total Claims**: All factual assertions in document
- **Referenced Claims**: Claims with proper citations
- **Coverage Percentage**: (Referenced / Total) × 100

### Iteration Efficiency
- **Claims Added per Iteration**: New claims discovered
- **References Added per Iteration**: New citations provided
- **Quality Improvement**: Percentage point increase per cycle

### Final Quality Assessment
- **Threshold Achievement**: Whether target quality was reached
- **Remaining Gaps**: Unreferenced claims still present
- **Source Quality**: Credibility of added references

## Technical Implementation

### While Loop Variables
```yaml
# Available in while loop context
$iteration          # Current iteration number (1-based)
quality_score      # Produced by update_score step
```

### Template Logic
```yaml
{% if fact_check_loop.iterations %}
  {{ fact_check_loop.iterations[-1].extract_claims.percentage_referenced }}%
{% else %}
  N/A
{% endif %}
```

### Dependency Management
```yaml
dependencies:
  - load_initial_doc      # Before while loop
  - fact_check_loop       # After while loop completes
```

## Best Practices Demonstrated

1. **Iterative Improvement**: Gradually enhance document quality
2. **Quality Thresholds**: Define measurable success criteria
3. **Structured Data**: Use schemas for consistent analysis
4. **Progress Tracking**: Monitor improvement across iterations
5. **Comprehensive Reporting**: Document all changes and decisions
6. **Reference Verification**: Validate citation accessibility
7. **Flexible Termination**: Balance quality vs. iteration limits

## Common Use Cases

- **Academic Paper Review**: Ensure all claims have proper citations
- **Journalism Fact-Checking**: Verify article accuracy with sources
- **Content Quality Assurance**: Improve reference coverage
- **Research Validation**: Check scientific claim support
- **Legal Document Review**: Verify legal precedent citations
- **Educational Content**: Ensure teaching materials are well-sourced

## Troubleshooting

### Loop Not Terminating
- Check quality_score variable is being properly produced
- Verify threshold is achievable (not too high)
- Ensure max_iterations prevents infinite loops

### Quality Score Issues
- Confirm percentage calculation is correct (decimal format)
- Check if new claims are being added during iterations
- Verify structured extraction is consistent

### Reference Verification Problems
- Ensure URL accessibility checks are realistic
- Handle network timeouts and connectivity issues
- Consider rate limiting for external URL verification

## Related Examples
- [iterative_fact_checker_simple.md](iterative_fact_checker_simple.md) - Simplified version
- [fact_checker.md](fact_checker.md) - Single-pass fact-checking
- [enhanced_until_conditions_demo.md](enhanced_until_conditions_demo.md) - Advanced loop conditions

## Technical Requirements

- **Model**: Claude Sonnet 4 for high-quality structured generation and analysis
- **File System**: Read/write access for document processing and iteration storage
- **Template Engine**: Jinja2 for conditional logic and iteration management
- **Structured Generation**: JSON schema validation capabilities
- **Network Access**: Optional, for reference URL verification

This pipeline demonstrates enterprise-grade iterative document processing with measurable quality improvement through controlled while loop execution.