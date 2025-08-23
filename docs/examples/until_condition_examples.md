# Until Condition Examples Pipeline

**Pipeline**: `examples/until_condition_examples.yaml`  
**Category**: Control Flow  
**Complexity**: Advanced  
**Key Features**: Until conditions, While-until combinations, Sequential processing, Quality thresholds, Error recovery

## Overview

The Until Condition Examples Pipeline demonstrates advanced loop control using until conditions for quality verification, source validation, and error recovery scenarios. It showcases real-world applications of sequential processing with termination criteria and combines while loops with until conditions for robust workflow control.

## Key Features Demonstrated

### 1. Sequential Source Verification
```yaml
- id: verify_sources_sequential
  for_each: ["https://example.com/ai-research", "https://arxiv.org/invalid-link", "https://en.wikipedia.org/wiki/Artificial_intelligence"]
  until: "{{ validate_source.result }} == true or {{ validate_source.result }} == false"
```

### 2. While-Until Combination
```yaml
- id: quality_improvement_loop
  while: "{{ $iteration }} < 5"  # Safety limit
  until: "{{ current_quality }} >= {{ quality_threshold }}"
```

### 3. Error Recovery with Until
```yaml
- id: pdf_generation_with_recovery
  while: "{{ $iteration }} < 3"  # Max 3 attempts
  until: "{{ pdf_exists }} == true and {{ pdf_valid }} == true"
```

### 4. Step Result References in Until
```yaml
until: "{{ check_threshold.result }} == true"
```

## Pipeline Architecture

### Input Parameters
- **search_topic** (optional): Research topic for content generation (default: "artificial intelligence research")
- **quality_threshold** (optional): Quality score threshold (default: 0.8)

### Processing Flow

1. **Sequential Source Verification** - Validate sources until determination complete
2. **Quality Improvement Loop** - Enhance content until quality threshold met
3. **PDF Generation with Recovery** - Generate PDF with error recovery until successful
4. **Counter Example** - Simple counter until condition demonstration
5. **Process Items Until Complete** - Process array items until threshold met

## Until Condition Patterns

### Pattern 1: Validation Until Complete
```yaml
until: "{{ validate_source.result }} == true or {{ validate_source.result }} == false"
# Stops when source is either validated or determined invalid
```

### Pattern 2: Quality Threshold
```yaml
until: "{{ current_quality }} >= {{ quality_threshold }}"
# Stops when quality score meets or exceeds threshold
```

### Pattern 3: Complex Boolean Logic
```yaml
until: "{{ pdf_exists }} == true and {{ pdf_valid }} == true"
# Stops when both conditions are satisfied
```

### Pattern 4: Numerical Threshold
```yaml
until: "{{ counter }} >= 3"
# Stops when counter reaches specified value
```

### Pattern 5: Step Result Reference
```yaml
until: "{{ check_threshold.result }} == true"
# Stops based on specific step result
```

## Usage Examples

### Basic Until Condition Testing
```bash
python scripts/run_pipeline.py examples/until_condition_examples.yaml \
  -i search_topic="machine learning applications"
```

### Custom Quality Threshold
```bash
python scripts/run_pipeline.py examples/until_condition_examples.yaml \
  -i quality_threshold=0.9 \
  -i search_topic="quantum computing research"
```

### Lower Quality Threshold for Testing
```bash
python scripts/run_pipeline.py examples/until_condition_examples.yaml \
  -i quality_threshold=0.6 \
  -i search_topic="blockchain technology"
```

## Detailed Example Breakdown

### Example 1: Sequential Source Verification

#### Purpose
Validate a list of sources sequentially, stopping when each source is definitively validated or marked invalid.

#### Loop Structure
```yaml
for_each: ["https://example.com/ai-research", "https://arxiv.org/invalid-link", "https://en.wikipedia.org/wiki/Artificial_intelligence"]
until: "{{ validate_source.result }} == true or {{ validate_source.result }} == false"
```

#### Process Flow
1. **Check Source**: Send HEAD request to URL
2. **Validate Source**: Check if status code is 200
3. **Log Result**: Record validation outcome
4. **Until Check**: Stop when validation is complete (true or false)

#### Execution Pattern
```
Source 1: https://example.com/ai-research
├─ check_source → HEAD request → 200 OK
├─ validate_source → true
├─ log_result → "Source verified"
└─ until → true == true → STOP

Source 2: https://arxiv.org/invalid-link
├─ check_source → HEAD request → 404 Not Found
├─ validate_source → false  
├─ log_result → "Source invalid"
└─ until → false == false → STOP
```

### Example 2: Quality Improvement Loop

#### Purpose
Iteratively improve content quality until it meets the specified threshold.

#### Loop Structure
```yaml
while: "{{ $iteration }} < 5"  # Safety limit
until: "{{ current_quality }} >= {{ quality_threshold }}"
```

#### Process Flow
1. **Generate Content**: Create/improve content using LLM
2. **Evaluate Quality**: Score content quality (0-1 scale)
3. **Update Variables**: Store content and quality for next iteration
4. **Until Check**: Stop when quality threshold reached

#### Variable Management
```yaml
produces: improved_content     # Content from generation step
produces: quality_score       # Quality assessment
parameters:
  previous_content: "{{ generate_content.result }}"
  current_quality: "{{ evaluate_quality.result | float }}"
```

### Example 3: PDF Generation with Recovery

#### Purpose
Generate PDF with automatic error recovery, retrying until successful.

#### Loop Structure
```yaml
while: "{{ $iteration }} < 3"  # Max 3 attempts
until: "{{ pdf_exists }} == true and {{ pdf_valid }} == true"
```

#### Recovery Process
1. **Generate Markdown**: Create markdown source file
2. **Compile PDF**: Use pandoc to convert markdown to PDF
3. **Check PDF Exists**: Verify PDF file was created
4. **Validate PDF**: Confirm file is valid PDF format
5. **Debug on Failure**: Log diagnostics if generation failed
6. **Until Check**: Stop when PDF exists and is valid

#### Error Handling
```yaml
on_error: continue              # Don't fail on individual step errors
timeout: 30                     # Prevent hanging operations
```

### Example 4: Simple Counter

#### Purpose
Demonstrate basic until condition with counter increment.

#### Loop Logic
```yaml
while: "true"                   # Always continue while condition
until: "{{ counter }} >= 3"     # Stop when counter reaches 3
```

#### Counter Management
```yaml
- id: increment_counter
  parameters:
    condition: "{{ counter | default(0) | int + 1 }}"
  produces: new_counter

- id: update_counter
  action: set_variables
  parameters:
    counter: "{{ increment_counter.result }}"
```

### Example 5: Process Items Until Complete

#### Purpose
Process array items sequentially until a threshold condition is met.

#### Processing Logic
```yaml
for_each: [1, 2, 3, 4, 5]
until: "{{ check_threshold.result }} == true"
```

#### Threshold Logic
```yaml
- id: process_item
  parameters:
    condition: "{{ $item }} * 2"       # Double the item value
  produces: processed_value

- id: check_threshold  
  parameters:
    condition: "{{ process_item.result }} >= 6"  # Check if >= 6
  produces: threshold_met
```

#### Execution Sequence
```
Item 1: 1 * 2 = 2 → 2 >= 6? false → CONTINUE
Item 2: 2 * 2 = 4 → 4 >= 6? false → CONTINUE  
Item 3: 3 * 2 = 6 → 6 >= 6? true → STOP
```

## Advanced Until Condition Features

### Variable Production and Reference
```yaml
produces: quality_score         # Produce variable from step
until: "{{ quality_score }} >= {{ quality_threshold }}"  # Reference in condition
```

### Complex Boolean Logic
```yaml
until: "{{ condition_a }} == true and {{ condition_b }} == true"
until: "{{ result }} >= {{ threshold }} or {{ max_attempts_reached }}"
```

### Step Result References
```yaml
until: "{{ specific_step.result }} == expected_value"
until: "{{ validation_step.success }} == true"
```

### Default Value Handling
```yaml
until: "{{ counter | default(0) }} >= 3"
until: "{{ quality | default(0.0) | float }} >= {{ threshold }}"
```

## Safety and Best Practices

### Iteration Limits
```yaml
while: "{{ $iteration }} < 5"   # Prevent infinite loops
# Combined with until condition for dual protection
```

### Error Continuity
```yaml
on_error: continue              # Continue processing despite step failures
# Allows recovery attempts to proceed
```

### Variable Initialization
```yaml
parameters:
  counter: "{{ counter | default(0) | int + 1 }}"
  quality: "{{ current_quality | default(0.3) }}"
```

### Timeout Protection
```yaml
timeout: 30                     # Prevent hanging operations
timeout: 5                      # Quick operations
```

## Common Use Cases

### Quality Assurance Workflows
- Content quality improvement until threshold met
- Code quality validation until standards satisfied
- Data validation until accuracy achieved

### Resource Availability Checking
- Wait until service becomes available
- Check until file processing complete
- Monitor until resource requirements met

### Error Recovery Patterns
- Retry operations until successful
- Generate outputs until valid
- Process items until completion criteria met

### Sequential Processing
- Validate items until all processed
- Transform data until requirements satisfied
- Aggregate results until target reached

## Technical Implementation

### Until Condition Evaluation
```yaml
# Evaluated after each loop iteration
# Boolean expression that determines loop termination
until: "{{ expression }}"
```

### Variable Scope
```yaml
# Loop variables available in until conditions:
$iteration                      # Current iteration number
$item                          # Current for_each item (if applicable)
# Plus all produced variables from steps
```

### Template Expression Support
```yaml
until: "{{ variable | filter }} == value"
until: "{{ step_result.property }} >= {{ threshold }}"
```

## Troubleshooting

### Infinite Loop Prevention
- Always include while condition as safety limit
- Use reasonable iteration counts (typically < 10)
- Test until conditions with various input scenarios

### Variable Reference Issues
- Ensure produced variables are correctly named
- Use default values for potentially undefined variables
- Check variable scope within loop context

### Boolean Logic Errors
- Test complex boolean expressions independently
- Use parentheses for clarity in complex conditions
- Validate comparison operators and value types

## Related Examples
- [enhanced_until_conditions_demo.md](enhanced_until_conditions_demo.md) - Enhanced until condition patterns
- [control_flow_while_loop.md](control_flow_while_loop.md) - Basic while loop patterns
- [iterative_fact_checker.md](iterative_fact_checker.md) - Quality-driven iterative processing

## Technical Requirements

- **Loop Support**: While and until condition processing
- **Variable Production**: Step-level variable creation and scoping
- **Boolean Evaluation**: Complex boolean expression evaluation
- **Template Engine**: Variable interpolation and filtering
- **Error Handling**: Graceful error recovery mechanisms

This pipeline demonstrates production-ready until condition patterns suitable for quality-driven workflows, error recovery scenarios, and sequential processing with termination criteria.