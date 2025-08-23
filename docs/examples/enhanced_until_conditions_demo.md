# Enhanced Until Conditions Demo Pipeline

**Pipeline**: `examples/enhanced_until_conditions_demo.yaml`  
**Category**: Control Flow & Logic  
**Complexity**: Intermediate  
**Key Features**: Until conditions, AUTO evaluation, Quality improvement loops, Performance tracking, Variable scope

## Overview

The Enhanced Until Conditions Demo Pipeline demonstrates sophisticated until condition evaluation with performance tracking. It showcases how pipelines can continue iterating until specific quality thresholds are met, using AI-powered condition evaluation to determine when to stop processing based on dynamic criteria.

## Key Features Demonstrated

### 1. AUTO Until Condition Evaluation
```yaml
until: "<AUTO>Quality score is {{ quality_score }}. Is this >= {{ target_quality }}? Answer 'true' or 'false'.</AUTO>"
```

### 2. Combined While/Until Loop Structure
```yaml
while: "true"  # Always continue from while perspective  
until: "<AUTO>Quality score evaluation</AUTO>"
max_iterations: "{{ max_attempts }}"
loop_name: "quality_loop"
```

### 3. Dynamic Quality Scoring
```yaml
# Simulate quality improvement over iterations
message: "Improving quality from {{ quality_score | default('0.2') }} to {{ ($iteration + 1) * 0.25 }}"
quality_score: "{{ ($iteration + 1) * 0.25 }}"
```

### 4. Loop Variable Access
```yaml
parameters:
  message: "Iteration {{ $iteration }}: Current quality score: {{ quality_score | default('0.2') }}"
  message: "Loop {{ quality_loop.loop_name }} iteration {{ $iteration }} complete"
```

### 5. Performance Tracking
```yaml
outputs:
  - name: final_iteration_count
    value: "{{ quality_loop.iteration }}"
```

## Pipeline Architecture

### Input Parameters
- **target_quality** (optional): Quality threshold to achieve (default: 0.8)
- **max_attempts** (optional): Maximum loop iterations allowed (default: 5)

### Processing Flow

1. **Metrics Initialization** - Sets up the quality improvement demonstration
2. **Quality Improvement Loop** - Iterative processing until quality threshold reached:
   - **Work Simulation** - Simulates quality improvement work
   - **Quality Score Update** - Increments quality score based on iteration
   - **Progress Check** - Reports loop progress and current quality
3. **Final Report** - Generates completion summary

### Until Condition Logic

The pipeline uses sophisticated until condition evaluation:
```yaml
until: "<AUTO>Quality score is {{ quality_score }}. Is this >= {{ target_quality }}? Answer 'true' or 'false'.</AUTO>"
```

This demonstrates:
- **AI-Powered Evaluation**: Uses AUTO tags for intelligent condition assessment
- **Dynamic Variables**: References changing quality scores and target thresholds
- **Boolean Response**: Expects clear true/false answers for loop continuation
- **Context Awareness**: Understands the comparison relationship between values

## Usage Examples

### Basic Quality Improvement Loop
```bash
python scripts/run_pipeline.py examples/enhanced_until_conditions_demo.yaml \
  --input target_quality=0.8
```

### Custom Quality Target
```bash
python scripts/run_pipeline.py examples/enhanced_until_conditions_demo.yaml \
  --input target_quality=0.9 \
  --input max_attempts=8
```

### Low Quality Threshold Test
```bash
python scripts/run_pipeline.py examples/enhanced_until_conditions_demo.yaml \
  --input target_quality=0.6 \
  --input max_attempts=3
```

### Maximum Iterations Test
```bash
python scripts/run_pipeline.py examples/enhanced_until_conditions_demo.yaml \
  --input target_quality=1.0 \
  --input max_attempts=10
```

## Sample Output Structure

### Debug Messages During Execution
```
Starting enhanced until condition demo with target quality: 0.8
Iteration 0: Current quality score: 0.2
Improving quality from 0.2 to 0.25
Loop quality_loop iteration 0 complete. Quality: 0.25

Iteration 1: Current quality score: 0.25
Improving quality from 0.25 to 0.5
Loop quality_loop iteration 1 complete. Quality: 0.5

Iteration 2: Current quality score: 0.5
Improving quality from 0.5 to 0.75
Loop quality_loop iteration 2 complete. Quality: 0.75

Iteration 3: Current quality score: 0.75
Improving quality from 0.75 to 1.0
Loop quality_loop iteration 3 complete. Quality: 1.0

Quality improvement loop completed. Final quality score achieved.
```

### Output Variables
```yaml
completion_status: "Enhanced until conditions demo completed successfully"
final_iteration_count: "3"  # Number of iterations required to reach target
```

## Technical Implementation

### Until Condition Evaluation
```yaml
# AI-powered condition evaluation
until: "<AUTO>Quality score is {{ quality_score }}. Is this >= {{ target_quality }}? Answer 'true' or 'false'.</AUTO>"

# The AUTO tag provides context-aware evaluation:
# - Compares current quality_score against target_quality
# - Returns boolean response for loop control
# - Enables natural language condition specification
```

### Quality Score Simulation
```yaml
# Progressive quality improvement
quality_score: "{{ ($iteration + 1) * 0.25 }}"

# Simulates realistic improvement:
# - Iteration 0: 0.25 (25% quality)
# - Iteration 1: 0.50 (50% quality)  
# - Iteration 2: 0.75 (75% quality)
# - Iteration 3: 1.00 (100% quality)
```

### Variable Scoping
```yaml
# Loop iteration variable
{{ $iteration }}

# Loop metadata access
{{ quality_loop.loop_name }}
{{ quality_loop.iteration }}

# Dynamic variable with defaults
{{ quality_score | default('0.2') }}
```

### Performance Tracking
```yaml
# Named loops enable metadata tracking
loop_name: "quality_loop"

# Outputs capture final metrics
final_iteration_count: "{{ quality_loop.iteration }}"
```

## Advanced Features

### Dual Loop Control
The pipeline uses both `while` and `until` conditions:
```yaml
while: "true"  # Continue indefinitely from while perspective
until: "<AUTO>...</AUTO>"  # Stop when condition becomes true
max_iterations: "{{ max_attempts }}"  # Safety limit
```

This pattern enables:
- **Flexible Termination**: Multiple exit conditions
- **Safety Limits**: Prevents infinite loops
- **Clear Logic**: Natural language condition specification

### Context-Aware AUTO Evaluation
```yaml
until: "<AUTO>Quality score is {{ quality_score }}. Is this >= {{ target_quality }}? Answer 'true' or 'false'.</AUTO>"

# The AUTO tag understands:
# - Variable values (quality_score, target_quality)
# - Comparison operations (>=)
# - Expected response format (true/false)
# - Loop continuation logic
```

### Dynamic Default Values
```yaml
# Provides fallback when quality_score is undefined
{{ quality_score | default('0.2') }}

# Enables graceful handling of initial loop iterations
```

### Loop Naming and Metadata
```yaml
loop_name: "quality_loop"
# Enables reference to loop metadata:
# - quality_loop.iteration (current iteration count)
# - quality_loop.loop_name (loop identifier)
```

## Common Use Cases

- **Quality Assurance Workflows**: Iterate until quality standards are met
- **Performance Optimization**: Continue tuning until performance targets achieved
- **Data Processing**: Process data until completeness criteria satisfied
- **Training Loops**: Continue training until accuracy thresholds reached
- **Convergence Testing**: Iterate until mathematical convergence achieved
- **User Experience Testing**: Refine until user satisfaction targets met
- **Content Generation**: Improve content until quality benchmarks reached

## Best Practices Demonstrated

1. **Clear Exit Conditions**: Well-defined until conditions with AI evaluation
2. **Progress Tracking**: Comprehensive logging of loop progress
3. **Safety Limits**: Maximum iteration safeguards against infinite loops
4. **Variable Scoping**: Proper access to loop variables and metadata
5. **Performance Metrics**: Capture and report loop execution statistics
6. **Graceful Degradation**: Default values for undefined variables
7. **Natural Language Logic**: Human-readable condition specifications

## Troubleshooting

### Common Issues
- **Condition Evaluation**: Ensure AUTO tag receives clear context for evaluation
- **Variable Scope**: Verify loop variables are accessible in condition statements
- **Infinite Loops**: Always set appropriate max_iterations values
- **Quality Score Updates**: Ensure quality_score variable is properly updated

### Performance Considerations
- **AUTO Evaluation Overhead**: Each iteration requires AI model evaluation
- **Debug Message Volume**: Consider reducing debug output for production use
- **Loop Complexity**: Complex conditions may slow execution
- **Variable Resolution**: Template variable resolution adds processing time

### Debugging Tips
- **Enable Debug Messages**: Use debug action for comprehensive loop tracing
- **Monitor Quality Progression**: Track quality_score changes across iterations
- **Check Until Condition Logic**: Verify AUTO evaluation logic is sound
- **Validate Variable Access**: Ensure all referenced variables are available

## Related Examples
- [until_condition_examples.md](until_condition_examples.md) - Basic until condition patterns
- [control_flow_while_loop.md](control_flow_while_loop.md) - While loop implementations
- [control_flow_for_loop.md](control_flow_for_loop.md) - For loop patterns
- [iterative_fact_checker.md](iterative_fact_checker.md) - Iterative quality improvement

## Technical Requirements

- **Models**: Support for AUTO tag evaluation and boolean logic
- **Variables**: Template variable resolution with default values
- **Loops**: Support for named loops with metadata tracking
- **Debug**: Debug action capability for progress tracking

This pipeline demonstrates sophisticated loop control patterns essential for building adaptive workflows that continue processing until quality, performance, or completion criteria are satisfied through intelligent condition evaluation.