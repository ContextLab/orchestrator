# Iterative Number Guessing Pipeline

**Pipeline**: `examples/control_flow_while_loop.yaml`  
**Category**: Control Flow & Logic  
**Complexity**: Intermediate  
**Key Features**: While loops, State management, File-based iteration, AUTO model selection, Template variables

## Overview

The Iterative Number Guessing Pipeline demonstrates while loop control flow through an interactive number guessing game. It showcases stateful iteration, file-based state persistence, and template variable access within loop contexts, providing a practical example of iterative processing patterns.

## Key Features Demonstrated

### 1. While Loop Structure
```yaml
# While loop with max iterations safety
- id: guessing_loop
  while: 'true'  # Simple always-true condition
  max_iterations: "{{ max_attempts }}"
```

### 2. Loop Iteration Variables
```yaml
parameters:
  prompt: |
    Current guess: {{ read_guess.content }}
    Iteration: {{ guessing_loop.iteration }}
    
    What number should we guess next?
```

### 3. State File Management
```yaml
# Read current state
- id: read_guess
  tool: filesystem
  action: read
  parameters:
    path: "examples/outputs/control_flow_while_loop/state/current_guess.txt"

# Update state
- id: update_state  
  tool: filesystem
  action: write
  parameters:
    path: "examples/outputs/control_flow_while_loop/state/current_guess.txt"
    content: "{{ generate_guess.result | regex_search('[0-9]+') | default('25') }}"
```

### 4. Dynamic Logging
```yaml
# Log each iteration
- id: log_attempt
  tool: filesystem
  action: write
  parameters:
    path: "examples/outputs/control_flow_while_loop/logs/attempt_{{ guessing_loop.iteration }}.txt"
    content: |
      Iteration: {{ guessing_loop.iteration }}
      Previous guess: {{ read_guess.content }}
      New guess: {{ generate_guess.result | regex_search('[0-9]+') | default('25') }}
```

### 5. Regular Expression Processing
```yaml
# Extract numbers from AI responses
content: "{{ generate_guess.result | regex_search('[0-9]+') | default('25') }}"
```

## Pipeline Architecture

### Input Parameters
- **target_number** (optional): Number to guess (default: 42)
- **max_attempts** (optional): Maximum iterations allowed (default: 10)

### Processing Flow

1. **Initialization** - Sets up the guessing game with target number
2. **State Initialization** - Creates initial state file with starting guess (0)
3. **While Loop Execution** - Iterative guessing process:
   - **Read Current State** - Loads previous guess from state file
   - **Generate New Guess** - AI generates next number guess
   - **Extract Number** - Parses number from AI response using regex
   - **Update State** - Saves new guess to state file
   - **Log Attempt** - Records iteration details to log file
   - **Check Result** - Analyzes if guess matches target
   - **Update Loop State** - Prepares for next iteration
4. **Final Result** - Generates summary report of all attempts

### State Management System

#### State Files
- **Current Guess**: `state/current_guess.txt` - Tracks the latest guess
- **Attempt Logs**: `logs/attempt_N.txt` - Individual iteration records
- **Final Result**: `result.txt` - Complete game summary

#### State Persistence
The pipeline maintains state across iterations using filesystem storage:
- Reads previous state at loop start
- Updates state after each guess
- Logs each iteration for debugging and analysis
- Preserves state between pipeline executions

## Usage Examples

### Basic Number Guessing
```bash
python scripts/run_pipeline.py examples/control_flow_while_loop.yaml \
  -i target_number=42
```

### Custom Target and Attempt Limit
```bash
python scripts/run_pipeline.py examples/control_flow_while_loop.yaml \
  -i target_number=75 \
  -i max_attempts=15
```

### Quick Testing
```bash
python scripts/run_pipeline.py examples/control_flow_while_loop.yaml \
  -i target_number=5 \
  -i max_attempts=5
```

### Large Number Challenge
```bash
python scripts/run_pipeline.py examples/control_flow_while_loop.yaml \
  -i target_number=999 \
  -i max_attempts=20
```

## Sample Output Structure

### Final Result File (`result.txt`)
```markdown
# Number Guessing Results

Target number: 42
Total attempts: 3
Success: False
```

### Individual Attempt Logs (`logs/attempt_N.txt`)
```
Iteration: 0
Previous guess: 0
New guess: 21
Target: 42
```

### State File (`state/current_guess.txt`)
```
21
```

### Generated Output Files
Check the actual generated outputs in:
- [result.txt](../../examples/outputs/control_flow_while_loop/result.txt)
- [logs/attempt_*.txt](../../examples/outputs/control_flow_while_loop/logs/)
- [state/current_guess.txt](../../examples/outputs/control_flow_while_loop/state/current_guess.txt)

## Technical Implementation

### While Loop Mechanics
```yaml
# Loop configuration
while: 'true'  # Always true condition
max_iterations: "{{ max_attempts }}"  # Safety limit

# Loop variables available in steps:
# - guessing_loop.iteration (current iteration number)
# - guessing_loop.iterations (total completed iterations)
# - guessing_loop.completed (loop completion status)
```

### Regular Expression Processing
```yaml
# Extract numbers from AI responses with fallback
{{ generate_guess.result | regex_search('[0-9]+') | default('25') }}

# Pattern matches first sequence of digits
# Falls back to '25' if no number found
```

### File-Based State Management
```yaml
# State directory structure
examples/outputs/control_flow_while_loop/
├── state/
│   └── current_guess.txt      # Current guess state
├── logs/
│   ├── attempt_0.txt          # First iteration log
│   ├── attempt_1.txt          # Second iteration log
│   └── ...                    # Additional attempts
└── result.txt                 # Final results summary
```

### Template Variable Access
Within loop steps, access to multiple variable scopes:
```yaml
prompt: |
  We're trying to guess the number {{ target_number }}.        # Pipeline parameter
  Current guess: {{ read_guess.content }}                      # Step result
  Iteration: {{ guessing_loop.iteration }}                     # Loop variable
```

## Advanced Features

### Robust Number Extraction
```yaml
# Multi-step number processing
- id: generate_guess      # AI generates response
- id: extract_number      # Analyzes and validates number
- id: update_state        # Saves validated number

# Regex with default fallback ensures state consistency
```

### Comprehensive Logging
Each iteration creates a detailed log file:
```yaml
content: |
  Iteration: {{ guessing_loop.iteration }}
  Previous guess: {{ read_guess.content }}
  New guess: {{ generate_guess.result | regex_search('[0-9]+') | default('25') }}
  Target: {{ target_number }}
```

### Loop Safety Mechanisms
- **Max Iterations**: Prevents infinite loops
- **State Validation**: Ensures valid numbers are processed
- **Default Values**: Fallbacks prevent pipeline failures
- **Comprehensive Logging**: Enables debugging and analysis

### Model Selection Strategy
```yaml
model: <AUTO task="summarize">Select a model for initialization</AUTO>
model: <AUTO task="generate">Select a model for number generation</AUTO>
model: <AUTO task="analyze">Select a model for number extraction</AUTO>
model: <AUTO task="analyze">Select a model for comparison</AUTO>
```

## Common Use Cases

- **Iterative Optimization**: Algorithms that improve through iterations
- **State-Based Processing**: Workflows requiring persistent state
- **Progressive Data Analysis**: Multi-pass data processing
- **Simulation Workflows**: Step-by-step simulation execution
- **Training Loops**: Machine learning training iterations
- **Convergence Testing**: Algorithms that iterate until convergence
- **Batch Processing**: Processing items until conditions are met

## Best Practices Demonstrated

1. **State Persistence**: File-based state management for reliability
2. **Loop Safety**: Max iterations prevent infinite execution
3. **Comprehensive Logging**: Detailed iteration tracking for debugging
4. **Error Handling**: Regex fallbacks and default values
5. **Template Flexibility**: Rich variable access within loops
6. **File Organization**: Structured output directory management
7. **Variable Scoping**: Proper access to pipeline, step, and loop variables

## Troubleshooting

### Common Issues
- **State File Corruption**: Ensure write permissions for state directory
- **Infinite Loops**: Always set appropriate max_iterations
- **Number Extraction**: Validate AI responses contain extractable numbers
- **File Path Errors**: Verify output directory structure exists

### Performance Considerations
- **File I/O Overhead**: Each iteration involves multiple file operations
- **Model Selection**: AUTO selection optimizes for task requirements
- **Log File Size**: Multiple iterations create numerous log files
- **State Size**: Keep state data minimal for efficiency

### Debugging Tips
- **Check Log Files**: Each iteration creates detailed logs
- **Verify State Files**: Inspect current_guess.txt for state consistency
- **Monitor Iterations**: Track guessing_loop.iteration values
- **Validate Regex**: Test number extraction patterns

## Related Examples
- [control_flow_conditional.md](control_flow_conditional.md) - Conditional processing
- [control_flow_for_loop.md](control_flow_for_loop.md) - For loop patterns  
- [until_condition_examples.md](until_condition_examples.md) - Until loop patterns
- [enhanced_until_conditions_demo.md](enhanced_until_conditions_demo.md) - Advanced until conditions

## Technical Requirements

- **Models**: Support for text generation, analysis, and number processing
- **Tools**: Filesystem access for state and log file management
- **Templates**: Jinja2 templating with regex filter support
- **File System**: Write permissions for output directories

This pipeline provides a practical foundation for implementing iterative processing patterns with state management, essential for building sophisticated workflows that require persistent state across multiple execution cycles.