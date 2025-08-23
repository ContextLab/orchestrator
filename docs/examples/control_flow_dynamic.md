# Dynamic Flow Control Pipeline

**Pipeline**: `examples/control_flow_dynamic.yaml`  
**Category**: Control Flow & Logic  
**Complexity**: Advanced  
**Key Features**: Dynamic control flow, Error handling, Terminal integration, Risk assessment, AUTO model selection

## Overview

The Dynamic Flow Control Pipeline demonstrates sophisticated error handling and dynamic execution patterns by processing terminal operations through multiple validation stages. It showcases real-world error handling, risk assessment, and conditional execution flows that adapt based on runtime conditions and execution results.

## Key Features Demonstrated

### 1. Multi-Stage Validation Process
```yaml
# Input validation
- id: validate_input
  action: generate_text
  prompt: |
    Validate if this is a safe operation to execute: "{{ operation }}"
    Return only "valid" if safe, or "invalid" if unsafe.

# Risk assessment
- id: assess_risk
  action: generate_text  
  prompt: |
    Assess the risk level of this operation: "{{ operation }}"
    Return ONLY one word: low, medium, or high
```

### 2. Conditional Safety Checks
```yaml
# High-risk operations get additional safety validation
- id: safety_check
  action: generate_text
  condition: "{{ assess_risk == 'high' }}"
  parameters:
    prompt: |
      Perform additional safety check for high-risk operation: {{ operation }}
      Return "safe" or "unsafe"
```

### 3. Real Terminal Execution
```yaml
# Actual command execution
- id: execute_operation
  tool: terminal
  action: execute
  parameters:
    command: "{{ operation }}"
```

### 4. Dynamic Result Handling
```yaml
# Success path
- id: success_handler
  condition: "{{ check_result == 'success' }}"

# Failure path  
- id: failure_handler
  condition: "{{ check_result == 'failure' }}"
```

### 5. Comprehensive Model Selection
```yaml
model: <AUTO task="validate">Select a model for validation</AUTO>
model: <AUTO task="assess">Select a model for risk assessment</AUTO>
model: <AUTO task="safety">Select a model</AUTO>
model: <AUTO task="check">Select a model</AUTO>
```

## Pipeline Architecture

### Input Parameters
- **operation** (optional): Terminal command to execute (default: "echo 'Hello, World!'")
- **retry_limit** (optional): Maximum retry attempts (default: 3)

### Processing Flow

1. **Input Validation** - Validates the safety of the requested operation
2. **Risk Assessment** - Determines risk level (low/medium/high) of the operation
3. **Operation Preparation** - Prepares for execution based on validation results
4. **Conditional Safety Check** - Additional validation for high-risk operations
5. **Terminal Execution** - Executes the actual terminal command
6. **Result Analysis** - Analyzes execution output to determine success/failure
7. **Conditional Result Handling** - Routes to success or failure handling
8. **Cleanup Operations** - Performs final cleanup tasks
9. **Report Generation** - Creates comprehensive execution report

### Risk Assessment Levels

#### Low Risk Operations
- **Examples**: `echo`, `pwd`, `date`, basic file listing
- **Processing**: Standard validation and execution
- **Safety**: Minimal additional checks required

#### Medium Risk Operations  
- **Examples**: File operations, directory changes, basic calculations
- **Processing**: Enhanced validation and monitoring
- **Safety**: Standard precautions applied

#### High Risk Operations
- **Examples**: System modifications, network operations, process management
- **Processing**: Maximum validation with additional safety checks
- **Safety**: Comprehensive safety validation before execution

## Usage Examples

### Basic Safe Operation
```bash
python scripts/run_pipeline.py examples/control_flow_dynamic.yaml \
  -i operation="echo 'Hello, World!'"
```

### File System Operation
```bash
python scripts/run_pipeline.py examples/control_flow_dynamic.yaml \
  -i operation="ls -la" \
  -i retry_limit=2
```

### System Information Commands
```bash
python scripts/run_pipeline.py examples/control_flow_dynamic.yaml \
  -i operation="date && whoami && pwd"
```

### Mathematical Operations
```bash
python scripts/run_pipeline.py examples/control_flow_dynamic.yaml \
  -i operation="python -c 'print(2+2)'"
```

### With Retry Configuration
```bash
python scripts/run_pipeline.py examples/control_flow_dynamic.yaml \
  -i operation="curl -s https://httpbin.org/status/200" \
  -i retry_limit=5
```

## Sample Output Structure

### Execution Report Format
```markdown
# Dynamic Flow Control Execution Report

**Operation:** [command]
**Risk Level:** [low|medium|high]

## Execution Summary

- Validation: [valid|invalid]
- Preparation: [ready|not ready]
- Execution Result: [success|failure]

## Command Execution Details

- **Command:** [executed command]
- **Return Code:** [exit code]
- **Success:** [true|false]
- **Execution Time:** [time in ms]

### Command Output (stdout):
```
[command output]
```

### Command Errors (stderr):
```
[error output if any]
```

## Report Details

[Success or failure summary with bulleted details]

## Cleanup Status

cleaned

---
*Generated by Dynamic Flow Control Pipeline*
```

### Example Success Report
```markdown
- Operation executed successfully
- Risk level: low
- Command completed without errors
```

### Example Failure Report
```markdown
- Operation: [command]
- Risk Level: [level]
- Status: FAILED
- Error: Execution did not complete successfully
```

## Technical Implementation

### Validation Chain Design
```yaml
# Multi-stage validation
validate_input → assess_risk → prepare_operation
                      ↓
             [conditional safety_check]
                      ↓
                execute_operation
                      ↓
                 check_result
                      ↓
        [success_handler | failure_handler]
                      ↓
                   cleanup
                      ↓
                 save_report
```

### Dynamic Dependency Management
The pipeline uses sophisticated dependency management:
```yaml
dependencies:
  - success_handler  # Depends on conditional execution
  - failure_handler  # Depends on conditional execution
```

### Terminal Integration
Real terminal command execution with comprehensive result capture:
```yaml
tool: terminal
action: execute
parameters:
  command: "{{ operation }}"
# Captures: stdout, stderr, return_code, execution_time, success
```

### Template-Based Reporting
Advanced Jinja2 templating for dynamic report generation:
```yaml
content: |
  {% if check_result == 'success' %}
  {{ success_handler }}
  {% else %}
  {{ failure_handler }}
  {% endif %}
```

## Advanced Features

### Robust Error Detection
```yaml
prompt: |
  Analyze this execution output: {{ execute_operation }}
  
  Return EXACTLY one word - either "success" or "failure".
  Do not add any explanation or other text.
```

### Professional Output Standards
```yaml
# Success handler formatting
Format as a simple bulleted list with these exact items:
- Operation executed successfully
- Risk level: {{ assess_risk }}
- Command completed without errors

# Failure handler formatting  
Format as a simple bulleted list.
```

### Comprehensive Execution Metadata
The pipeline captures detailed execution information:
- Command executed
- Return code
- Success status
- Execution time in milliseconds
- Standard output
- Standard error output

### Security-Focused Design
- Input validation before execution
- Risk assessment for operation classification
- Additional safety checks for high-risk operations
- Comprehensive logging and audit trails

## Common Use Cases

- **DevOps Automation**: Automated deployment and system administration tasks
- **CI/CD Pipelines**: Build and deployment command execution with error handling
- **System Monitoring**: Automated system checks with comprehensive reporting
- **Development Workflows**: Safe execution of development and testing commands
- **Quality Assurance**: Automated testing command execution with result validation
- **Infrastructure Management**: Server maintenance and configuration tasks

## Best Practices Demonstrated

1. **Multi-Stage Validation**: Comprehensive input and risk validation
2. **Conditional Execution**: Dynamic flow based on risk assessment
3. **Real Integration**: Actual terminal command execution
4. **Comprehensive Error Handling**: Success/failure path management
5. **Professional Reporting**: Detailed execution reports with metadata
6. **Security-First Design**: Risk assessment and safety validation
7. **Flexible Configuration**: Parameterized retry limits and operations

## Troubleshooting

### Common Issues
- **Command Validation**: Some commands may be flagged as unsafe
- **Execution Permissions**: Ensure proper permissions for command execution
- **Path Dependencies**: Commands may require specific working directories
- **Environment Variables**: Some operations may need environment setup

### Performance Considerations
- **Long-Running Commands**: May exceed default timeouts
- **High-Risk Operations**: Additional validation steps increase processing time
- **Output Size**: Large command outputs may affect report generation
- **Model Selection**: AUTO selection optimizes for task-specific requirements

### Security Considerations
- **Command Injection**: Input validation helps prevent malicious commands
- **Privilege Escalation**: Risk assessment identifies potentially dangerous operations
- **Audit Trail**: Comprehensive logging for security monitoring
- **Safety Checks**: Additional validation for high-risk operations

## Related Examples
- [control_flow_conditional.md](control_flow_conditional.md) - Conditional processing patterns
- [terminal_automation.md](terminal_automation.md) - Terminal automation workflows
- [error_handling_examples.md](error_handling_examples.md) - Error handling strategies
- [simple_error_handling.md](simple_error_handling.md) - Basic error handling patterns

## Technical Requirements

- **Models**: Support for validation, risk assessment, and text analysis
- **Tools**: Terminal access for command execution
- **Permissions**: Appropriate system permissions for command execution
- **Security**: Safe command execution environment

This pipeline provides production-ready dynamic control flow patterns essential for building robust automation systems that can safely execute operations while providing comprehensive error handling and reporting capabilities.