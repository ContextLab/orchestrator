# Terminal Automation Pipeline

**Pipeline**: `examples/terminal_automation.yaml`  
**Category**: System Operations  
**Complexity**: Intermediate  
**Key Features**: Terminal command execution, System information gathering, Report generation, Output capture

## Overview

The Terminal Automation Pipeline demonstrates system-level command execution using the terminal tool. It gathers system information, checks Python environment details, monitors disk usage, and generates comprehensive system reports, showcasing automated system administration and monitoring capabilities.

## Key Features Demonstrated

### 1. Terminal Command Execution
```yaml
- id: check_python
  tool: terminal
  action: execute
  parameters:
    command: "python --version"
    capture_output: true
```

### 2. Pipe Command Usage
```yaml
- id: check_packages
  tool: terminal
  parameters:
    command: "pip list | grep -E '(numpy|pandas|matplotlib)'"
```

### 3. System Information Gathering
```yaml
- id: system_info
  tool: terminal
  parameters:
    command: "uname -a"
```

### 4. Output Templating
```yaml
content: |
  ## Python Environment
  ```
  {{ check_python.stdout }}
  ```
```

## Pipeline Architecture

### Input Parameters
None (uses system commands that work across platforms)

### Processing Flow

1. **Check Python** - Verify Python installation and version
2. **Check Packages** - List installed data science packages
3. **System Info** - Gather operating system information
4. **Disk Usage** - Monitor disk space utilization
5. **Create Report** - Generate comprehensive system report

### Command Categories

#### Python Environment
```yaml
command: "python --version"     # Check Python version
command: "pip list | grep -E '(numpy|pandas|matplotlib)'"  # Check packages
```

#### System Information
```yaml
command: "uname -a"             # Operating system details
command: "df -h | head -5"      # Disk usage summary
```

## Usage Examples

### Basic System Check
```bash
python scripts/run_pipeline.py examples/terminal_automation.yaml
```

### Custom Output Location
```bash
python scripts/run_pipeline.py examples/terminal_automation.yaml \
  -o examples/outputs/my_system_check
```

### System Monitoring
```bash
# Run periodically for monitoring
python scripts/run_pipeline.py examples/terminal_automation.yaml \
  -o system_reports/$(date +%Y%m%d_%H%M%S)
```

## Terminal Commands Detailed

### Python Version Check
```yaml
command: "python --version"
# Expected output: "Python 3.9.7" or similar
# Captures: Python version information
# Purpose: Validate Python environment
```

### Package Installation Check
```yaml
command: "pip list | grep -E '(numpy|pandas|matplotlib)'"
# Expected output: Package list with versions
# Example:
# numpy        1.21.0
# pandas       1.3.0
# matplotlib   3.4.2
```

### System Information
```yaml
command: "uname -a"  
# Expected output: Complete system information
# Example: "Darwin MacBook-Pro.local 21.5.0 Darwin Kernel..."
# Captures: OS, kernel, architecture details
```

### Disk Usage Report
```yaml
command: "df -h | head -5"
# Expected output: Filesystem usage in human-readable format
# Example:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/disk1s1   466Gi  350Gi  114Gi  76% /
```

## Sample Output Report

### Generated System Report
```markdown
# System Information Report

## Python Environment
```
Python 3.9.7
```

## Installed Packages
```
numpy        1.21.0
pandas       1.3.0
matplotlib   3.4.2
```

## System Details
```
Darwin MacBook-Pro.local 21.5.0 Darwin Kernel Version 21.5.0: 
Tue Apr 26 21:08:22 PDT 2022; root:xnu-8020.121.3~4/RELEASE_X86_64 x86_64
```

## Disk Usage
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/disk1s1   466Gi  350Gi  114Gi  76% /
/dev/disk1s4   466Gi  8.0Gi  114Gi   7% /private/var/vm
```

Generated on: 2024-08-23T10:30:00Z
```

## Terminal Tool Features

### Output Capture
```yaml
capture_output: true
# Captures both stdout and stderr
# Available as: step_id.stdout and step_id.stderr
```

### Command Execution
```yaml
action: execute
# Executes shell commands in system terminal
# Supports complex commands with pipes and redirects
```

### Cross-Platform Commands
```yaml
# Works on Unix-like systems (Linux, macOS)
command: "uname -a"
command: "df -h"
command: "python --version"
```

## Advanced Command Patterns

### Complex Pipe Commands
```yaml
command: "ps aux | grep python | wc -l"
# Count running Python processes

command: "ls -la /tmp | grep -v '^d' | wc -l"  
# Count files (not directories) in /tmp
```

### Conditional Commands
```yaml
command: "which python3 && python3 --version || echo 'Python3 not found'"
# Check for python3, show version if found, error message if not
```

### System Monitoring
```yaml
command: "top -l 1 | grep -E '(CPU|PhysMem)'"
# Get current CPU and memory usage (macOS)

command: "free -h && uptime"
# Memory usage and system uptime (Linux)
```

## Error Handling and Validation

### Command Failure Handling
```yaml
# Commands may fail, check return codes
{{ check_packages.stdout | default('No data science packages found') }}
```

### Output Validation
```yaml
# Handle missing or empty output gracefully
stdout: "{{ step_result.stdout | default('Command output not available') }}"
```

### Cross-Platform Considerations
```yaml
# Different commands for different operating systems
# macOS: "df -h"
# Linux: "df -h"  
# Windows: "dir" (would need different pipeline)
```

## System Administration Applications

### Environment Validation
```yaml
# Check development environment setup
commands:
  - "python --version"
  - "pip --version"
  - "git --version"
  - "node --version"
```

### Security Auditing
```yaml
# Basic security checks
commands:
  - "last | head -10"        # Recent logins
  - "ps aux | grep ssh"      # SSH processes
  - "netstat -an | grep LISTEN"  # Listening ports
```

### Performance Monitoring
```yaml
# System performance metrics
commands:
  - "uptime"                 # Load average
  - "free -h"               # Memory usage
  - "df -h"                 # Disk usage
  - "iostat 1 1"            # I/O statistics
```

### Service Management
```yaml
# Service status checks
commands:
  - "systemctl status nginx"
  - "ps aux | grep docker"
  - "curl -I http://localhost:8000"
```

## Best Practices Demonstrated

1. **Output Capture**: Always capture command output for processing
2. **Error Handling**: Use default filters for missing output
3. **Command Chaining**: Use pipes for complex data processing
4. **Documentation**: Clear command purposes and expected outputs
5. **Cross-Platform Awareness**: Use universally available commands
6. **Report Generation**: Structure output in readable format

## Security Considerations

### Command Safety
- Avoid commands that modify system state without explicit intent
- Validate command inputs to prevent injection attacks
- Use read-only commands for information gathering
- Implement proper output sanitization

### Privilege Management
- Run with minimal required privileges
- Avoid sudo commands unless absolutely necessary
- Consider containerized execution for isolation
- Monitor and log command execution

## Performance Optimization

### Command Efficiency
```yaml
# Use efficient commands
command: "df -h | head -5"      # Limit output with head
command: "pip list | grep numpy" # Filter with grep
```

### Parallel Execution
```yaml
# Commands without dependencies can run in parallel
- id: check_python    # Can run independently
- id: system_info     # Can run independently  
- id: disk_usage      # Can run independently
```

## Troubleshooting

### Command Not Found
- Verify commands are available on target system
- Check PATH environment variable
- Use full command paths when necessary

### Permission Issues
- Ensure appropriate file/directory permissions
- Consider running with elevated privileges if needed
- Check user access rights for commands

### Output Parsing Problems
- Validate command output format
- Handle empty or unexpected output
- Use appropriate filters and defaults

## Related Examples
- [simple_timeout_test.md](simple_timeout_test.md) - Command execution with timeouts
- [validation_pipeline.md](validation_pipeline.md) - System validation patterns
- [simple_error_handling.md](simple_error_handling.md) - Error handling for system operations

## Technical Requirements

- **Terminal Access**: Shell/terminal execution capabilities
- **Operating System**: Unix-like system (Linux, macOS) for shown commands
- **File System**: Write access for report generation
- **Command Tools**: Standard system utilities (python, pip, uname, df)
- **Template Engine**: Output templating support

This pipeline provides a foundation for system automation, monitoring, and administration tasks while demonstrating safe and effective terminal command execution patterns.