# Platform Examples

This directory contains examples that demonstrate platform-specific optimizations and cross-platform compatibility patterns. These examples help you deploy and optimize pipelines across different operating systems, environments, and deployment targets.

## Examples Overview

### 🌐 [cross_platform_compatibility.yaml](cross_platform_compatibility.yaml)
**Cross-Platform Compatibility Testing**
- Automatic platform detection and adaptation
- Platform-specific command and tool selection
- File system and path handling across platforms
- Performance and security analysis by platform

```bash
# Test compatibility across all platforms
python scripts/execution/run_pipeline.py examples/platform/cross_platform_compatibility.yaml \
  -i target_platforms='["macos", "linux", "windows"]' \
  -i test_mode="comprehensive"

# Quick basic compatibility test
python scripts/execution/run_pipeline.py examples/platform/cross_platform_compatibility.yaml \
  -i test_mode="basic"

# Stress test for production readiness
python scripts/execution/run_pipeline.py examples/platform/cross_platform_compatibility.yaml \
  -i test_mode="stress_test"
```

### 🚀 [deployment_environments.yaml](deployment_environments.yaml)
**Environment-Specific Deployment Optimization**
- Development, CI/CD, container, cloud, edge, and production optimizations
- Resource allocation and performance tuning
- Environment-specific security and monitoring configuration
- Comprehensive deployment guides generation

```bash
# Development environment optimization
python scripts/execution/run_pipeline.py examples/platform/deployment_environments.yaml \
  -i environment_type="development" \
  -i optimization_focus="speed"

# Production deployment configuration
python scripts/execution/run_pipeline.py examples/platform/deployment_environments.yaml \
  -i environment_type="production" \
  -i resource_constraints='{"memory_limit":"16GB","cpu_limit":"8","timeout":"1800"}' \
  -i optimization_focus="reliability"

# Edge computing optimization
python scripts/execution/run_pipeline.py examples/platform/deployment_environments.yaml \
  -i environment_type="edge" \
  -i resource_constraints='{"memory_limit":"2GB","cpu_limit":"2","timeout":"60"}' \
  -i optimization_focus="memory"
```

## Platform Optimization Patterns

### 🎯 **Cross-Platform Compatibility**

#### Platform Detection
```yaml
# Automatic platform detection
- id: detect_platform
  action: generate_text
  parameters:
    prompt: "Detect current platform and provide system information"
    model: <AUTO task="system_analysis">Platform detection model</AUTO>
```

#### Conditional Platform Logic
```yaml
# Platform-specific commands
command: |
  {% if platform == 'windows' %}
  dir /s /b *.log
  {% else %}
  find . -name "*.log" -type f
  {% endif %}
shell: "{{ 'cmd' if platform == 'windows' else 'bash' }}"
```

#### Cross-Platform File Handling
```yaml
# Use forward slashes - work everywhere
path: "output/results/{{ filename }}"
create_directories: true  # Handle directory creation across platforms
```

### 🏗️ **Environment-Specific Optimization**

#### Resource Constraints
```yaml
# Adapt based on environment
max_tokens: >-
  {%- if environment_type == 'edge' -%}200
  {%- elif environment_type == 'development' -%}500
  {%- else -%}1000
  {%- endif %}
```

#### Concurrent Processing
```yaml
# Environment-appropriate concurrency
parallel: true
max_concurrent: >-
  {%- if environment_type == 'edge' -%}1
  {%- elif environment_type == 'development' -%}2
  {%- elif environment_type == 'production' -%}8
  {%- else -%}4
  {%- endif %}
```

#### Model Selection by Environment
```yaml
# Environment-optimized models
model: >-
  {%- if environment_type == 'edge' -%}<AUTO quality="fast">Edge-optimized model</AUTO>
  {%- elif environment_type == 'production' -%}<AUTO quality="premium">Production-quality model</AUTO>
  {%- else -%}<AUTO>Balanced model selection</AUTO>
  {%- endif %}
```

### 🔧 **Deployment Configuration Patterns**

#### Container Optimization
```yaml
# Container-specific settings
condition: "{{ environment_type == 'container' }}"
parameters:
  health_check_enabled: true
  graceful_shutdown_timeout: 30
  resource_limits_enforced: true
```

#### Cloud Optimization
```yaml
# Cloud-specific settings
condition: "{{ environment_type == 'cloud' }}"
parameters:
  auto_scaling_enabled: true
  multi_region_deployment: true
  serverless_preferred: true
```

#### CI/CD Integration
```yaml
# CI/CD-specific settings
condition: "{{ environment_type == 'ci_cd' }}"
parameters:
  parallel_test_execution: true
  artifact_caching: true
  fast_fail_enabled: true
```

## Platform Support Matrix

### 🖥️ **Operating Systems**

| Feature | macOS | Linux | Windows | Notes |
|---------|-------|-------|---------|--------|
| **File Operations** | ✅ | ✅ | ✅ | Universal support |
| **Shell Commands** | ✅ (bash/zsh) | ✅ (bash/sh) | ✅ (cmd/PowerShell) | Auto-detection |
| **Path Handling** | ✅ | ✅ | ✅ | Forward slash normalization |
| **Environment Variables** | ✅ | ✅ | ✅ | Cross-platform patterns |
| **Package Management** | Homebrew | apt/yum/pacman | Chocolatey/winget | Auto-detection |
| **Python Integration** | ✅ | ✅ | ✅ | Universal Python support |

### 🏢 **Deployment Environments**

| Environment | Optimization Focus | Resource Profile | Use Cases |
|-------------|-------------------|------------------|-----------|
| **Development** | Speed, debugging | High resources | Local development, testing |
| **CI/CD** | Reliability, speed | Limited time | Automated testing, builds |
| **Container** | Efficiency, portability | Resource limits | Kubernetes, Docker |
| **Cloud** | Scalability, cost | Elastic resources | AWS, Azure, GCP |
| **Edge** | Low latency, efficiency | Constrained resources | IoT, embedded systems |
| **Production** | Reliability, performance | High availability | Mission-critical workloads |

### 🏗️ **Architecture Support**

| Architecture | Support Level | Notes |
|--------------|--------------|--------|
| **x86_64** | Full | Primary architecture |
| **ARM64** | Full | Apple Silicon, ARM servers |
| **ARM32** | Limited | Edge computing, IoT |

## Performance Optimization Strategies

### ⚡ **Speed Optimization**

**Development Environment:**
- Local model preferences
- Aggressive caching
- Parallel processing where safe
- Skip non-essential validations

**CI/CD Environment:**
- Lightweight model selection
- Parallel test execution
- Cached dependencies
- Fast failure detection

### 💾 **Memory Optimization**

**Edge Computing:**
- Minimal model footprints
- Streaming processing patterns
- Local storage optimization
- Memory-efficient algorithms

**Container Deployment:**
- Resource limit compliance
- Memory leak prevention
- Garbage collection tuning
- Shared resource utilization

### 🛡️ **Reliability Optimization**

**Production Environment:**
- Comprehensive error handling
- Circuit breaker patterns
- Health check implementation
- Graceful degradation
- Automated recovery

### 💰 **Cost Optimization**

**Cloud Deployment:**
- Right-sizing resources
- Spot instance utilization
- Auto-scaling policies
- Serverless function optimization
- Data transfer minimization

## Security Considerations

### 🔐 **Platform-Specific Security**

#### File System Security
```yaml
# Platform-appropriate permissions
parameters:
  file_permissions: >-
    {%- if platform == 'windows' -%}
    # Windows ACL handling
    {%- else -%}
    "644"  # Unix permissions
    {%- endif %}
```

#### Process Security
```yaml
# Secure process execution
parameters:
  run_as_user: "{{ 'orchestrator' if platform != 'windows' else null }}"
  sandbox_enabled: true
  resource_limits_enforced: true
```

#### Network Security
```yaml
# Environment-specific network security
parameters:
  tls_required: "{{ environment_type in ['production', 'cloud'] }}"
  certificate_validation: strict
  firewall_rules: environment_specific
```

## Monitoring and Observability

### 📊 **Performance Metrics**

**Cross-Platform Metrics:**
- Execution time by platform
- Resource utilization patterns
- Error rates by environment
- Platform-specific bottlenecks

**Environment-Specific Metrics:**
- Container resource usage
- Cloud cost optimization
- Edge device performance
- CI/CD pipeline efficiency

### 🚨 **Alerting Strategies**

**Development:**
- Debug-focused alerts
- Performance degradation warnings
- Resource exhaustion notifications

**Production:**
- SLA breach alerts
- Security event notifications
- Auto-recovery status updates
- Capacity planning alerts

## Best Practices

### 🎨 **Design Principles**

1. **Platform Agnostic Design**
   - Use abstracted operations
   - Avoid platform-specific hardcoding
   - Implement feature detection
   - Provide graceful fallbacks

2. **Environment Awareness**
   - Detect deployment environment
   - Adapt resource usage accordingly
   - Optimize for environment constraints
   - Implement environment-specific features

3. **Resource Efficiency**
   - Monitor resource usage
   - Implement appropriate limits
   - Use efficient algorithms
   - Optimize for target environment

### 🔧 **Implementation Guidelines**

1. **Configuration Management**
   - Use environment variables
   - Implement configuration layering
   - Provide sensible defaults
   - Support runtime configuration

2. **Error Handling**
   - Implement platform-specific error handling
   - Provide meaningful error messages
   - Support graceful degradation
   - Include recovery mechanisms

3. **Testing Strategy**
   - Test on target platforms
   - Validate environment configurations
   - Benchmark performance characteristics
   - Monitor production behavior

## Troubleshooting

### 🔍 **Common Platform Issues**

**Path and File System Issues:**
```bash
# Check path handling
python -c "import os; print('Platform:', os.name, 'Path sep:', os.sep)"

# Test file operations
python scripts/platform/test_file_operations.py
```

**Permission Problems:**
```bash
# Unix systems
chmod +x scripts/setup.sh
sudo chown -R orchestrator:orchestrator /opt/orchestrator

# Windows (run as administrator)
icacls "C:\orchestrator" /grant orchestrator:(OI)(CI)F
```

**Environment Detection:**
```bash
# Check environment variables
python scripts/platform/check_environment.py

# Validate platform detection
python scripts/platform/detect_platform.py
```

### 🚨 **Performance Issues**

**Memory Constraints:**
- Reduce model size for constrained environments
- Implement memory monitoring
- Use streaming processing for large data
- Configure garbage collection

**CPU Limitations:**
- Adjust concurrency limits
- Use efficient algorithms
- Profile CPU usage patterns
- Implement load balancing

**Network Issues:**
- Test connectivity to external services
- Implement retry mechanisms
- Use local caching where appropriate
- Monitor network performance

## Migration and Scaling

### 📈 **Scaling Strategies**

**Horizontal Scaling:**
- Container orchestration (Kubernetes)
- Cloud auto-scaling groups
- Load balancer configuration
- Distributed processing patterns

**Vertical Scaling:**
- Resource limit adjustments
- Performance tuning
- Memory optimization
- CPU efficiency improvements

### 🔄 **Environment Migration**

**Development to Production:**
1. Performance testing and validation
2. Security configuration review
3. Monitoring setup and validation
4. Resource allocation optimization
5. Rollback plan preparation

**On-Premises to Cloud:**
1. Dependency mapping and validation
2. Network configuration updates
3. Storage migration planning
4. Security policy adaptation
5. Cost optimization analysis

## Next Steps

After mastering platform examples:
- **[Basic Examples](../basic/)** - Apply platform knowledge to simple patterns
- **[Advanced Examples](../advanced/)** - Implement sophisticated cross-platform workflows
- **[Integration Examples](../integrations/)** - Connect platform-optimized pipelines with external services
- **[Migration Examples](../migration/)** - Plan platform-specific migration strategies