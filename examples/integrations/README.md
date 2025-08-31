# Integration Examples

This directory contains examples that demonstrate integration with external services, APIs, and cloud platforms. These examples show how to build robust, production-ready pipelines that leverage external capabilities.

## Examples Overview

### 🔧 [mcp_tools.yaml](mcp_tools.yaml)
**Model Context Protocol Tools Integration**
- Comprehensive MCP tool usage
- Automatic tool detection and server management
- Multi-tool coordination workflows
- Project analysis with multiple tools

```bash
# Analyze a project with comprehensive scope
python scripts/execution/run_pipeline.py examples/integrations/mcp_tools.yaml \
  -i project_directory="examples/test_project" \
  -i analysis_scope="comprehensive" \
  -i auto_fix=false

# Security-focused analysis with auto-fix
python scripts/execution/run_pipeline.py examples/integrations/mcp_tools.yaml \
  -i project_directory="src/" \
  -i analysis_scope="security_focused" \
  -i auto_fix=true
```

### 🌐 [external_apis.yaml](external_apis.yaml)
**External API Integration**
- Multi-source data aggregation
- Authentication and rate limiting
- Error handling and retry logic
- Cross-platform data synthesis

```bash
# Research across multiple API sources
python scripts/execution/run_pipeline.py examples/integrations/external_apis.yaml \
  -i research_topic="artificial intelligence trends" \
  -i data_sources='["news", "academic", "social"]' \
  -i max_results_per_source=15

# Financial data integration
python scripts/execution/run_pipeline.py examples/integrations/external_apis.yaml \
  -i research_topic="market volatility" \
  -i data_sources='["news", "financial"]' \
  -i include_sentiment=true
```

### ☁️ [cloud_services.yaml](cloud_services.yaml)
**Cloud Services Integration**
- Multi-cloud platform support (AWS, Azure, GCP)
- Cloud AI service utilization
- Serverless function orchestration
- Cross-cloud performance comparison

```bash
# AWS cloud processing pipeline
python scripts/execution/run_pipeline.py examples/integrations/cloud_services.yaml \
  -i cloud_provider="aws" \
  -i operation_type="ai_pipeline" \
  -i data_file="examples/data/sample_data.json"

# Multi-cloud comparison analysis
python scripts/execution/run_pipeline.py examples/integrations/cloud_services.yaml \
  -i cloud_provider="multi_cloud" \
  -i operation_type="ai_pipeline" \
  -i data_file="examples/data/large_dataset.json"
```

## Integration Patterns Demonstrated

### 🔗 **MCP Tool Integration**
- **Automatic Detection**: Tools automatically detected from pipeline requirements
- **Server Management**: MCP servers started and managed automatically
- **Multi-Tool Coordination**: Complex workflows using multiple tools in sequence
- **Error Resilience**: Graceful handling when tools are unavailable
- **State Management**: Persistent storage of analysis results

### 🌍 **External API Patterns**
- **Authentication Handling**: Multiple auth methods (API keys, OAuth, tokens)
- **Rate Limiting**: Respect API rate limits with intelligent backoff
- **Data Normalization**: Standardize data from different API formats
- **Error Recovery**: Retry logic and fallback strategies
- **Response Validation**: Quality checks on API responses

### ☁️ **Cloud Service Patterns**
- **Multi-Cloud Support**: Single pipeline works across cloud providers
- **Service Abstraction**: Unified interface for similar services
- **Cost Optimization**: Intelligent service selection based on requirements
- **Hybrid Workflows**: Combine on-premises and cloud processing
- **Performance Monitoring**: Track and compare cloud service performance

## Advanced Integration Features

### 🛡️ **Security and Authentication**

**API Key Management:**
```yaml
# Secure API key usage
headers:
  Authorization: "Bearer {{ env.API_KEY }}"
  User-Agent: "Orchestrator Bot 1.0"
```

**Cloud Authentication:**
```yaml
# Cloud service authentication
tool: aws-s3
parameters:
  region: "{{ config.aws.region }}"
  # Uses AWS credentials from environment or IAM roles
```

**MCP Tool Security:**
```yaml
# Automatic tool availability checking
tool: filesystem
action: read
on_failure: skip  # Graceful degradation when tools unavailable
```

### ⚡ **Performance Optimization**

**Parallel API Calls:**
```yaml
foreach: "{{ api_endpoints }}"
parallel: true
max_concurrent: 3  # Respect rate limits
```

**Intelligent Caching:**
```yaml
# Store results for reuse
tool: memory
action: store
parameters:
  key: "cache_{{ query | hash }}"
  ttl: 3600  # 1 hour cache
```

**Conditional Processing:**
```yaml
# Skip expensive operations when not needed
condition: "{{ 'premium' in user_tier }}"
```

### 🔄 **Error Handling and Resilience**

**Retry Strategies:**
```yaml
retry: 3
backoff_strategy: "exponential"
max_backoff_time: 30
```

**Graceful Degradation:**
```yaml
on_failure: continue  # Keep processing other sources
fallback_action: use_cached_data
```

**Circuit Breaker Pattern:**
```yaml
# Automatically disable failing services
max_consecutive_failures: 5
disable_duration: 300  # 5 minutes
```

## Service-Specific Examples

### 📊 **Database Integration**
```yaml
# Multi-database support
tool: database
action: query
parameters:
  connection_string: "{{ env.DATABASE_URL }}"
  query: "SELECT * FROM research WHERE topic = ?"
  parameters: ["{{ topic }}"]
```

### 📈 **Analytics Integration**
```yaml
# Analytics service integration
tool: analytics
action: track_event
parameters:
  event: "pipeline_execution"
  properties:
    pipeline_id: "{{ pipeline.id }}"
    duration: "{{ execution_time }}"
```

### 📧 **Notification Integration**
```yaml
# Multi-channel notifications
tool: notifications
action: send
parameters:
  channels: ["email", "slack", "webhook"]
  message: "Pipeline completed: {{ pipeline.name }}"
  metadata: "{{ outputs }}"
```

## Requirements by Example

### MCP Tools Example
- **MCP Server**: Orchestrator MCP server running
- **Tool Availability**: filesystem, code-analyzer, git, memory, terminal, browser tools
- **Permissions**: File system read/write access
- **Dependencies**: Git repository (optional)

### External APIs Example
- **API Keys**: News API, Academic API, Social API, Financial API keys
- **Network Access**: Outbound HTTP/HTTPS requests
- **Rate Limits**: Respect for API quotas and limits
- **Data Storage**: Database for result persistence (optional)

### Cloud Services Example
- **Cloud Credentials**: AWS, Azure, and/or GCP authentication
- **Service Permissions**: Storage, compute, AI service access
- **Network Access**: Cloud service endpoints
- **Resource Quotas**: Sufficient cloud service limits

## Configuration Management

### Environment Variables
```bash
# API Keys
export NEWS_API_KEY="your_news_api_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"

# Cloud Credentials  
export AWS_ACCESS_KEY_ID="your_aws_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret"
export AZURE_CLIENT_ID="your_azure_client_id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcp-key.json"

# Database
export DATABASE_URL="postgresql://user:pass@localhost/db"
```

### Configuration Files
```yaml
# config/integrations.yaml
apis:
  rate_limits:
    default: 100  # requests per hour
    premium: 1000
  timeouts:
    default: 30   # seconds
    file_upload: 300

cloud:
  preferred_regions:
    aws: "us-east-1"
    azure: "East US"  
    gcp: "us-central1"
  cost_optimization: true
  auto_scaling: true
```

## Best Practices

### 🎯 **Design Principles**
- **Fail Fast**: Validate external dependencies early
- **Graceful Degradation**: Continue processing when non-critical services fail
- **Idempotency**: Ensure operations can be safely retried
- **Monitoring**: Track external service performance and reliability
- **Security**: Never expose credentials in pipeline definitions

### 🔧 **Implementation Guidelines**
- **Rate Limiting**: Respect external service limits
- **Caching**: Cache expensive API calls when appropriate
- **Timeout Handling**: Set reasonable timeouts for all external calls
- **Error Classification**: Distinguish between retryable and permanent errors
- **Documentation**: Document external service dependencies clearly

### 📊 **Monitoring and Observability**
- **Success Rates**: Track API call success/failure rates
- **Response Times**: Monitor external service performance
- **Cost Tracking**: Monitor cloud service usage and costs
- **Alert Thresholds**: Set alerts for service degradation
- **Dependency Health**: Monitor external service status

## Troubleshooting

### Common Integration Issues

**Authentication Failures:**
- Check API keys are correctly configured
- Verify cloud credentials have sufficient permissions
- Check for expired tokens or certificates

**Rate Limiting:**
- Reduce concurrent request limits
- Implement exponential backoff
- Consider upgrading API plans for higher limits

**Network Issues:**
- Check firewall and proxy settings
- Verify DNS resolution for service endpoints
- Test connectivity with simple curl commands

**Service Unavailability:**
- Implement circuit breaker patterns
- Use health check endpoints when available
- Have fallback strategies for critical dependencies

### Performance Optimization

**Slow API Responses:**
- Implement timeout handling
- Use parallel processing where appropriate
- Cache frequently accessed data
- Consider service geographic proximity

**High Costs:**
- Monitor cloud service usage
- Use spot instances and reserved capacity
- Implement auto-scaling policies
- Regular cost optimization reviews

## Security Considerations

### 🔒 **Authentication Security**
- Store credentials securely (environment variables, key vaults)
- Use least-privilege access principles
- Rotate credentials regularly
- Monitor for credential compromise

### 🛡️ **Data Security**
- Encrypt data in transit and at rest
- Implement proper access controls
- Log and monitor data access
- Follow data residency requirements

### 🚨 **Monitoring Security**
- Track unusual API usage patterns
- Monitor for credential stuffing attacks
- Set up alerts for security events
- Regular security audits of integrations

## Next Steps

After mastering integration examples, explore:
- **[Advanced Examples](../advanced/)** - Complex workflow patterns
- **[Migration Examples](../migration/)** - Legacy system integration
- **[Platform Examples](../platform/)** - Platform-specific considerations

## Contributing

When creating new integration examples:
- Include comprehensive error handling
- Document all external dependencies
- Provide configuration templates
- Add monitoring and observability features
- Include security best practices
- Test across different environments