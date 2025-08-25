# Production Runbooks - Wrapper Integration System

## Overview

This document provides comprehensive operational procedures for the production wrapper integration system deployed via Issue #247. These runbooks cover monitoring, troubleshooting, maintenance, and emergency procedures.

**System Overview:**
- **Wrapper Architecture**: Unified wrapper framework for RouteLLM, POML, and external tools
- **Monitoring**: Real-time dashboard at port 5000 with automated alerting
- **Deployment**: Blue-green deployment with automated rollback capabilities
- **Security**: Production-hardened with HTTPS, authentication, and security monitoring

---

## 1. System Monitoring

### 1.1 Monitoring Dashboard

**Access:** `https://localhost:5000` (or configured monitoring port)

**Key Metrics to Monitor:**
- System health score (target: >95%)
- Wrapper success rates (target: >95%)
- Response times (target: <5000ms)
- Error rates (target: <5%)
- Active operations count
- Resource utilization

**Dashboard Sections:**
1. **System Overview**: Overall health and status
2. **Wrapper Metrics**: Individual wrapper performance
3. **Performance Charts**: Time-series performance data
4. **Alerts**: Active alerts and notifications
5. **Logs**: Real-time log streaming

### 1.2 Health Check Procedures

**Automated Health Checks:**
- Interval: Every 30 seconds
- Timeout: 10 seconds
- Retry attempts: 3

**Manual Health Check:**
```bash
# Check system health via API
curl -X GET https://localhost:5000/api/system/health

# Expected response:
{
  "success": true,
  "data": {
    "overall_health_score": 0.95,
    "system_status": "Healthy",
    "active_wrappers": 3,
    "healthy_wrappers": 3
  }
}
```

**Health Check Endpoints:**
- System Health: `/api/system/health`
- Wrapper Health: `/api/wrappers/health`
- Individual Wrapper: `/api/wrappers/<name>/metrics`

### 1.3 Log Monitoring

**Log Locations:**
- Application logs: `/tmp/orchestrator_monitoring/logs/application.log`
- Access logs: `/tmp/orchestrator_monitoring/logs/access.log`
- Error logs: `/tmp/orchestrator_monitoring/logs/error.log`
- Monitoring logs: `/tmp/orchestrator_monitoring/logs/monitoring.log`

**Log Monitoring Commands:**
```bash
# Real-time log monitoring
tail -f /tmp/orchestrator_monitoring/logs/application.log

# Search for errors
grep "ERROR" /tmp/orchestrator_monitoring/logs/application.log

# Check wrapper-specific logs
grep "wrapper" /tmp/orchestrator_monitoring/logs/application.log
```

---

## 2. Alerting and Incident Response

### 2.1 Alert Thresholds

**Critical Alerts:**
- Success rate < 95%
- Error rate > 5%
- System health score < 50%
- Service unavailable

**Warning Alerts:**
- Response time > 5000ms
- Health score < 80%
- High resource usage
- Configuration drift

### 2.2 Incident Response Procedures

#### 2.2.1 Critical Service Outage

**Immediate Actions (0-5 minutes):**
1. Acknowledge the alert
2. Check monitoring dashboard for system status
3. Verify service accessibility
4. Check recent deployments or changes

**Investigation Steps (5-15 minutes):**
1. Review error logs for root cause
2. Check resource utilization (CPU, memory, disk)
3. Verify network connectivity
4. Check wrapper-specific health

**Resolution Actions:**
1. If deployment-related: Execute rollback
2. If resource-related: Scale resources or restart services
3. If configuration-related: Restore known-good configuration
4. If external dependency: Activate fallback mechanisms

#### 2.2.2 Performance Degradation

**Investigation Steps:**
1. Check response time metrics
2. Analyze success rate trends
3. Review resource utilization
4. Check for external API issues

**Resolution Actions:**
1. Restart affected wrapper services
2. Scale resources if needed
3. Enable rate limiting if overwhelmed
4. Switch to backup/fallback services

#### 2.2.3 Security Incident

**Immediate Actions:**
1. Isolate affected systems
2. Preserve logs and evidence
3. Change API keys and credentials
4. Notify security team

**Investigation Steps:**
1. Review access logs for suspicious activity
2. Check authentication failures
3. Analyze unusual traffic patterns
4. Verify configuration integrity

---

## 3. Operational Procedures

### 3.1 Deployment Management

#### 3.1.1 Production Deployment

**Pre-deployment Checklist:**
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Backup created
- [ ] Rollback plan ready
- [ ] Monitoring alerts configured

**Deployment Command:**
```bash
cd /Users/jmanning/orchestrator
python deployment/production_deployment.py --environment production
```

**Post-deployment Validation:**
- [ ] All services responding
- [ ] Health checks passing
- [ ] Monitoring dashboard active
- [ ] Performance within thresholds

#### 3.1.2 Blue-Green Deployment

**Manual Traffic Switch:**
```python
# Switch to green environment
from deployment.blue_green_deployment import BlueGreenDeployment
from deployment.production_deployment import DeploymentConfig

config = DeploymentConfig()
bg_deployment = BlueGreenDeployment(config)

# Switch traffic
result = await bg_deployment.switch_traffic_manual("green")
```

**Check Environment Status:**
```python
# Get current environment status
status = await bg_deployment.get_environments_status()
print(f"Active environment: {status['active_environment']}")
```

#### 3.1.3 Rollback Procedures

**Automatic Rollback:** System automatically triggers rollback on:
- Success rate < 95%
- Response time > 5000ms
- Error rate > 5%
- Health score < 50%

**Manual Rollback:**
```python
from deployment.rollback_procedures import RollbackManager

rollback_manager = RollbackManager(config)

# Rollback to stable state
result = await rollback_manager.rollback_to_stable()

# Or rollback to specific point
result = await rollback_manager.rollback_to_point("rollback-id")
```

### 3.2 Configuration Management

#### 3.2.1 Configuration Updates

**Safe Configuration Change Process:**
1. Create rollback point
2. Update configuration files
3. Validate configuration
4. Test changes in staging
5. Deploy to production
6. Monitor for issues

**Configuration Files:**
- Wrapper config: `src/orchestrator/core/wrapper_config.py`
- Monitoring config: `/tmp/orchestrator_monitoring/config/`
- Security config: `deployment/security_hardening.py`

#### 3.2.2 Secrets Management

**Best Practices:**
- Use environment variables for secrets
- Rotate API keys regularly
- Never commit secrets to code
- Use secure secret stores in production

**Secret Rotation Procedure:**
1. Generate new secret
2. Update secret store
3. Deploy updated configuration
4. Verify system functionality
5. Deactivate old secret

### 3.3 Maintenance Procedures

#### 3.3.1 Regular Maintenance Tasks

**Daily:**
- [ ] Check system health dashboard
- [ ] Review error logs
- [ ] Verify backup completion
- [ ] Check disk space usage

**Weekly:**
- [ ] Review performance trends
- [ ] Check security alerts
- [ ] Validate monitoring alerts
- [ ] Update documentation

**Monthly:**
- [ ] Rotate API keys and secrets
- [ ] Security vulnerability scan
- [ ] Performance optimization review
- [ ] Disaster recovery test

#### 3.3.2 System Updates

**Update Procedure:**
1. Schedule maintenance window
2. Create system backup
3. Update dependencies
4. Run security scan
5. Deploy to staging
6. Test thoroughly
7. Deploy to production
8. Monitor for issues

---

## 4. Troubleshooting Guide

### 4.1 Common Issues

#### 4.1.1 Wrapper Service Not Responding

**Symptoms:**
- HTTP 503 errors
- Timeout errors
- Health check failures

**Troubleshooting Steps:**
1. Check service status
2. Review error logs
3. Check resource utilization
4. Restart service if needed
5. Verify configuration

**Commands:**
```bash
# Check wrapper health
curl https://localhost:5000/api/wrappers/health

# Check specific wrapper
curl https://localhost:5000/api/wrappers/routellm/metrics

# Restart monitoring service (if needed)
python ops/monitoring_setup.py --restart dashboard
```

#### 4.1.2 High Response Times

**Symptoms:**
- Response times > 5000ms
- Performance degradation alerts
- User complaints

**Investigation:**
1. Check resource utilization
2. Analyze traffic patterns
3. Review external API latency
4. Check database performance

**Resolution:**
- Scale resources
- Enable caching
- Optimize queries
- Switch to faster APIs

#### 4.1.3 Authentication Failures

**Symptoms:**
- 401/403 errors
- Authentication alerts
- Access denied errors

**Investigation:**
1. Check API key validity
2. Verify authentication configuration
3. Review access logs
4. Check token expiration

**Resolution:**
- Refresh API keys
- Update authentication config
- Clear authentication cache
- Verify user permissions

### 4.2 Emergency Procedures

#### 4.2.1 System Recovery

**If monitoring dashboard is down:**
1. Check process status
2. Check port availability
3. Restart dashboard service
4. Verify configuration files
5. Check log files for errors

**If all services are down:**
1. Check system resources
2. Restart entire system
3. Restore from backup if needed
4. Verify network connectivity
5. Contact system administrator

#### 4.2.2 Data Recovery

**Backup Locations:**
- System backups: `/tmp/orchestrator_backup/`
- Rollback points: `/tmp/rollback_backups/`
- Configuration backups: `/tmp/orchestrator_monitoring/config/`

**Recovery Commands:**
```bash
# List available backups
ls /tmp/orchestrator_backup/

# Restore from backup
python deployment/rollback_procedures.py --restore backup-id

# Verify restore
python deployment/production_deployment.py --validate
```

---

## 5. Performance Optimization

### 5.1 Performance Monitoring

**Key Performance Indicators (KPIs):**
- Response time percentiles (P50, P95, P99)
- Throughput (requests per second)
- Error rate trends
- Resource utilization trends

**Performance Optimization Targets:**
- P95 response time: < 2000ms
- P99 response time: < 5000ms
- Success rate: > 99%
- CPU utilization: < 70%
- Memory utilization: < 80%

### 5.2 Optimization Techniques

**Response Time Optimization:**
- Implement response caching
- Optimize API calls
- Use connection pooling
- Enable compression

**Throughput Optimization:**
- Scale horizontally
- Optimize resource allocation
- Implement load balancing
- Use asynchronous processing

**Resource Optimization:**
- Monitor memory leaks
- Optimize database queries
- Clean up temporary files
- Manage connection pools

---

## 6. Security Operations

### 6.1 Security Monitoring

**Security Metrics:**
- Failed authentication attempts
- Unusual access patterns
- API rate limit violations
- Configuration changes

**Security Alerts:**
- Multiple failed login attempts
- Access from unusual locations
- Suspicious API usage
- Configuration drift

### 6.2 Security Procedures

#### 6.2.1 Security Incident Response

1. **Detect**: Monitor security alerts and logs
2. **Contain**: Isolate affected systems
3. **Analyze**: Investigate root cause
4. **Eradicate**: Remove threats
5. **Recover**: Restore normal operations
6. **Learn**: Update procedures

#### 6.2.2 Regular Security Tasks

**Daily:**
- Review security alerts
- Check failed authentication logs
- Monitor API usage patterns

**Weekly:**
- Review access permissions
- Check for configuration drift
- Update security policies

**Monthly:**
- Rotate API keys
- Security vulnerability scan
- Update dependencies
- Review security metrics

---

## 7. Contact Information

### 7.1 Escalation Matrix

**Level 1 - Operations Team:**
- Response time: 15 minutes
- Responsibilities: Monitoring, basic troubleshooting

**Level 2 - Engineering Team:**
- Response time: 1 hour
- Responsibilities: Code issues, complex troubleshooting

**Level 3 - Architecture Team:**
- Response time: 4 hours
- Responsibilities: System design issues, major incidents

### 7.2 Communication Channels

**Normal Operations:**
- Slack: #wrapper-operations
- Email: ops-team@company.com
- Ticketing: Jira

**Emergency:**
- Phone: Emergency hotline
- Slack: #incident-response
- Email: emergency@company.com

---

## 8. Appendix

### 8.1 Configuration Files

**Key Configuration Files:**
- `src/orchestrator/core/wrapper_config.py`: Wrapper system configuration
- `src/orchestrator/core/wrapper_monitoring.py`: Monitoring configuration  
- `deployment/production_deployment.py`: Deployment configuration
- `ops/monitoring_setup.py`: Monitoring setup configuration

### 8.2 Useful Commands

**System Status:**
```bash
# Check all wrapper health
curl https://localhost:5000/api/wrappers/health | jq

# Get system metrics
curl https://localhost:5000/api/system/health | jq

# Export metrics
curl https://localhost:5000/api/metrics/export | jq
```

**Log Analysis:**
```bash
# Count errors by type
grep "ERROR" application.log | cut -d' ' -f5 | sort | uniq -c

# Monitor real-time errors
tail -f application.log | grep "ERROR"

# Check wrapper-specific issues
grep -i "wrapper.*error" application.log
```

**Performance Analysis:**
```bash
# Response time analysis
grep "response_time" monitoring.log | awk '{print $NF}' | sort -n

# Success rate calculation
grep "success_rate" monitoring.log | tail -1

# Check active operations
curl https://localhost:5000/api/system/health | jq '.data.active_operations'
```

### 8.3 Emergency Contacts

- **System Administrator**: admin@company.com
- **Security Team**: security@company.com  
- **Development Team**: dev-team@company.com
- **Infrastructure Team**: infra@company.com

---

**Document Version:** 1.0  
**Last Updated:** 2025-08-25  
**Next Review:** 2025-09-25  
**Owner:** Operations Team