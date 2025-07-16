Deployment
===========

This guide covers deploying the Orchestrator in various environments, from local development to production-scale deployments with high availability and scalability.

Local Development Deployment
-----------------------------

For local development, you can run the Orchestrator with minimal configuration:

.. code-block:: bash

    # Install dependencies
    pip install -r requirements.txt
    
    # Set environment variables
    export ORCHESTRATOR_ENV=development
    export ORCHESTRATOR_LOG_LEVEL=DEBUG
    
    # Run with development settings
    python -m orchestrator.orchestrator --config config/development.yaml

Development Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # config/development.yaml
    environment: development
    debug: true
    
    storage:
      backend: "file"
      path: "./dev_data"
    
    models:
      default_provider: "openai"
      openai:
        api_key: "${OPENAI_API_KEY}"
    
    logging:
      level: "DEBUG"
      format: "detailed"
    
    cache:
      backend: "memory"
      ttl: 300

Docker Deployment
-----------------

Deploy the Orchestrator using Docker for consistent environments:

Dockerfile
^^^^^^^^^^

.. code-block:: dockerfile

    FROM python:3.11-slim
    
    WORKDIR /app
    
    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy requirements and install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy application code
    COPY src/ src/
    COPY config/ config/
    
    # Create non-root user
    RUN useradd -m -u 1000 orchestrator
    USER orchestrator
    
    # Set environment variables
    ENV ORCHESTRATOR_ENV=production
    ENV ORCHESTRATOR_CONFIG_PATH=/app/config/production.yaml
    
    # Expose ports
    EXPOSE 8000
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
        CMD python -m orchestrator.health_check
    
    # Start the application
    CMD ["python", "-m", "orchestrator.orchestrator"]

Docker Compose
^^^^^^^^^^^^^^

.. code-block:: yaml

    # docker-compose.yml
    version: '3.8'
    
    services:
      orchestrator:
        build: .
        ports:
          - "8000:8000"
        environment:
          - ORCHESTRATOR_ENV=production
          - REDIS_URL=redis://redis:6379
          - POSTGRES_URL=postgresql://postgres:password@postgres:5432/orchestrator
        depends_on:
          - redis
          - postgres
        volumes:
          - ./config:/app/config
          - ./data:/app/data
        networks:
          - orchestrator-network
    
      redis:
        image: redis:7-alpine
        ports:
          - "6379:6379"
        volumes:
          - redis-data:/data
        networks:
          - orchestrator-network
    
      postgres:
        image: postgres:15-alpine
        environment:
          - POSTGRES_DB=orchestrator
          - POSTGRES_USER=postgres
          - POSTGRES_PASSWORD=password
        volumes:
          - postgres-data:/var/lib/postgresql/data
        networks:
          - orchestrator-network
    
      nginx:
        image: nginx:alpine
        ports:
          - "80:80"
          - "443:443"
        volumes:
          - ./nginx.conf:/etc/nginx/nginx.conf
          - ./ssl:/etc/nginx/ssl
        depends_on:
          - orchestrator
        networks:
          - orchestrator-network
    
    volumes:
      redis-data:
      postgres-data:
    
    networks:
      orchestrator-network:
        driver: bridge

Production Deployment
---------------------

For production environments, consider these deployment strategies:

Kubernetes Deployment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # k8s/deployment.yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: orchestrator
      namespace: production
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: orchestrator
      template:
        metadata:
          labels:
            app: orchestrator
        spec:
          containers:
          - name: orchestrator
            image: orchestrator:latest
            ports:
            - containerPort: 8000
            env:
            - name: ORCHESTRATOR_ENV
              value: "production"
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: orchestrator-secrets
                  key: redis-url
            - name: POSTGRES_URL
              valueFrom:
                secretKeyRef:
                  name: orchestrator-secrets
                  key: postgres-url
            resources:
              requests:
                memory: "512Mi"
                cpu: "250m"
              limits:
                memory: "1Gi"
                cpu: "500m"
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 10
            readinessProbe:
              httpGet:
                path: /ready
                port: 8000
              initialDelaySeconds: 5
              periodSeconds: 5
          imagePullSecrets:
          - name: docker-registry-secret

Service Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # k8s/service.yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: orchestrator-service
      namespace: production
    spec:
      selector:
        app: orchestrator
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8000
      type: LoadBalancer

Ingress Configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # k8s/ingress.yaml
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: orchestrator-ingress
      namespace: production
      annotations:
        kubernetes.io/ingress.class: "nginx"
        cert-manager.io/cluster-issuer: "letsencrypt-prod"
        nginx.ingress.kubernetes.io/rate-limit: "100"
    spec:
      tls:
      - hosts:
        - orchestrator.example.com
        secretName: orchestrator-tls
      rules:
      - host: orchestrator.example.com
        http:
          paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: orchestrator-service
                port:
                  number: 80

Configuration Management
------------------------

Production Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # config/production.yaml
    environment: production
    debug: false
    
    server:
      host: "0.0.0.0"
      port: 8000
      workers: 4
    
    database:
      url: "${POSTGRES_URL}"
      pool_size: 20
      max_overflow: 30
    
    cache:
      backend: "redis"
      url: "${REDIS_URL}"
      ttl: 3600
    
    models:
      registry:
        health_check_interval: 300
        max_retries: 3
      providers:
        openai:
          api_key: "${OPENAI_API_KEY}"
          rate_limit: 100
        anthropic:
          api_key: "${ANTHROPIC_API_KEY}"
          rate_limit: 50
    
    logging:
      level: "INFO"
      format: "json"
      output: "file"
      file_path: "/var/log/orchestrator/app.log"
    
    security:
      api_key_required: true
      rate_limiting:
        enabled: true
        requests_per_minute: 1000
      cors:
        enabled: true
        allowed_origins: ["https://app.example.com"]
    
    monitoring:
      metrics:
        enabled: true
        endpoint: "/metrics"
      health_checks:
        enabled: true
        endpoint: "/health"
      tracing:
        enabled: true
        jaeger_endpoint: "http://jaeger:14268"

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Production environment variables
    export ORCHESTRATOR_ENV=production
    export ORCHESTRATOR_CONFIG_PATH=/app/config/production.yaml
    
    # Database
    export POSTGRES_URL=postgresql://user:password@localhost:5432/orchestrator
    
    # Cache
    export REDIS_URL=redis://localhost:6379
    
    # API Keys
    export OPENAI_API_KEY=your-openai-key
    export ANTHROPIC_API_KEY=your-anthropic-key
    
    # Security
    export ORCHESTRATOR_API_KEY=your-secure-api-key
    export ORCHESTRATOR_SECRET_KEY=your-secret-key
    
    # Monitoring
    export SENTRY_DSN=https://your-sentry-dsn
    export DATADOG_API_KEY=your-datadog-key

High Availability Setup
-----------------------

Load Balancer Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: nginx

    # nginx.conf
    upstream orchestrator_backend {
        server orchestrator-1:8000;
        server orchestrator-2:8000;
        server orchestrator-3:8000;
    }
    
    server {
        listen 80;
        server_name orchestrator.example.com;
        
        location / {
            proxy_pass http://orchestrator_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Health check
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
            proxy_connect_timeout 5s;
            proxy_send_timeout 10s;
            proxy_read_timeout 10s;
        }
        
        location /health {
            access_log off;
            return 200 "healthy\n";
        }
    }

Database High Availability
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # PostgreSQL HA with replication
    services:
      postgres-primary:
        image: postgres:15
        environment:
          POSTGRES_REPLICATION_MODE: master
          POSTGRES_REPLICATION_USER: replicator
          POSTGRES_REPLICATION_PASSWORD: replicator_password
        volumes:
          - postgres-primary-data:/var/lib/postgresql/data
      
      postgres-replica:
        image: postgres:15
        environment:
          POSTGRES_REPLICATION_MODE: slave
          POSTGRES_REPLICATION_USER: replicator
          POSTGRES_REPLICATION_PASSWORD: replicator_password
          POSTGRES_MASTER_SERVICE: postgres-primary
        depends_on:
          - postgres-primary

Monitoring and Observability
-----------------------------

Prometheus Metrics
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Add to your application
    from prometheus_client import Counter, Histogram, Gauge
    
    # Metrics
    REQUEST_COUNT = Counter('orchestrator_requests_total', 'Total requests')
    REQUEST_DURATION = Histogram('orchestrator_request_duration_seconds', 'Request duration')
    ACTIVE_PIPELINES = Gauge('orchestrator_active_pipelines', 'Active pipelines')
    
    # Export metrics endpoint
    @app.route('/metrics')
    def metrics():
        return Response(generate_latest(), mimetype='text/plain')

Grafana Dashboard
^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "dashboard": {
        "title": "Orchestrator Dashboard",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(orchestrator_requests_total[5m])",
                "legendFormat": "Requests/sec"
              }
            ]
          },
          {
            "title": "Response Time",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, orchestrator_request_duration_seconds_bucket)",
                "legendFormat": "95th percentile"
              }
            ]
          }
        ]
      }
    }

Health Checks
^^^^^^^^^^^^^

.. code-block:: python

    # src/orchestrator/health_check.py
    import asyncio
    from typing import Dict, Any
    
    class HealthChecker:
        """Comprehensive health checking system."""
        
        def __init__(self, components: Dict[str, Any]):
            self.components = components
        
        async def check_health(self) -> Dict[str, Any]:
            """Check health of all components."""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "components": {}
            }
            
            for name, component in self.components.items():
                try:
                    if hasattr(component, 'health_check'):
                        status = await component.health_check()
                        health_status["components"][name] = {
                            "status": "healthy" if status else "unhealthy"
                        }
                    else:
                        health_status["components"][name] = {
                            "status": "unknown"
                        }
                except Exception as e:
                    health_status["components"][name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status["status"] = "degraded"
            
            return health_status

Scaling Strategies
------------------

Horizontal Scaling
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # k8s/hpa.yaml
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: orchestrator-hpa
      namespace: production
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: orchestrator
      minReplicas: 3
      maxReplicas: 20
      metrics:
      - type: Resource
        resource:
          name: cpu
          target:
            type: Utilization
            averageUtilization: 70
      - type: Resource
        resource:
          name: memory
          target:
            type: Utilization
            averageUtilization: 80

Vertical Scaling
^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # k8s/vpa.yaml
    apiVersion: autoscaling.k8s.io/v1
    kind: VerticalPodAutoscaler
    metadata:
      name: orchestrator-vpa
      namespace: production
    spec:
      targetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: orchestrator
      updatePolicy:
        updateMode: "Auto"
      resourcePolicy:
        containerPolicies:
        - containerName: orchestrator
          maxAllowed:
            cpu: 2
            memory: 4Gi
          minAllowed:
            cpu: 100m
            memory: 256Mi

Security Considerations
-----------------------

TLS Configuration
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # TLS configuration
    server:
      tls:
        enabled: true
        cert_file: "/etc/ssl/certs/orchestrator.crt"
        key_file: "/etc/ssl/private/orchestrator.key"
        min_version: "1.2"
        cipher_suites:
          - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
          - "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"

API Authentication
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Authentication middleware
    from functools import wraps
    from flask import request, jsonify
    
    def require_api_key(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            if not api_key or not validate_api_key(api_key):
                return jsonify({'error': 'Invalid API key'}), 401
            return f(*args, **kwargs)
        return decorated_function

Secrets Management
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # k8s/secrets.yaml
    apiVersion: v1
    kind: Secret
    metadata:
      name: orchestrator-secrets
      namespace: production
    type: Opaque
    data:
      redis-url: <base64-encoded-redis-url>
      postgres-url: <base64-encoded-postgres-url>
      openai-api-key: <base64-encoded-openai-key>
      anthropic-api-key: <base64-encoded-anthropic-key>

Backup and Recovery
-------------------

Database Backup
^^^^^^^^^^^^^^^

.. code-block:: bash

    #!/bin/bash
    # backup.sh
    
    BACKUP_DIR="/backups/orchestrator"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    # Create backup directory
    mkdir -p $BACKUP_DIR
    
    # Backup PostgreSQL
    pg_dump $POSTGRES_URL > $BACKUP_DIR/postgres_backup_$TIMESTAMP.sql
    
    # Backup Redis
    redis-cli --rdb $BACKUP_DIR/redis_backup_$TIMESTAMP.rdb
    
    # Compress backups
    gzip $BACKUP_DIR/*_$TIMESTAMP.*
    
    # Clean old backups (keep last 7 days)
    find $BACKUP_DIR -name "*.gz" -mtime +7 -delete

State Recovery
^^^^^^^^^^^^^^

.. code-block:: python

    # Recovery script
    import asyncio
    from orchestrator.state.state_manager import StateManager
    
    async def recover_pipeline(pipeline_id: str, checkpoint_path: str):
        """Recover pipeline from checkpoint."""
        state_manager = StateManager()
        
        # Load checkpoint
        checkpoint = await state_manager.load_checkpoint(checkpoint_path)
        
        # Restore pipeline state
        pipeline = await state_manager.restore_pipeline(pipeline_id, checkpoint)
        
        # Resume execution
        await pipeline.resume()
        
        print(f"Pipeline {pipeline_id} recovered successfully")

Deployment Checklist
---------------------

Pre-deployment
^^^^^^^^^^^^^^

- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Load balancer configured
- [ ] Monitoring dashboards set up
- [ ] Backup procedures tested

Post-deployment
^^^^^^^^^^^^^^^

- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being generated
- [ ] Performance benchmarks met
- [ ] Security scans completed
- [ ] Documentation updated

Troubleshooting Common Issues
-----------------------------

Connection Issues
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Check service connectivity
    kubectl get pods -n production
    kubectl logs orchestrator-pod-name -n production
    
    # Test database connection
    kubectl exec -it orchestrator-pod-name -- python -c "
    from orchestrator.database import test_connection
    test_connection()
    "

Performance Issues
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Check resource usage
    kubectl top pods -n production
    kubectl describe pod orchestrator-pod-name -n production
    
    # Check metrics
    curl http://orchestrator.example.com/metrics

This comprehensive deployment guide covers everything from local development to production-scale deployments with high availability, monitoring, and security considerations.
