Code Analysis Suite
===================

This example demonstrates how to build a comprehensive code analysis pipeline using the Orchestrator's declarative YAML framework. The suite performs automated code review, identifies security vulnerabilities, suggests improvements, and generates detailed reports - all defined in pure YAML with no custom Python code required.

.. note::
   **Level:** Advanced  
   **Duration:** 60-90 minutes  
   **Prerequisites:** Orchestrator framework installed, code repository to analyze

Overview
--------

The Code Analysis Suite performs:

1. **Code Discovery**: Find and catalog all source files
2. **Static Analysis**: Syntax, linting, and complexity checks
3. **Security Scanning**: Identify vulnerabilities and risks
4. **AI Code Review**: Intelligent suggestions and improvements
5. **Documentation Analysis**: Coverage and quality assessment
6. **Performance Profiling**: Identify bottlenecks
7. **Test Coverage**: Analyze testing completeness
8. **Architecture Review**: Design and structure analysis
9. **Report Generation**: Comprehensive insights and metrics

**Key Features Demonstrated:**
- Declarative YAML pipeline definition
- AUTO tag resolution for natural language task descriptions
- Multi-language support (Python, JavaScript, TypeScript, Go, etc.)
- Parallel analysis for large codebases
- Security vulnerability detection
- AI-powered code review
- CI/CD integration artifacts
- No Python code required

Quick Start
-----------

.. code-block:: bash

   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Run code analysis on a repository
   orchestrator run examples/code_analysis_suite.yaml \
     --input repo_path="/path/to/your/code" \
     --input languages='["python", "javascript"]' \
     --input analysis_depth="comprehensive"

Complete YAML Pipeline
----------------------

The complete pipeline is defined in ``examples/code_analysis_suite.yaml``. Here are the key sections:

**Pipeline Structure:**

.. code-block:: yaml

   name: "Code Analysis Suite"
   description: "Comprehensive code analysis with security scanning and AI review"

   inputs:
     repo_path:
       type: string
       description: "Path to repository to analyze"
       required: true
     
     languages:
       type: list
       description: "Programming languages to analyze"
       default: ["python", "javascript", "typescript"]
     
     analysis_depth:
       type: string
       description: "Analysis depth (quick, standard, comprehensive)"
       default: "comprehensive"

**Key Pipeline Steps:**

1. **Code Discovery:**

.. code-block:: yaml

   - id: discover_code
     action: <AUTO>discover all code files in {{repo_path}}:
       1. Find files matching patterns for languages: {{languages}}
       2. Exclude directories: .git, node_modules, venv
       3. Count lines of code per file
       4. Identify test files vs source files
       5. Find documentation files</AUTO>

2. **Security Scanning:**

.. code-block:: yaml

   - id: security_scan
     action: <AUTO>scan for security vulnerabilities:
       1. Check for hardcoded secrets/credentials
       2. Identify SQL injection risks
       3. Find XSS vulnerabilities
       4. Detect insecure dependencies
       5. Check for command injection risks
       6. Identify authentication/authorization issues</AUTO>

3. **AI-Powered Review:**

.. code-block:: yaml

   - id: ai_code_review
     action: <AUTO>review code quality using AI analysis:
       1. Code structure and organization
       2. Design patterns and best practices
       3. Variable/function naming conventions
       4. Error handling completeness
       5. Performance optimization opportunities
       6. Maintainability assessment
       7. Suggested refactoring improvements</AUTO>
     loop:
       foreach: "{{discover_code.result.file_list}}"
       parallel: true

How It Works
------------

**1. Intelligent Analysis**

The framework automatically:
- Detects programming languages and applies appropriate tools
- Runs security scans based on language-specific vulnerabilities
- Performs AI review focusing on code quality and best practices
- Generates insights tailored to your codebase

**2. Parallel Processing**

For efficiency:
- Files analyzed in parallel
- Independent checks run simultaneously
- Results aggregated intelligently
- Scales to large codebases

**3. Comprehensive Reporting**

The suite generates:
- Executive summary with key metrics
- Prioritized list of issues
- Actionable recommendations
- CI/CD integration artifacts
- Trend analysis (when historical data available)

Running the Pipeline
--------------------

**Using the CLI:**

.. code-block:: bash

   # Basic analysis
   orchestrator run code_analysis_suite.yaml \
     --input repo_path="./my-project"

   # Comprehensive analysis with all features
   orchestrator run code_analysis_suite.yaml \
     --input repo_path="./my-project" \
     --input languages='["python", "javascript", "go"]' \
     --input analysis_depth="comprehensive" \
     --input security_scan=true \
     --input performance_check=true

   # Quick analysis for CI/CD
   orchestrator run code_analysis_suite.yaml \
     --input repo_path="." \
     --input analysis_depth="quick" \
     --input severity_threshold="high"

**Using Python SDK:**

.. code-block:: python

   from orchestrator import Orchestrator
   
   # Initialize orchestrator
   orchestrator = Orchestrator()
   
   # Run code analysis
   result = await orchestrator.run_pipeline(
       "code_analysis_suite.yaml",
       inputs={
           "repo_path": "/path/to/project",
           "languages": ["python", "typescript"],
           "analysis_depth": "comprehensive",
           "security_scan": True
       }
   )
   
   # Access results
   print(f"Quality Score: {result['outputs']['quality_score']}/100")
   print(f"Security Score: {result['outputs']['security_score']}/100")
   print(f"Critical Issues: {result['outputs']['critical_issues']}")

Example Output
--------------

**Console Output:**

.. code-block:: text

   🔍 Code Analysis Suite
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ discover_code: Found 156 files (45,231 lines) (3.2s)
   ⟳ static_analysis: Analyzing 156 files in parallel...
     ✓ 156/156 files analyzed (12.4s)
   ✓ security_scan: Found 3 critical, 7 high vulnerabilities (8.7s)
   ⟳ ai_code_review: AI reviewing code quality...
     ✓ 50/50 files reviewed (max limit) (32.1s)
   ✓ documentation_check: Coverage: 72% (4.3s)
   ✓ performance_analysis: Found 5 bottlenecks (6.8s)
   ✓ dependency_check: 3 outdated, 1 vulnerable dependency (2.1s)
   ✓ test_coverage: Coverage: 67% (8.4s)
   ✓ architecture_review: Good separation, 2 circular deps (5.2s)
   ✓ generate_insights: Created prioritized action plan (3.1s)
   ✓ generate_report: Report generated (4.7s)
   ✓ generate_artifacts: CI/CD artifacts created (1.2s)
   
   ✅ Analysis completed in 92.4s
   
   📊 RESULTS SUMMARY
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   🎯 Quality Score: 78/100
   🔐 Security Score: 65/100
   📚 Documentation: 72%
   🧪 Test Coverage: 67%
   
   🚨 Critical Issues: 3
   ⚠️  Total Issues: 142

**Generated Report Example:**

.. code-block:: markdown

   # Code Analysis Report - MyProject
   
   ## Executive Summary
   
   **Overall Health**: 🟡 Moderate (Score: 72/100)
   
   ### Key Metrics
   - **Code Quality**: 78/100
   - **Security**: 65/100 ⚠️
   - **Documentation**: 72%
   - **Test Coverage**: 67%
   - **Technical Debt**: 23 hours estimated
   
   ## Critical Issues (Immediate Action Required)
   
   ### 🔐 Security Vulnerabilities
   
   1. **SQL Injection Risk** - `database/queries.py:45`
      ```python
      query = f"SELECT * FROM users WHERE id = {user_id}"
      ```
      **Fix**: Use parameterized queries
   
   2. **Hardcoded API Key** - `config/settings.py:12`
      ```python
      API_KEY = "sk-1234567890abcdef"
      ```
      **Fix**: Use environment variables
   
   3. **Command Injection** - `utils/shell.py:23`
      ```python
      os.system(f"echo {user_input}")
      ```
      **Fix**: Use subprocess with shell=False
   
   ## Top Recommendations
   
   1. **Fix Security Vulnerabilities** (3 hours)
      - Address 3 critical security issues
      - Update 1 vulnerable dependency
   
   2. **Improve Test Coverage** (8 hours)
      - Add tests for 15 uncovered functions
      - Increase coverage from 67% to 80%
   
   3. **Reduce Complexity** (5 hours)
      - Refactor 3 functions with complexity > 15
      - Split large modules into smaller ones
   
   4. **Update Documentation** (4 hours)
      - Add docstrings to 23 public functions
      - Update README with API examples

Advanced Features
-----------------

**1. Custom Analysis Rules:**

.. code-block:: yaml

   - id: custom_checks
     action: <AUTO>check custom coding standards:
       - Max function length: 50 lines
       - Max file length: 500 lines
       - Naming conventions per style guide
       - Required file headers
       - Banned functions/patterns</AUTO>

**2. Incremental Analysis:**

.. code-block:: yaml

   - id: incremental_check
     action: <AUTO>analyze only files changed since {{last_commit}}:
       - Get git diff for changed files
       - Run full analysis on changed files
       - Quick scan on dependent files
       - Update baseline metrics</AUTO>
     condition: "{{incremental_mode}} == true"

**3. Language-Specific Analysis:**

.. code-block:: yaml

   - id: python_specific
     action: <AUTO>run Python-specific analysis:
       - Type hints coverage
       - PEP 8 compliance
       - Import order
       - Async/await usage patterns</AUTO>
     condition: "'python' in {{languages}}"

CI/CD Integration
-----------------

**GitHub Actions:**

.. code-block:: yaml

   - name: Code Analysis
     uses: orchestrator/analyze@v1
     with:
       config: code_analysis_suite.yaml
       depth: comprehensive
       fail-on: critical

**GitLab CI:**

.. code-block:: yaml

   code_analysis:
     script:
       - orchestrator run code_analysis_suite.yaml
         --input repo_path="$CI_PROJECT_DIR"
     artifacts:
       reports:
         codequality: analysis_report.json

**Jenkins:**

.. code-block:: groovy

   stage('Code Analysis') {
       steps {
           sh '''
               orchestrator run code_analysis_suite.yaml \
                 --input repo_path="${WORKSPACE}"
           '''
       }
       post {
           always {
               publishHTML([
                   reportDir: '.',
                   reportFiles: 'analysis_report.html',
                   reportName: 'Code Analysis Report'
               ])
           }
       }
   }

Performance Optimization
------------------------

The pipeline optimizes performance through:

**1. Parallel Execution**
- Multiple files analyzed simultaneously
- Independent checks run in parallel
- Configurable worker pool size

**2. Intelligent Caching**
- Cache analysis results for unchanged files
- Reuse static analysis data
- Skip redundant checks

**3. Incremental Mode**
- Analyze only changed files in CI/CD
- Update metrics incrementally
- Faster feedback loops

Customization Examples
----------------------

**1. Security-Focused Analysis:**

.. code-block:: yaml

   orchestrator run code_analysis_suite.yaml \
     --input repo_path="." \
     --input security_scan=true \
     --input performance_check=false \
     --input doc_check=false \
     --input severity_threshold="medium"

**2. Quick PR Check:**

.. code-block:: yaml

   orchestrator run code_analysis_suite.yaml \
     --input repo_path="." \
     --input analysis_depth="quick" \
     --input languages='["python"]' \
     --input severity_threshold="high"

**3. Full Audit:**

.. code-block:: yaml

   orchestrator run code_analysis_suite.yaml \
     --input repo_path="." \
     --input analysis_depth="comprehensive" \
     --input security_scan=true \
     --input performance_check=true \
     --input doc_check=true

Best Practices
--------------

1. **Start with Warnings**: Begin with non-blocking analysis before enforcing
2. **Customize Rules**: Tailor checks to your team's standards
3. **Regular Scans**: Run comprehensive analysis weekly
4. **CI/CD Integration**: Add quick checks to every PR
5. **Track Trends**: Monitor quality metrics over time
6. **Team Training**: Help developers understand and fix issues
7. **Incremental Improvement**: Fix issues gradually by priority

Key Takeaways
-------------

This example demonstrates the power of Orchestrator's declarative framework:

1. **Zero Code Required**: Complete analysis pipeline in pure YAML
2. **Comprehensive Coverage**: Security, quality, performance, and more
3. **AI-Powered Insights**: Intelligent recommendations beyond static rules
4. **Parallel Processing**: Scales to large codebases efficiently
5. **CI/CD Ready**: Integrates seamlessly with existing workflows
6. **Customizable**: Adapt to any language or coding standard

The declarative approach makes sophisticated code analysis accessible to all teams.

Next Steps
----------

- Try the :doc:`automated_testing_system` for test generation
- Explore :doc:`customer_support_automation` for issue management
- Read the :doc:`../../advanced/security_scanning` guide
- Check the :doc:`../../user_guide/ci_cd_integration` guide