Code Analysis Suite
===================

This example demonstrates how to build a comprehensive code analysis pipeline that performs automated code review, identifies potential issues, suggests improvements, and generates detailed reports. The suite leverages multiple AI models for different analysis tasks.

.. note::
   **Level:** Advanced  
   **Duration:** 60-90 minutes  
   **Prerequisites:** Python knowledge, understanding of code quality concepts, Orchestrator framework installed

Overview
--------

The Code Analysis Suite performs:

1. **Static Code Analysis**: Syntax checking, linting, and type checking
2. **Security Scanning**: Identify vulnerabilities and security issues
3. **Code Quality Review**: Complexity analysis and maintainability scoring
4. **AI-Powered Review**: Intelligent code review with improvement suggestions
5. **Documentation Analysis**: Check documentation coverage and quality
6. **Performance Profiling**: Identify performance bottlenecks
7. **Report Generation**: Comprehensive analysis reports with actionable insights

**Key Features:**
- Multi-language support (Python, JavaScript, TypeScript, Go, etc.)
- Parallel analysis for large codebases
- Incremental analysis for continuous integration
- Customizable rule sets and thresholds
- Integration with popular development tools

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/orchestrator.git
   cd orchestrator
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export GITHUB_TOKEN="your-github-token"  # Optional, for PR integration
   
   # Run the example
   python examples/code_analysis_suite.py --repo /path/to/your/code

Complete Implementation
-----------------------

Pipeline Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # code_analysis_pipeline.yaml
   id: code_analysis_suite
   name: Comprehensive Code Analysis Pipeline
   version: "1.0"
   
   metadata:
     description: "Multi-stage code analysis with AI-powered insights"
     author: "DevOps Team"
     tags: ["code-analysis", "security", "quality", "ci-cd"]
   
   models:
     code_reviewer:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.2
     security_analyzer:
       provider: "anthropic"
       model: "claude-3-opus"
       temperature: 0.1
     doc_reviewer:
       provider: "openai"
       model: "gpt-3.5-turbo"
       temperature: 0.3
   
   context:
     languages: ["python", "javascript", "typescript", "go"]
     analysis_depth: "comprehensive"
     parallel_workers: 5
   
   tasks:
     - id: discover_code
       name: "Discover Code Files"
       action: "discover_files"
       parameters:
         path: "{{ repo_path }}"
         patterns: <AUTO>Determine file patterns based on languages</AUTO>
         exclude_dirs: [".git", "node_modules", "venv", "__pycache__"]
       outputs:
         - file_list
         - statistics
     
     - id: static_analysis
       name: "Static Code Analysis"
       action: "run_static_analysis"
       parallel: true
       for_each: "{{ discover_code.file_list }}"
       parameters:
         file: "{{ item }}"
         checks:
           - syntax
           - linting
           - type_checking
           - complexity
       dependencies:
         - discover_code
       outputs:
         - issues
         - metrics
     
     - id: security_scan
       name: "Security Vulnerability Scan"
       action: "security_analysis"
       model: "security_analyzer"
       parameters:
         files: "{{ discover_code.file_list }}"
         scan_types: <AUTO>Select appropriate security scans based on language</AUTO>
         severity_threshold: "medium"
       dependencies:
         - discover_code
       outputs:
         - vulnerabilities
         - security_score
     
     - id: code_review
       name: "AI-Powered Code Review"
       action: "ai_code_review"
       model: "code_reviewer"
       parallel: true
       max_workers: 3
       for_each: "{{ discover_code.file_list }}"
       parameters:
         file: "{{ item }}"
         context: "{{ static_analysis.issues[item] }}"
         review_aspects: <AUTO>Focus on code quality, best practices, and maintainability</AUTO>
       dependencies:
         - static_analysis
       outputs:
         - review_comments
         - improvement_suggestions
     
     - id: documentation_check
       name: "Documentation Analysis"
       action: "analyze_documentation"
       model: "doc_reviewer"
       parameters:
         code_files: "{{ discover_code.file_list }}"
         doc_files: "{{ discover_code.doc_files }}"
         coverage_threshold: 0.8
       dependencies:
         - discover_code
       outputs:
         - doc_coverage
         - missing_docs
         - doc_quality_score
     
     - id: performance_analysis
       name: "Performance Profiling"
       action: "profile_performance"
       parameters:
         files: "{{ discover_code.file_list }}"
         profile_types: <AUTO>Select profiling based on language and file type</AUTO>
         threshold_ms: 100
       dependencies:
         - static_analysis
       outputs:
         - hotspots
         - optimization_suggestions
     
     - id: generate_report
       name: "Generate Analysis Report"
       action: "compile_report"
       model: "code_reviewer"
       parameters:
         static_results: "{{ static_analysis }}"
         security_results: "{{ security_scan }}"
         review_results: "{{ code_review }}"
         doc_results: "{{ documentation_check }}"
         perf_results: "{{ performance_analysis }}"
         format: <AUTO>Choose report format: detailed, summary, or CI-friendly</AUTO>
       dependencies:
         - code_review
         - security_scan
         - documentation_check
         - performance_analysis
       outputs:
         - full_report
         - action_items
         - metrics_summary

Python Implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # code_analysis_suite.py
   import asyncio
   import argparse
   import os
   from pathlib import Path
   from typing import Dict, List, Any, Optional
   import yaml
   import json
   from datetime import datetime
   
   from orchestrator import Orchestrator
   from orchestrator.compiler.yaml_compiler import YAMLCompiler
   from orchestrator.tools.system_tools import FileSystemTool, TerminalTool
   from orchestrator.tools.code_tools import (
       CodeAnalysisTool,
       SecurityScannerTool,
       DocumentationAnalyzerTool,
       PerformanceProfilerTool
   )
   from orchestrator.integrations.github import GitHubIntegration
   
   
   class CodeAnalysisSuite:
       """
       Comprehensive code analysis suite for automated code review.
       
       Features:
       - Multi-language support
       - Parallel analysis
       - AI-powered insights
       - Security scanning
       - Performance profiling
       """
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.orchestrator = None
           self.github = None
           self._setup_orchestrator()
       
       def _setup_orchestrator(self):
           """Initialize orchestrator with analysis tools."""
           self.orchestrator = Orchestrator()
           
           # Register AI models
           self._register_models()
           
           # Initialize analysis tools
           self.tools = {
               'file_system': FileSystemTool(),
               'terminal': TerminalTool(),
               'code_analyzer': CodeAnalysisTool(self.config),
               'security_scanner': SecurityScannerTool(self.config),
               'doc_analyzer': DocumentationAnalyzerTool(),
               'profiler': PerformanceProfilerTool()
           }
           
           # Setup GitHub integration if token available
           if self.config.get('github_token'):
               self.github = GitHubIntegration(
                   token=self.config['github_token']
               )
       
       def _register_models(self):
           """Register AI models for code analysis."""
           # Similar to research assistant example
           pass
       
       async def analyze_repository(
           self,
           repo_path: str,
           options: Optional[Dict[str, Any]] = None
       ) -> Dict[str, Any]:
           """
           Analyze a complete repository.
           
           Args:
               repo_path: Path to repository
               options: Analysis options
               
           Returns:
               Analysis results and report
           """
           print(f"🔍 Starting analysis of: {repo_path}")
           
           # Default options
           options = options or {}
           options.setdefault('languages', ['python', 'javascript'])
           options.setdefault('depth', 'comprehensive')
           options.setdefault('parallel', True)
           
           # Load pipeline
           compiler = YAMLCompiler()
           pipeline = compiler.compile_file("code_analysis_pipeline.yaml")
           
           # Set context
           pipeline.set_context({
               'repo_path': repo_path,
               'options': options,
               'timestamp': datetime.now().isoformat()
           })
           
           # Execute analysis
           try:
               results = await self.orchestrator.execute_pipeline(
                   pipeline,
                   progress_callback=self._progress_callback
               )
               
               # Process results
               analysis_report = await self._process_results(results)
               
               # Create PR comment if in CI environment
               if os.getenv('CI') and self.github:
                   await self._post_pr_comment(analysis_report)
               
               return analysis_report
               
           except Exception as e:
               print(f"❌ Analysis failed: {str(e)}")
               raise
       
       async def _progress_callback(self, task_id: str, progress: float, message: str):
           """Handle progress updates."""
           icons = {
               'discover_code': '📁',
               'static_analysis': '🔍',
               'security_scan': '🔐',
               'code_review': '🤖',
               'documentation_check': '📚',
               'performance_analysis': '⚡',
               'generate_report': '📊'
           }
           icon = icons.get(task_id, '▶️')
           print(f"{icon} {task_id}: {progress:.0%} - {message}")
       
       async def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
           """Process and format analysis results."""
           # Calculate overall scores
           quality_score = self._calculate_quality_score(results)
           security_score = results.get('security_scan', {}).get('security_score', 0)
           doc_score = results.get('documentation_check', {}).get('doc_quality_score', 0)
           
           # Categorize issues by severity
           issues = self._categorize_issues(results)
           
           # Generate recommendations
           recommendations = self._generate_recommendations(results)
           
           return {
               'summary': {
                   'quality_score': quality_score,
                   'security_score': security_score,
                   'documentation_score': doc_score,
                   'total_issues': sum(len(v) for v in issues.values()),
                   'files_analyzed': len(results.get('discover_code', {}).get('file_list', []))
               },
               'issues': issues,
               'recommendations': recommendations,
               'detailed_results': results,
               'report': results.get('generate_report', {}).get('full_report', ''),
               'action_items': results.get('generate_report', {}).get('action_items', [])
           }
       
       def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
           """Calculate overall code quality score."""
           scores = []
           
           # Static analysis score
           static_issues = results.get('static_analysis', {}).get('issues', {})
           if static_issues:
               total_files = len(static_issues)
               files_with_issues = sum(1 for issues in static_issues.values() if issues)
               scores.append(1 - (files_with_issues / total_files))
           
           # Complexity score
           metrics = results.get('static_analysis', {}).get('metrics', {})
           if metrics:
               avg_complexity = sum(m.get('complexity', 0) for m in metrics.values()) / len(metrics)
               scores.append(max(0, 1 - (avg_complexity / 20)))  # 20 is max acceptable complexity
           
           return sum(scores) / len(scores) if scores else 0
       
       def _categorize_issues(self, results: Dict[str, Any]) -> Dict[str, List[Dict]]:
           """Categorize all issues by severity."""
           categorized = {
               'critical': [],
               'high': [],
               'medium': [],
               'low': []
           }
           
           # Security vulnerabilities
           vulnerabilities = results.get('security_scan', {}).get('vulnerabilities', [])
           for vuln in vulnerabilities:
               categorized[vuln.get('severity', 'low')].append({
                   'type': 'security',
                   'issue': vuln
               })
           
           # Code quality issues
           static_issues = results.get('static_analysis', {}).get('issues', {})
           for file, issues in static_issues.items():
               for issue in issues:
                   severity = self._map_issue_severity(issue)
                   categorized[severity].append({
                       'type': 'quality',
                       'file': file,
                       'issue': issue
                   })
           
           return categorized
       
       def _map_issue_severity(self, issue: Dict[str, Any]) -> str:
           """Map issue type to severity level."""
           severity_map = {
               'error': 'high',
               'warning': 'medium',
               'info': 'low',
               'complexity_high': 'high',
               'complexity_medium': 'medium',
               'security': 'critical'
           }
           return severity_map.get(issue.get('type', ''), 'low')
       
       def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
           """Generate actionable recommendations."""
           recommendations = []
           
           # Security recommendations
           security_score = results.get('security_scan', {}).get('security_score', 100)
           if security_score < 80:
               recommendations.append("🔐 Address critical security vulnerabilities immediately")
           
           # Documentation recommendations
           doc_coverage = results.get('documentation_check', {}).get('doc_coverage', 1.0)
           if doc_coverage < 0.8:
               recommendations.append("📚 Improve documentation coverage (currently {:.0%})".format(doc_coverage))
           
           # Performance recommendations
           hotspots = results.get('performance_analysis', {}).get('hotspots', [])
           if hotspots:
               recommendations.append(f"⚡ Optimize {len(hotspots)} performance hotspots")
           
           # Code quality recommendations
           review_results = results.get('code_review', {}).get('improvement_suggestions', [])
           if review_results:
               top_suggestions = review_results[:3]
               for suggestion in top_suggestions:
                   recommendations.append(f"💡 {suggestion}")
           
           return recommendations
       
       async def _post_pr_comment(self, analysis_report: Dict[str, Any]):
           """Post analysis results as PR comment."""
           if not self.github:
               return
           
           # Format comment
           comment = self._format_pr_comment(analysis_report)
           
           # Post to PR
           pr_number = os.getenv('GITHUB_PR_NUMBER')
           if pr_number:
               await self.github.post_pr_comment(
                   pr_number=int(pr_number),
                   comment=comment
               )

Tool Integration
^^^^^^^^^^^^^^^^

The analysis suite uses specialized tools:

.. code-block:: python

   # Code Analysis Tools
   # CodeAnalysisTool: Runs static analysis using language-specific tools
   # SecurityScannerTool: Scans for vulnerabilities using bandit, safety, etc.
   # DocumentationAnalyzerTool: Analyzes documentation coverage and quality
   # PerformanceProfilerTool: Profiles code performance
   
   # Language-specific analyzers:
   # Python: pylint, mypy, black, isort, bandit
   # JavaScript/TypeScript: eslint, tsc, prettier
   # Go: go vet, golint, gofmt
   
   # Security scanners:
   # - SAST (Static Application Security Testing)
   # - Dependency vulnerability scanning
   # - Secret detection
   # - License compliance

Running the Analysis
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # main.py
   import asyncio
   import argparse
   from code_analysis_suite import CodeAnalysisSuite
   
   async def main():
       parser = argparse.ArgumentParser(description='Code Analysis Suite')
       parser.add_argument('--repo', required=True, help='Repository path')
       parser.add_argument('--languages', nargs='+', default=['python'])
       parser.add_argument('--output', default='analysis_report.json')
       parser.add_argument('--format', choices=['json', 'markdown', 'html'], default='markdown')
       
       args = parser.parse_args()
       
       # Configuration
       config = {
           'openai_api_key': os.getenv('OPENAI_API_KEY'),
           'github_token': os.getenv('GITHUB_TOKEN'),
           'analysis_config': {
               'languages': args.languages,
               'parallel_workers': 5,
               'cache_results': True
           }
       }
       
       # Run analysis
       suite = CodeAnalysisSuite(config)
       results = await suite.analyze_repository(
           repo_path=args.repo,
           options={'languages': args.languages}
       )
       
       # Save report
       if args.format == 'json':
           with open(args.output, 'w') as f:
               json.dump(results, f, indent=2)
       elif args.format == 'markdown':
           with open(args.output, 'w') as f:
               f.write(results['report'])
       
       # Print summary
       print("\n📊 Analysis Complete!")
       print(f"Quality Score: {results['summary']['quality_score']:.2f}/1.0")
       print(f"Security Score: {results['summary']['security_score']:.2f}/1.0")
       print(f"Documentation: {results['summary']['documentation_score']:.2f}/1.0")
       print(f"Total Issues: {results['summary']['total_issues']}")
       
       # Print top recommendations
       print("\n🎯 Top Recommendations:")
       for i, rec in enumerate(results['recommendations'][:5], 1):
           print(f"{i}. {rec}")
   
   if __name__ == "__main__":
       asyncio.run(main())

Advanced Features
-----------------

Incremental Analysis
^^^^^^^^^^^^^^^^^^^^

Analyze only changed files for faster CI/CD:

.. code-block:: python

   class IncrementalAnalyzer:
       """Analyze only changed files since last run."""
       
       def __init__(self, cache_dir: str = ".analysis_cache"):
           self.cache_dir = Path(cache_dir)
           self.cache_dir.mkdir(exist_ok=True)
       
       async def get_changed_files(self, repo_path: str) -> List[str]:
           """Get files changed since last analysis."""
           last_run_file = self.cache_dir / "last_run.json"
           
           if last_run_file.exists():
               with open(last_run_file) as f:
                   last_run = json.load(f)
               
               # Get files modified since last run
               cmd = f"git diff --name-only {last_run['commit_hash']} HEAD"
               result = await self.run_command(cmd, cwd=repo_path)
               return result.stdout.strip().split('\n')
           else:
               # First run, analyze all files
               return None
       
       def save_run_info(self, commit_hash: str):
           """Save current run information."""
           with open(self.cache_dir / "last_run.json", 'w') as f:
               json.dump({
                   'commit_hash': commit_hash,
                   'timestamp': datetime.now().isoformat()
               }, f)

Custom Rule Sets
^^^^^^^^^^^^^^^^

Define custom analysis rules:

.. code-block:: yaml

   # custom_rules.yaml
   rules:
     python:
       complexity:
         max_function_complexity: 10
         max_file_complexity: 50
       
       naming:
         function_pattern: "^[a-z_][a-z0-9_]*$"
         class_pattern: "^[A-Z][a-zA-Z0-9]*$"
       
       docstring:
         require_module_docstring: true
         require_class_docstring: true
         require_function_docstring: true
         min_docstring_length: 10
     
     security:
       banned_functions:
         - name: "eval"
           severity: "critical"
           message: "Use of eval() is a security risk"
         - name: "exec"
           severity: "critical"
           message: "Use of exec() is a security risk"
       
       required_validations:
         - type: "sql_injection"
           pattern: ".*SELECT.*FROM.*WHERE.*"
           requires: "parameterized_query"

CI/CD Integration
-----------------

GitHub Actions Example
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # .github/workflows/code-analysis.yml
   name: Code Analysis
   
   on:
     pull_request:
       types: [opened, synchronize]
   
   jobs:
     analyze:
       runs-on: ubuntu-latest
       
       steps:
         - uses: actions/checkout@v3
           with:
             fetch-depth: 0  # Full history for incremental analysis
         
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.11'
         
         - name: Install dependencies
           run: |
             pip install orchestrator
             pip install -r requirements-dev.txt
         
         - name: Run Code Analysis
           env:
             OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
             GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
             GITHUB_PR_NUMBER: ${{ github.event.pull_request.number }}
           run: |
             python -m orchestrator.examples.code_analysis_suite \
               --repo . \
               --languages python javascript \
               --output analysis_report.md \
               --format markdown
         
         - name: Upload Report
           uses: actions/upload-artifact@v3
           with:
             name: analysis-report
             path: analysis_report.md
         
         - name: Comment PR
           if: failure()
           uses: actions/github-script@v6
           with:
             script: |
               github.rest.issues.createComment({
                 issue_number: context.issue.number,
                 owner: context.repo.owner,
                 repo: context.repo.repo,
                 body: '❌ Code analysis found issues. Please check the analysis report.'
               })

GitLab CI Example
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   # .gitlab-ci.yml
   code_analysis:
     stage: test
     image: python:3.11
     
     before_script:
       - pip install orchestrator
       - pip install -r requirements-dev.txt
     
     script:
       - |
         python -m orchestrator.examples.code_analysis_suite \
           --repo $CI_PROJECT_DIR \
           --languages python \
           --output analysis_report.json \
           --format json
     
     artifacts:
       reports:
         codequality: analysis_report.json
       paths:
         - analysis_report.json
     
     only:
       - merge_requests

Performance Optimization
------------------------

Parallel Analysis
^^^^^^^^^^^^^^^^^

.. code-block:: python

   async def analyze_files_parallel(files: List[str], max_workers: int = 5):
       """Analyze multiple files in parallel."""
       semaphore = asyncio.Semaphore(max_workers)
       
       async def analyze_with_limit(file: str):
           async with semaphore:
               return await analyze_single_file(file)
       
       tasks = [analyze_with_limit(f) for f in files]
       return await asyncio.gather(*tasks)

Caching Strategy
^^^^^^^^^^^^^^^^

.. code-block:: python

   class AnalysisCache:
       """Cache analysis results for unchanged files."""
       
       def __init__(self, cache_backend='redis'):
           self.cache = self._init_cache(cache_backend)
       
       def get_file_hash(self, file_path: str) -> str:
           """Get hash of file content."""
           with open(file_path, 'rb') as f:
               return hashlib.sha256(f.read()).hexdigest()
       
       async def get_cached_result(self, file_path: str) -> Optional[Dict]:
           """Get cached analysis result if file unchanged."""
           file_hash = self.get_file_hash(file_path)
           cache_key = f"analysis:{file_path}:{file_hash}"
           
           return await self.cache.get(cache_key)
       
       async def cache_result(self, file_path: str, result: Dict):
           """Cache analysis result."""
           file_hash = self.get_file_hash(file_path)
           cache_key = f"analysis:{file_path}:{file_hash}"
           
           await self.cache.set(cache_key, result, ttl=86400)  # 24 hours

Testing the Suite
-----------------

.. code-block:: python

   # test_code_analysis.py
   import pytest
   from code_analysis_suite import CodeAnalysisSuite
   
   @pytest.mark.asyncio
   async def test_python_analysis():
       """Test Python code analysis."""
       suite = CodeAnalysisSuite({})
       
       # Create test file
       test_code = '''
   def complex_function(x, y, z):
       """This function has high complexity."""
       if x > 0:
           if y > 0:
               if z > 0:
                   return x + y + z
               else:
                   return x + y
           else:
               return x
       else:
           return 0
   '''
       
       with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
           f.write(test_code)
           f.flush()
           
           results = await suite.analyze_repository(
               repo_path=os.path.dirname(f.name),
               options={'languages': ['python']}
           )
           
           # Check complexity was detected
           assert results['summary']['quality_score'] < 1.0
           assert any('complexity' in str(issue).lower() 
                     for issue in results['issues']['high'])
   
   @pytest.mark.asyncio
   async def test_security_scanning():
       """Test security vulnerability detection."""
       suite = CodeAnalysisSuite({})
       
       # Create file with security issue
       vulnerable_code = '''
   import os
   
   def run_command(user_input):
       """Vulnerable to command injection."""
       os.system(f"echo {user_input}")  # Security issue!
   '''
       
       # Test should detect the vulnerability
       # ... test implementation ...

Best Practices
--------------

1. **Configure for Your Stack**: Customize rules and tools for your technology stack
2. **Start with Warnings**: Begin with non-blocking warnings before enforcing rules
3. **Incremental Adoption**: Enable rules gradually to avoid overwhelming developers
4. **Cache Results**: Use caching for large codebases to improve performance
5. **Integrate Early**: Add to CI/CD pipeline early in development process
6. **Regular Updates**: Keep analysis tools and rules updated
7. **Team Training**: Educate team on understanding and fixing issues

Summary
-------

The Code Analysis Suite demonstrates how to:

- Build comprehensive code analysis pipelines
- Integrate multiple analysis tools and AI models
- Handle large codebases with parallel processing
- Generate actionable insights and recommendations
- Integrate with CI/CD workflows
- Maintain code quality standards automatically

This example provides a foundation for building custom code analysis solutions tailored to specific needs and technologies.