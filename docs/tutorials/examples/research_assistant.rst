Research Assistant Pipeline
===========================

This example demonstrates how to build a comprehensive research assistant that can conduct web research, analyze findings, synthesize information, and generate detailed reports. This showcases the Orchestrator's ability to handle complex, multi-step workflows with intelligent model selection and robust error handling.

.. note::
   **Level:** Intermediate  
   **Duration:** 45-60 minutes  
   **Prerequisites:** Basic Python knowledge, Orchestrator framework installed

Overview
--------

The Research Assistant pipeline performs the following workflow:

1. **Query Analysis**: Analyze research query and generate search terms
2. **Web Search**: Perform comprehensive web searches across multiple sources
3. **Content Extraction**: Extract and clean relevant content from web pages
4. **Information Synthesis**: Analyze and synthesize findings from multiple sources
5. **Report Generation**: Create structured research reports with citations
6. **Quality Assurance**: Validate findings and check for accuracy

**Key Features Demonstrated:**
- Multi-step pipeline orchestration
- Intelligent model selection based on task complexity
- Error handling with retry logic
- State management and checkpointing
- Performance optimization with caching
- Real-time progress monitoring

Quick Start
-----------

.. code-block:: bash

   # Clone the repository and install dependencies
   git clone https://github.com/your-org/orchestrator.git
   cd orchestrator
   pip install -r requirements.txt
   
   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Run the example
   python examples/research_assistant.py

Complete Implementation
-----------------------

Pipeline Configuration (YAML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, let's define the research pipeline in YAML:

.. code-block:: yaml

   # research_assistant_pipeline.yaml
   id: research_assistant
   name: Comprehensive Research Assistant
   version: "1.0"
   
   metadata:
     description: "Multi-step research pipeline with web search and analysis"
     author: "Research Team"
     tags: ["research", "web-search", "analysis", "reporting"]
   
   models:
     analyzer: 
       provider: "openai"
       model: "gpt-4"
       temperature: 0.1
     searcher:
       provider: "anthropic" 
       model: "claude-3-opus"
       temperature: 0.0
     synthesizer:
       provider: "openai"
       model: "gpt-4"
       temperature: 0.3
   
   context:
     max_sources: 10
     search_depth: 3
     quality_threshold: 0.7
   
   tasks:
     - id: analyze_query
       name: "Analyze Research Query"
       action: "analyze_research_query"
       model: "analyzer"
       parameters:
         query: "{{ user_query }}"
         context: "{{ research_context }}"
         focus_areas: <AUTO>Identify key research focus areas based on the query</AUTO>
       outputs:
         - search_terms
         - research_objectives
         - expected_sources
   
     - id: web_search
       name: "Conduct Web Search"
       action: "comprehensive_web_search"
       model: "searcher"
       parameters:
         search_terms: "{{ analyze_query.search_terms }}"
         max_results: "{{ max_sources }}"
         search_depth: "{{ search_depth }}"
         search_engines: ["google", "bing", "duckduckgo"]
         filters: <AUTO>Determine appropriate search filters for academic reliability</AUTO>
       dependencies:
         - analyze_query
       outputs:
         - search_results
         - source_metadata
   
     - id: extract_content
       name: "Extract and Clean Content"
       action: "extract_web_content"
       parameters:
         urls: "{{ web_search.search_results }}"
         extraction_method: <AUTO>Choose optimal extraction method based on content type</AUTO>
         quality_filter: "{{ quality_threshold }}"
       dependencies:
         - web_search
       outputs:
         - extracted_content
         - content_quality_scores
   
     - id: analyze_sources
       name: "Analyze Source Credibility"
       action: "analyze_source_credibility"
       model: "analyzer"
       parameters:
         sources: "{{ extract_content.extracted_content }}"
         metadata: "{{ web_search.source_metadata }}"
         credibility_criteria: <AUTO>Define credibility criteria for sources</AUTO>
       dependencies:
         - extract_content
       outputs:
         - credibility_scores
         - reliable_sources
   
     - id: synthesize_information
       name: "Synthesize Research Findings"
       action: "synthesize_research"
       model: "synthesizer"
       parameters:
         content: "{{ analyze_sources.reliable_sources }}"
         objectives: "{{ analyze_query.research_objectives }}"
         synthesis_approach: <AUTO>Choose synthesis approach: thematic, chronological, or comparative</AUTO>
       dependencies:
         - analyze_sources
       outputs:
         - key_findings
         - supporting_evidence
         - knowledge_gaps
   
     - id: generate_report
       name: "Generate Research Report"
       action: "generate_research_report"
       model: "synthesizer"
       parameters:
         findings: "{{ synthesize_information.key_findings }}"
         evidence: "{{ synthesize_information.supporting_evidence }}"
         sources: "{{ analyze_sources.reliable_sources }}"
         format: <AUTO>Choose optimal report format: academic, executive summary, or detailed analysis</AUTO>
       dependencies:
         - synthesize_information
       outputs:
         - research_report
         - citation_list
         - recommendations
   
     - id: quality_check
       name: "Quality Assurance Check"
       action: "validate_research_quality"
       model: "analyzer"
       parameters:
         report: "{{ generate_report.research_report }}"
         sources: "{{ analyze_sources.reliable_sources }}"
         objectives: "{{ analyze_query.research_objectives }}"
         quality_criteria: <AUTO>Define quality criteria for research validation</AUTO>
       dependencies:
         - generate_report
       outputs:
         - quality_score
         - validation_report
         - improvement_suggestions

Python Implementation
^^^^^^^^^^^^^^^^^^^^^

Now, let's implement the research assistant in Python:

.. code-block:: python

   # research_assistant.py
   import asyncio
   import logging
   import yaml
   import os
   from datetime import datetime
   from typing import Dict, List, Any, Optional
   
   from orchestrator import Orchestrator
   from orchestrator.compiler.yaml_compiler import YAMLCompiler
   from orchestrator.integrations.openai_model import OpenAIModel
   from orchestrator.integrations.anthropic_model import AnthropicModel
   from orchestrator.state.state_manager import StateManager
   from orchestrator.tools.web_tools import WebSearchTool, HeadlessBrowserTool
   from orchestrator.tools.data_tools import DataProcessingTool
   from orchestrator.core.cache import MemoryCache
   
   # Configure logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   class ResearchAssistant:
       """
       Comprehensive research assistant using the Orchestrator framework.
       
       This class demonstrates advanced features including:
       - Multi-model orchestration
       - Intelligent caching
       - Error handling and retries
       - Progress monitoring
       - State management
       """
       
       def __init__(self, config: Dict[str, Any]):
           self.config = config
           self.orchestrator = None
           self.state_manager = None
           self.cache = None
           # Load orchestrator configuration for web tools
           self.orchestrator_config = self._load_orchestrator_config()
           self._setup_orchestrator()
       
       def _setup_orchestrator(self):
           """Initialize the orchestrator with models and tools."""
           # Initialize state manager for checkpointing
           self.state_manager = StateManager(
               backend_type="file",
               storage_path="./research_checkpoints",
               compression_enabled=True
           )
           
           # Initialize caching for performance
           self.cache = MemoryCache(
               max_size=1000,
               default_ttl=3600  # 1 hour
           )
           
           # Initialize orchestrator
           self.orchestrator = Orchestrator(
               state_manager=self.state_manager,
               cache=self.cache
           )
           
           # Register models
           self._register_models()
           
           # Register tools
           self._register_tools()
       
       def _register_models(self):
           """Register AI models with the orchestrator based on config/models.yaml."""
           # Register OpenAI models if API key is available
           if self.config.get("openai_api_key"):
               try:
                   # Use gpt-4.1 from config/models.yaml
                   gpt4 = OpenAIModel(
                       model_name="gpt-4.1",
                       api_key=self.config["openai_api_key"],
                       max_retries=3,
                       timeout=30.0
                   )
                   self.orchestrator.model_registry.register_model(gpt4)
               except Exception as e:
                   print(f"Failed to register OpenAI model: {e}")
           
           # Register Anthropic models if API key is available  
           if self.config.get("anthropic_api_key"):
               try:
                   # Use claude-4-sonnet from config/models.yaml
                   claude = AnthropicModel(
                       model_name="claude-sonnet-4-20250514",
                       api_key=self.config["anthropic_api_key"],
                       max_retries=3,
                       timeout=30.0
                   )
                   self.orchestrator.model_registry.register_model(claude)
               except Exception as e:
                   print(f"Failed to register Anthropic model: {e}")
       
       def _load_orchestrator_config(self) -> Dict[str, Any]:
           """Load orchestrator configuration for web tools."""
           config_path = "config/orchestrator.yaml"
           if os.path.exists(config_path):
               with open(config_path, 'r') as f:
                   return yaml.safe_load(f)
           else:
               # Default configuration for web tools
               return {
                   "web_tools": {
                       "search": {
                           "default_backend": "duckduckgo",
                           "max_results": 10,
                           "timeout": 30
                       },
                       "scraping": {
                           "timeout": 30,
                           "max_content_length": 1048576,
                           "user_agent": "Mozilla/5.0 (compatible; Research Assistant)"
                       },
                       "browser": {
                           "headless": True,
                           "timeout": 30
                       },
                       "rate_limiting": {
                           "enabled": True,
                           "requests_per_minute": 30,
                           "delay_between_requests": 2
                       },
                       "caching": {
                           "enabled": True,
                           "ttl": 3600,
                           "max_cache_size": 100
                       }
                   }
               }
       
       def _register_tools(self):
           """Register tools for web search and content extraction."""
           # Tools are handled by the control system in the orchestrator
           # For this example, we'll store them as instance variables
           self.web_search = WebSearchTool(self.orchestrator_config)
           self.browser_tool = HeadlessBrowserTool(self.orchestrator_config)
           self.data_analyzer = DataProcessingTool()
       
       async def conduct_research(self, query: str, context: str = "") -> Dict[str, Any]:
           """
           Conduct comprehensive research on a given query.
           
           Args:
               query: The research question or topic
               context: Additional context to guide the research
               
           Returns:
               Dictionary containing research results, report, and metadata
           """
           logger.info(f"Starting research for query: {query}")
           
           # Load pipeline configuration
           compiler = YAMLCompiler()
           pipeline = compiler.compile_file("research_assistant_pipeline.yaml")
           
           # Set pipeline context
           pipeline.set_context({
               "user_query": query,
               "research_context": context,
               "start_time": datetime.now().isoformat()
           })
           
           # Execute pipeline with progress monitoring
           try:
               result = await self.orchestrator.execute_pipeline(
                   pipeline,
                   progress_callback=self._progress_callback,
                   error_callback=self._error_callback
               )
               
               # Extract key results
               research_results = {
                   "query": query,
                   "context": context,
                   "search_terms": result.get("analyze_query", {}).get("search_terms", []),
                   "sources_found": len(result.get("web_search", {}).get("search_results", [])),
                   "reliable_sources": result.get("analyze_sources", {}).get("reliable_sources", []),
                   "key_findings": result.get("synthesize_information", {}).get("key_findings", []),
                   "research_report": result.get("generate_report", {}).get("research_report", ""),
                   "citations": result.get("generate_report", {}).get("citation_list", []),
                   "quality_score": result.get("quality_check", {}).get("quality_score", 0),
                   "recommendations": result.get("generate_report", {}).get("recommendations", []),
                   "execution_time": result.get("metadata", {}).get("execution_time", 0),
                   "model_costs": result.get("metadata", {}).get("model_costs", {})
               }
               
               logger.info(f"Research completed successfully. Quality score: {research_results['quality_score']}")
               return research_results
               
           except Exception as e:
               logger.error(f"Research failed: {str(e)}")
               # Attempt to recover from checkpoint
               return await self._recover_from_checkpoint(pipeline.id)
       
       async def _progress_callback(self, task_id: str, progress: float, message: str):
           """Handle progress updates during pipeline execution."""
           logger.info(f"Task {task_id}: {progress:.1%} - {message}")
       
       async def _error_callback(self, task_id: str, error: Exception):
           """Handle errors during pipeline execution."""
           logger.error(f"Task {task_id} failed: {str(error)}")
           # Implement retry logic or fallback strategies here
       
       async def _recover_from_checkpoint(self, pipeline_id: str) -> Dict[str, Any]:
           """Recover pipeline execution from the last checkpoint."""
           try:
               logger.info("Attempting to recover from checkpoint...")
               recovered_state = await self.state_manager.load_pipeline_state(pipeline_id)
               
               # Resume pipeline execution
               result = await self.orchestrator.resume_pipeline(pipeline_id)
               return result
               
           except Exception as e:
               logger.error(f"Recovery failed: {str(e)}")
               return {
                   "error": "Pipeline execution failed and could not be recovered",
                   "details": str(e)
               }
       
       def generate_research_summary(self, results: Dict[str, Any]) -> str:
           """Generate a formatted summary of research results."""
           summary = f"""
   Research Summary
   ================
   
   Query: {results['query']}
   Search Terms: {', '.join(results['search_terms'])}
   Sources Found: {results['sources_found']}
   Reliable Sources: {len(results['reliable_sources'])}
   Quality Score: {results['quality_score']:.2f}/1.0
   
   Key Findings:
   {chr(10).join(f"• {finding}" for finding in results['key_findings'])}
   
   Recommendations:
   {chr(10).join(f"• {rec}" for rec in results['recommendations'])}
   
   Execution Time: {results['execution_time']:.2f} seconds
   Model Costs: ${sum(results['model_costs'].values()):.4f}
   """
           return summary

Tool Integration
^^^^^^^^^^^^^^^^

The research assistant uses real web tools for actual data retrieval:

.. code-block:: python

   # Real Web Tools Implementation
   # WebSearchTool: Uses DuckDuckGo (ddgs library) for real web searches
   # HeadlessBrowserTool: Uses requests and BeautifulSoup for content extraction
   # DataProcessingTool: Analyzes source credibility and quality
   
   # These tools provide:
   # - Real web search using DuckDuckGo API (no API key required)
   # - Actual content extraction from web pages
   # - Source credibility analysis with real data
   # - Quality scoring based on actual content
   # - Rate limiting to prevent abuse
   # - Error handling for network issues
   
   # Dependencies required:
   # - ddgs>=9.0.0 (DuckDuckGo search)
   # - requests>=2.28.0 (HTTP requests)
   # - beautifulsoup4>=4.11.0 (HTML parsing)
   # - lxml>=4.9.0 (XML/HTML parser backend)

Running the Research Assistant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's how to use the research assistant:

.. code-block:: python

   # main.py
   import asyncio
   import os
   from research_assistant import ResearchAssistant
   
   async def main():
       # Configuration
       config = {
           "openai_api_key": os.getenv("OPENAI_API_KEY"),
           "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY")
       }
       
       # Initialize research assistant
       assistant = ResearchAssistant(config)
       
       # Conduct research
       query = "What are the latest developments in quantum computing for machine learning?"
       context = "Focus on practical applications and recent breakthroughs in 2024"
       
       results = await assistant.conduct_research(query, context)
       
       # Generate and display summary
       summary = assistant.generate_research_summary(results)
       print(summary)
       
       # Save detailed report
       with open("research_report.md", "w") as f:
           f.write(results["research_report"])
       
       print(f"Detailed report saved to research_report.md")
       print(f"Research quality score: {results['quality_score']:.2f}/1.0")
   
   if __name__ == "__main__":
       asyncio.run(main())

Advanced Features
-----------------

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

The research assistant includes several performance optimizations:

.. code-block:: python

   # performance_config.py
   PERFORMANCE_CONFIG = {
       "caching": {
           "enabled": True,
           "ttl": 3600,  # 1 hour
           "max_size": 1000
       },
       "parallel_processing": {
           "max_concurrent_searches": 5,
           "max_concurrent_extractions": 10
       },
       "resource_limits": {
           "max_memory": "2GB",
           "max_execution_time": 1800  # 30 minutes
       },
       "retry_strategy": {
           "max_retries": 3,
           "backoff_factor": 2.0,
           "timeout": 30
       }
   }

Error Handling and Recovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comprehensive error handling ensures robust operation:

.. code-block:: python

   # error_handling.py
   class ResearchError(Exception):
       """Base exception for research operations."""
       pass
   
   class SearchError(ResearchError):
       """Raised when web search fails."""
       pass
   
   class ExtractionError(ResearchError):
       """Raised when content extraction fails."""
       pass
   
   async def handle_search_error(self, error: SearchError, task_context: Dict):
       """Handle search-related errors with fallback strategies."""
       logger.warning(f"Search error: {str(error)}")
       
       # Try alternative search terms
       if "alternative_terms" in task_context:
           return await self._search_with_alternatives(task_context["alternative_terms"])
       
       # Fall back to cached results
       cached_results = await self.cache.get(f"search:{task_context['query']}")
       if cached_results:
           logger.info("Using cached search results")
           return cached_results
       
       # Last resort: use mock data for development
       if self.config.get("development_mode", False):
           return await self._generate_mock_results(task_context["query"])
       
       raise error

Monitoring and Analytics
^^^^^^^^^^^^^^^^^^^^^^^^

Track research performance and quality:

.. code-block:: python

   # monitoring.py
   from orchestrator.monitoring.metrics import MetricsCollector
   
   class ResearchMetrics:
       """Collect and analyze research performance metrics."""
       
       def __init__(self):
           self.metrics = MetricsCollector()
       
       def track_research_session(self, results: Dict[str, Any]):
           """Track metrics for a research session."""
           self.metrics.increment("research_sessions_total")
           self.metrics.histogram("research_duration", results["execution_time"])
           self.metrics.gauge("research_quality_score", results["quality_score"])
           self.metrics.histogram("sources_found", results["sources_found"])
           self.metrics.histogram("reliable_sources", len(results["reliable_sources"]))
       
       def get_performance_report(self) -> Dict[str, Any]:
           """Generate performance report."""
           return {
               "total_sessions": self.metrics.get_counter("research_sessions_total"),
               "average_duration": self.metrics.get_histogram_avg("research_duration"),
               "average_quality": self.metrics.get_gauge_avg("research_quality_score"),
               "success_rate": self.metrics.calculate_success_rate(),
               "cost_per_session": self.metrics.calculate_average_cost()
           }

Testing and Validation
-----------------------

Comprehensive testing ensures reliability:

.. code-block:: python

   # test_research_assistant.py
   import pytest
   import asyncio
   from unittest.mock import Mock, patch
   from research_assistant import ResearchAssistant
   
   class TestResearchAssistant:
       """Test suite for research assistant."""
       
       @pytest.fixture
       def assistant(self):
           config = {
               "openai_api_key": "test-key",
               "anthropic_api_key": "test-key", 
               "serp_api_key": "test-key"
           }
           return ResearchAssistant(config)
       
       @pytest.mark.asyncio
       async def test_basic_research_flow(self, assistant):
           """Test basic research workflow."""
           query = "Test query"
           context = "Test context"
           
           with patch.object(assistant, '_conduct_web_search') as mock_search:
               mock_search.return_value = {
                   "search_results": ["url1", "url2"],
                   "source_metadata": []
               }
               
               results = await assistant.conduct_research(query, context)
               
               assert results["query"] == query
               assert "research_report" in results
               assert results["quality_score"] > 0
       
       @pytest.mark.asyncio
       async def test_error_recovery(self, assistant):
           """Test error recovery mechanisms."""
           with patch.object(assistant.orchestrator, 'execute_pipeline') as mock_execute:
               mock_execute.side_effect = Exception("Test error")
               
               # Should attempt recovery
               results = await assistant.conduct_research("test query")
               
               assert "error" in results
               # Verify recovery was attempted
               assert assistant.state_manager.load_pipeline_state.called
       
       @pytest.mark.asyncio
       async def test_performance_optimization(self, assistant):
           """Test performance optimization features."""
           # Test caching
           query = "cached query"
           
           # First request
           results1 = await assistant.conduct_research(query)
           
           # Second request (should use cache)
           results2 = await assistant.conduct_research(query)
           
           # Verify cache was used
           assert assistant.cache.get_statistics().hit_rate > 0

Deployment Configuration
------------------------

Production deployment configuration:

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'
   
   services:
     research-assistant:
       build: .
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
         - SERP_API_KEY=${SERP_API_KEY}
         - POSTGRES_URL=${POSTGRES_URL}
         - REDIS_URL=${REDIS_URL}
       volumes:
         - ./research_data:/app/data
         - ./research_checkpoints:/app/checkpoints
       depends_on:
         - postgres
         - redis
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
   
     postgres:
       image: postgres:15
       environment:
         POSTGRES_DB: research_db
         POSTGRES_USER: research_user
         POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
       volumes:
         - postgres_data:/var/lib/postgresql/data
   
     redis:
       image: redis:7-alpine
       volumes:
         - redis_data:/data
   
   volumes:
     postgres_data:
     redis_data:

Key Takeaways
-------------

This research assistant example demonstrates:

1. **Complex Pipeline Orchestration**: Multi-step workflows with dependencies
2. **Intelligent Model Selection**: Different models for different tasks
3. **Robust Error Handling**: Comprehensive error recovery and fallback strategies
4. **Performance Optimization**: Caching, parallel processing, and resource management
5. **Production Readiness**: Monitoring, logging, and deployment configuration
6. **Extensibility**: Modular design allowing easy addition of new features

The example showcases how the Orchestrator framework can handle complex, real-world applications while maintaining clean, maintainable code and providing robust error handling and performance optimization.

Next Steps
----------

- Explore the :doc:`code_analysis_suite` example for development workflows
- Learn about :doc:`multi_agent_collaboration` for complex AI systems
- Check out the :doc:`../advanced/performance_optimization` guide for optimization techniques
- Review the :doc:`../advanced/deployment` guide for production deployment strategies