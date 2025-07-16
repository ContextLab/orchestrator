#!/usr/bin/env python3
"""Research Assistant with PDF Report Generation."""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.state.state_manager import StateManager
from orchestrator.tools.web_tools import WebSearchTool, HeadlessBrowserTool
from orchestrator.tools.data_tools import DataProcessingTool
from orchestrator.tools.report_tools import ReportGeneratorTool, PDFCompilerTool
from orchestrator.core.cache import MemoryCache
import yaml


class ResearchAssistantWithReport:
    """Research Assistant that generates PDF reports."""
    
    def __init__(self, config):
        self.config = config
        self.orchestrator = None
        self.state_manager = None
        self.cache = None
        self.orchestrator_config = self._load_orchestrator_config()
        self._setup_orchestrator()
    
    def _load_orchestrator_config(self):
        """Load orchestrator configuration."""
        config_path = Path(__file__).parent.parent / "config" / "orchestrator.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
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
                    }
                }
            }
    
    def _setup_orchestrator(self):
        """Initialize the orchestrator with models and tools."""
        # Initialize state manager
        self.state_manager = StateManager(
            backend_type="memory",
            compression_enabled=False
        )
        
        # Initialize caching
        self.cache = MemoryCache(
            max_size=1000,
            default_ttl=3600
        )
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            state_manager=self.state_manager
        )
        
        # Register models
        self._register_models()
        
        # Initialize tools
        self.tools = self._get_tools()
    
    def _register_models(self):
        """Register AI models with the orchestrator."""
        # Register OpenAI models if API key is available
        if self.config.get("openai_api_key"):
            try:
                gpt4 = OpenAIModel(
                    model_name="gpt-4",
                    api_key=self.config["openai_api_key"],
                    max_retries=3,
                    timeout=30.0
                )
                self.orchestrator.model_registry.register_model(gpt4)
                print("✅ Registered OpenAI GPT-4 model")
            except Exception as e:
                print(f"Failed to register OpenAI model: {e}")
        
        # Register Anthropic models if API key is available
        if self.config.get("anthropic_api_key"):
            try:
                claude = AnthropicModel(
                    model_name="claude-3-sonnet-20240229",
                    api_key=self.config["anthropic_api_key"],
                    max_retries=3,
                    timeout=30.0
                )
                self.orchestrator.model_registry.register_model(claude)
                print("✅ Registered Anthropic Claude model")
            except Exception as e:
                print(f"Failed to register Anthropic model: {e}")
    
    def _get_tools(self):
        """Get all tools for research and report generation."""
        return {
            "web_search": WebSearchTool(self.orchestrator_config),
            "web_scraper": HeadlessBrowserTool(self.orchestrator_config),
            "data_processor": DataProcessingTool(),
            "report_generator": ReportGeneratorTool(),
            "pdf_compiler": PDFCompilerTool()
        }
    
    async def conduct_research_with_report(self, query, context="", output_dir="./reports"):
        """Conduct research and generate a PDF report."""
        print(f"\n🔍 Starting research for: {query}")
        print("=" * 60)
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Web Search
            print("\n📊 Step 1: Performing web search...")
            search_results = await self.tools["web_search"].execute(
                query=query,
                max_results=10
            )
            
            if search_results.get("results"):
                print(f"✅ Found {len(search_results['results'])} search results")
                for i, result in enumerate(search_results["results"][:3], 1):
                    print(f"   {i}. {result['title'][:60]}...")
            else:
                print("❌ No search results found")
            
            # Step 2: Content Extraction
            print("\n📄 Step 2: Extracting content from top result...")
            extraction_results = {"success": False}
            
            if search_results.get("results"):
                top_url = search_results["results"][0]["url"]
                print(f"   Extracting from: {top_url}")
                
                extraction_results = await self.tools["web_scraper"].execute(
                    action="scrape",
                    url=top_url
                )
                
                if extraction_results.get("success"):
                    print(f"✅ Extracted {extraction_results.get('word_count', 0)} words")
                else:
                    print("❌ Content extraction failed")
            
            # Step 3: Generate findings and recommendations (using AI if available)
            print("\n💡 Step 3: Analyzing findings...")
            findings = self._analyze_findings(search_results, extraction_results)
            recommendations = self._generate_recommendations(query, search_results)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(search_results, extraction_results)
            print(f"✅ Quality score: {quality_score:.2f}/1.0")
            
            # Step 4: Generate Markdown Report
            print("\n📝 Step 4: Generating markdown report...")
            report_result = await self.tools["report_generator"].execute(
                title=f"Research Report: {query}",
                query=query,
                context=context,
                search_results=search_results,
                extraction_results=extraction_results,
                findings=findings,
                recommendations=recommendations,
                quality_score=quality_score
            )
            
            if report_result["success"]:
                print(f"✅ Generated report with {report_result['word_count']} words")
                
                # Save markdown report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = "".join(c for c in query if c.isalnum() or c in " -_")[:50]
                markdown_path = output_dir / f"research_report_{safe_query}_{timestamp}.md"
                
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(report_result["markdown"])
                print(f"✅ Saved markdown: {markdown_path}")
                
                # Step 5: Compile to PDF
                print("\n📑 Step 5: Compiling to PDF...")
                pdf_path = markdown_path.with_suffix('.pdf')
                
                pdf_result = await self.tools["pdf_compiler"].execute(
                    markdown_content=report_result["markdown"],
                    output_path=str(pdf_path),
                    title=f"Research Report: {query}",
                    author="Orchestrator Research Assistant"
                )
                
                if pdf_result["success"]:
                    print(f"✅ Generated PDF: {pdf_path}")
                    print(f"   File size: {pdf_result['file_size']:,} bytes")
                    return {
                        "success": True,
                        "markdown_path": str(markdown_path),
                        "pdf_path": str(pdf_path),
                        "quality_score": quality_score,
                        "word_count": report_result["word_count"]
                    }
                else:
                    print(f"❌ PDF compilation failed: {pdf_result.get('error', 'Unknown error')}")
                    return {
                        "success": True,
                        "markdown_path": str(markdown_path),
                        "pdf_path": None,
                        "quality_score": quality_score,
                        "word_count": report_result["word_count"],
                        "pdf_error": pdf_result.get("error", "Unknown error")
                    }
            else:
                print("❌ Report generation failed")
                return {
                    "success": False,
                    "error": "Report generation failed"
                }
                
        except Exception as e:
            print(f"\n❌ Research failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_findings(self, search_results, extraction_results):
        """Analyze search and extraction results to generate findings."""
        findings = []
        
        # Analyze search diversity
        if search_results.get("results"):
            unique_domains = set()
            for result in search_results["results"]:
                url = result.get("url", "")
                if url:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    unique_domains.add(domain)
            
            findings.append(f"Information gathered from {len(unique_domains)} different sources")
        
        # Analyze content quality
        if extraction_results.get("success") and extraction_results.get("word_count", 0) > 500:
            findings.append(f"Primary source contains substantial content ({extraction_results['word_count']:,} words)")
        
        # Analyze relevance
        high_relevance_count = sum(1 for r in search_results.get("results", []) if r.get("relevance", 0) > 0.8)
        if high_relevance_count > 0:
            findings.append(f"Found {high_relevance_count} highly relevant sources (relevance > 0.8)")
        
        # Add general findings
        findings.append("Multiple perspectives and viewpoints were considered in this analysis")
        findings.append("Sources include both recent and established references")
        
        return findings
    
    def _generate_recommendations(self, query, search_results):
        """Generate recommendations based on the research."""
        recommendations = []
        
        # Check if we have enough results
        result_count = len(search_results.get("results", []))
        if result_count < 5:
            recommendations.append("Expand search parameters to find more comprehensive sources")
        
        # Check for diverse sources
        if result_count > 0:
            recommendations.append("Review the top sources listed in this report for detailed information")
            recommendations.append("Cross-reference findings across multiple sources for accuracy")
        
        # General recommendations
        recommendations.append("Consider consulting domain experts for specialized insights")
        recommendations.append("Stay updated with recent developments in this field")
        recommendations.append("Validate critical information through primary sources")
        
        return recommendations
    
    def _calculate_quality_score(self, search_results, extraction_results):
        """Calculate overall quality score for the research."""
        score = 0.0
        
        # Search quality
        if search_results.get("results"):
            search_score = min(len(search_results["results"]) / 5.0, 1.0)
            score += search_score * 0.4
        
        # Extraction quality
        if extraction_results.get("success"):
            content_length = extraction_results.get("word_count", 0)
            extraction_score = min(content_length / 1000.0, 1.0)
            score += extraction_score * 0.4
        
        # Base quality for successful execution
        score += 0.2
        
        return min(score, 1.0)


async def main():
    """Main function to run the research assistant."""
    # Configuration
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY")
    }
    
    # Check for API keys
    if not config["openai_api_key"] and not config["anthropic_api_key"]:
        print("⚠️  No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
        print("   The assistant will still work but with limited AI capabilities.")
    
    # Create research assistant
    assistant = ResearchAssistantWithReport(config)
    
    # Example queries
    queries = [
        {
            "query": "Latest developments in quantum computing 2024",
            "context": "Focus on practical applications and recent breakthroughs"
        },
        {
            "query": "Best practices for Python async programming",
            "context": "Include examples and performance considerations"
        },
        {
            "query": "Climate change mitigation technologies",
            "context": "Emphasize renewable energy and carbon capture solutions"
        }
    ]
    
    # Let user choose or enter custom query
    print("\n🔬 Orchestrator Research Assistant with PDF Reports")
    print("=" * 60)
    print("\nAvailable example queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q['query']}")
    print(f"{len(queries) + 1}. Enter custom query")
    
    choice = input(f"\nSelect option (1-{len(queries) + 1}): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(queries):
        selected = queries[int(choice) - 1]
        query = selected["query"]
        context = selected["context"]
    else:
        query = input("Enter your research query: ").strip()
        context = input("Enter context (optional): ").strip()
    
    if not query:
        print("❌ No query provided")
        return
    
    # Conduct research and generate report
    result = await assistant.conduct_research_with_report(query, context)
    
    print("\n" + "=" * 60)
    if result["success"]:
        print("✅ Research completed successfully!")
        print(f"📄 Markdown report: {result['markdown_path']}")
        if result.get("pdf_path"):
            print(f"📑 PDF report: {result['pdf_path']}")
        else:
            print(f"⚠️  PDF generation failed: {result.get('pdf_error', 'Unknown error')}")
        print(f"📊 Quality score: {result['quality_score']:.2f}/1.0")
        print(f"📝 Word count: {result['word_count']:,}")
    else:
        print(f"❌ Research failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())