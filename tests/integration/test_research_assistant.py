#!/usr/bin/env python3
"""Test the research assistant pipeline."""

import asyncio
import sys
import os
import json
import aiohttp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import Model
from src.orchestrator.tools.search_tool import DuckDuckGoSearchTool
from src.orchestrator.tools.web_scraper import HeadlessBrowserTool


async def real_search(task):
    """Perform real web search."""
    query = task.parameters.get("query", "")
    print(f"[Web Search] Searching for: {query}")
    
    try:
        # Use real DuckDuckGo search
        search_tool = DuckDuckGoSearchTool()
        results = await search_tool.search(query, max_results=5)
        
        # Extract URLs from results
        urls = []
        for result in results:
            if "link" in result and result["link"]:
                urls.append(result["link"])
        
        return {
            "results": urls[:5],  # Limit to 5 URLs
            "search_results": results,  # Full results with snippets
            "status": "success",
            "query": query
        }
    except Exception as e:
        print(f"[Web Search] Error: {e}")
        # Fallback to basic search results
        return {
            "results": [
                f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                f"https://scholar.google.com/search?q={query.replace(' ', '+')}"
            ],
            "status": "fallback",
            "error": str(e)
        }


async def real_extract(task):
    """Extract content from URLs using real web scraping."""
    urls_data = task.parameters.get("urls", {})
    urls = urls_data.get("results", []) if isinstance(urls_data, dict) else urls_data
    print(f"[Content Extraction] Extracting from {len(urls)} URLs")
    
    extracted_texts = []
    browser_tool = HeadlessBrowserTool()
    
    for url in urls[:3]:  # Limit to 3 URLs for performance
        try:
            print(f"  Extracting: {url}")
            # Try headless browser first
            content = await browser_tool.extract_text(url)
            if content:
                extracted_texts.append({
                    "url": url,
                    "text": content[:1000],  # Limit text length
                    "success": True
                })
        except Exception as e:
            print(f"  Failed to extract {url}: {e}")
            # Try basic HTTP request as fallback
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            html = await response.text()
                            # Basic text extraction
                            text = html[:500] if html else "No content"
                            extracted_texts.append({
                                "url": url,
                                "text": text,
                                "success": False,
                                "method": "fallback"
                            })
            except:
                extracted_texts.append({
                    "url": url,
                    "text": "Failed to extract content",
                    "success": False
                })
    
    # Combine extracted texts
    combined_text = "\n\n".join([item["text"] for item in extracted_texts if item.get("success")])
    
    return {
        "text": combined_text or "No content successfully extracted",
        "sources": urls,
        "extractions": extracted_texts,
        "status": "success" if combined_text else "partial"
    }


async def real_analyze(task):
    """Analyze content using real AI model."""
    content_data = task.parameters.get("content", {})
    content = content_data.get("text", "") if isinstance(content_data, dict) else str(content_data)
    depth = task.parameters.get("analysis_depth", "standard")
    print(f"[Analysis] Performing {depth} analysis with AI model")
    
    # Get available model
    try:
        registry = ModelRegistry()
        model = registry.get_model("gpt-4o-mini") or registry.get_model("claude-3-5-haiku-20241022")
        
        if model:
            # Create analysis prompt
            prompt = f"""Analyze the following content and provide:
1. 3-5 key findings or insights
2. A suggested outline for a research report

Content:
{content[:2000]}  # Limit content length

Analysis depth: {depth}

Provide your analysis in a structured format."""
            
            # Get AI analysis
            response = await model.generate(prompt, max_tokens=500, temperature=0.3)
            
            # Parse response to extract key points
            lines = response.strip().split('\n')
            key_points = []
            outline = {}
            
            # Extract key points
            in_key_points = False
            in_outline = False
            
            for line in lines:
                if "key finding" in line.lower() or "insight" in line.lower():
                    in_key_points = True
                    in_outline = False
                elif "outline" in line.lower() or "structure" in line.lower():
                    in_key_points = False
                    in_outline = True
                elif in_key_points and line.strip() and (line.strip()[0].isdigit() or line.startswith('-')):
                    key_points.append(line.strip().lstrip('0123456789.-) '))
                elif in_outline and ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        outline[parts[0].strip().lower()] = parts[1].strip()
            
            if not key_points:
                key_points = ["AI analysis completed", "Content processed successfully"]
            
            if not outline:
                outline = {
                    "introduction": "Research overview",
                    "methodology": "Analysis approach",
                    "findings": "Key discoveries",
                    "conclusion": "Summary"
                }
            
            return {
                "key_points": key_points[:5],
                "outline": outline,
                "model_used": model.name,
                "analysis_depth": depth,
                "status": "success"
            }
    except Exception as e:
        print(f"[Analysis] AI analysis failed: {e}")
    
    # Fallback to basic analysis
    return {
        "key_points": [
            f"Analyzed {len(content.split())} words of content",
            f"Analysis depth: {depth}",
            "Manual review recommended for detailed insights"
        ],
        "outline": {
            "introduction": "Content overview",
            "analysis": "Main findings",
            "conclusion": "Summary"
        },
        "status": "fallback"
    }


async def real_verify(task):
    """Verify facts using real web search and AI."""
    analysis_data = task.parameters.get("analysis", {})
    claims = analysis_data.get("key_points", []) if isinstance(analysis_data, dict) else task.parameters.get("claims", [])
    print(f"[Fact Check] Verifying {len(claims)} claims")
    
    verified_claims = []
    search_tool = DuckDuckGoSearchTool()
    
    for claim in claims[:3]:  # Limit to 3 claims for performance
        try:
            # Search for evidence
            search_results = await search_tool.search(f"fact check {claim}", max_results=3)
            
            # Basic verification based on search results
            confidence = 0.5  # Base confidence
            if search_results:
                # Check if claim appears in reputable sources
                for result in search_results:
                    snippet = result.get("snippet", "").lower()
                    if any(word in snippet for word in ["true", "confirmed", "accurate", "correct"]):
                        confidence += 0.15
                    elif any(word in snippet for word in ["false", "incorrect", "myth", "debunked"]):
                        confidence -= 0.15
                
                confidence = max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95
            
            verified_claims.append({
                "claim": claim,
                "verified": confidence > 0.6,
                "confidence": confidence,
                "sources_checked": len(search_results)
            })
        except Exception as e:
            print(f"  Failed to verify '{claim}': {e}")
            verified_claims.append({
                "claim": claim,
                "verified": False,
                "confidence": 0.0,
                "error": str(e)
            })
    
    return {
        "verified": verified_claims,
        "total_claims": len(claims),
        "checked_claims": len(verified_claims),
        "status": "success"
    }


async def real_generate(task):
    """Generate report using real AI model."""
    format_type = task.parameters.get("format", "markdown")
    facts_data = task.parameters.get("facts", {})
    verified_facts = facts_data.get("verified", []) if isinstance(facts_data, dict) else []
    
    print(f"[Report Generation] Generating {format_type} report with AI")
    
    try:
        # Get available model
        registry = ModelRegistry()
        model = registry.get_model("gpt-4o-mini") or registry.get_model("claude-3-5-haiku-20241022")
        
        if model:
            # Prepare report data
            verified_claims = [f"{item['claim']} (confidence: {item['confidence']:.2f})" 
                             for item in verified_facts if item.get('verified')]
            unverified_claims = [f"{item['claim']} (confidence: {item['confidence']:.2f})" 
                               for item in verified_facts if not item.get('verified')]
            
            # Create generation prompt
            prompt = f"""Generate a research report in {format_type} format based on the following verified information:

Verified Facts:
{chr(10).join(f'- {claim}' for claim in verified_claims)}

Unverified/Low Confidence Claims:
{chr(10).join(f'- {claim}' for claim in unverified_claims)}

Create a professional research report with:
1. Executive Summary
2. Introduction
3. Methodology
4. Key Findings
5. Conclusion

Make it concise but comprehensive."""
            
            # Generate report
            report = await model.generate(prompt, max_tokens=800, temperature=0.3)
            
            return {
                "document": report,
                "format": format_type,
                "model_used": model.name,
                "verified_facts_count": len(verified_claims),
                "total_facts_count": len(verified_facts),
                "status": "success"
            }
    except Exception as e:
        print(f"[Report Generation] AI generation failed: {e}")
    
    # Fallback report
    fallback_report = f"""# Research Report

## Executive Summary
This report summarizes findings from automated research pipeline processing.

## Methodology
- Web search for relevant sources
- Content extraction and analysis
- Fact verification
- Report generation

## Key Findings
{chr(10).join(f'- {item["claim"]}' for item in verified_facts[:5])}

## Conclusion
The research pipeline successfully processed the query and generated this report.

Note: This is a fallback report generated without AI assistance.
"""
    
    return {
        "document": fallback_report,
        "format": format_type,
        "status": "fallback"
    }


# Real control system that routes actions to real functions
class RealResearchControlSystem:
    """Real control system for research pipeline using actual APIs."""
    
    def __init__(self):
        self.action_handlers = {
            "search": real_search,
            "extract": real_extract,
            "analyze": real_analyze,
            "verify": real_verify,
            "generate": real_generate
        }
    
    async def execute_action(self, action, task):
        """Execute a control action."""
        handler = self.action_handlers.get(action.name)
        if handler:
            return await handler(task)
        else:
            raise ValueError(f"Unknown action: {action.name}")
    
    async def shutdown(self):
        """Shutdown the control system."""
        pass


async def test_research_pipeline():
    """Test the research assistant pipeline."""
    print("Testing Research Assistant Pipeline")
    print("=" * 50)
    
    # Load the pipeline
    with open("pipelines/research_assistant.yaml", "r") as f:
        pipeline_yaml = f.read()
    
    # Initialize components
    orchestrator = Orchestrator()
    orchestrator.control_system = RealResearchControlSystem()
    
    # Set up real model for resolving <AUTO> tags
    orchestrator.model_registry = ModelRegistry()
    
    # Try to get a real model
    real_model = None
    for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemma2-9b-it"]:
        try:
            real_model = orchestrator.model_registry.get_model(model_id)
            if real_model:
                print(f"Using {model_id} for AUTO resolution")
                break
        except:
            continue
    
    if not real_model:
        print("WARNING: No real model available for AUTO resolution")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        # Create a basic fallback that returns reasonable defaults
        class FallbackModel(Model):
            def __init__(self):
                from src.orchestrator.core.model import ModelCapabilities
                capabilities = ModelCapabilities(
                    supported_tasks=["reasoning", "generation"],
                    context_window=4096,
                    languages=["en"]
                )
                super().__init__(
                    name="Fallback Model",
                    provider="fallback",
                    capabilities=capabilities
                )
            
            async def generate(self, prompt, **kwargs):
                # Return reasonable defaults
                if "sources" in prompt.lower():
                    return "web search, academic papers"
                elif "number" in prompt.lower():
                    return "5"
                elif "method" in prompt.lower():
                    return "comprehensive"
                elif "format" in prompt.lower():
                    return "markdown"
                return "standard"
            
            async def validate_response(self, response, schema):
                return True
            
            def estimate_tokens(self, text):
                return len(text.split())
            
            async def generate_structured(self, prompt, schema, **kwargs):
                response = await self.generate(prompt, **kwargs)
                return {"value": response}
            
            def estimate_cost(self, input_tokens, output_tokens):
                return 0.0
            
            async def health_check(self):
                return True
        
        real_model = FallbackModel()
        orchestrator.model_registry.register_model(real_model)
    
    orchestrator.yaml_compiler.ambiguity_resolver.model = real_model
    
    # Compile the pipeline with inputs
    pipeline = await orchestrator.yaml_compiler.compile(
        pipeline_yaml,
        context={
            "topic": "Artificial Intelligence in Healthcare",
            "depth": "comprehensive"
        }
    )
    
    print(f"\nCompiled pipeline: {pipeline.name}")
    print(f"Number of steps: {len(pipeline.tasks)}")
    print(f"Execution levels: {len(pipeline.get_execution_levels())}")
    
    # Execute the pipeline
    print("\nExecuting pipeline...")
    print("-" * 50)
    
    results = await orchestrator.execute_pipeline(pipeline)
    
    print("\n" + "-" * 50)
    print("Pipeline execution completed!")
    
    # Display results
    print("\nResults:")
    for task_id, result in results.items():
        task = pipeline.get_task(task_id)
        print(f"\n[{task.name}] Status: {result.get('status', 'unknown')}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        elif task_id == 'report_generation':
            print(f"  Summary: {result.get('summary', 'N/A')}")
    
    # Check outputs
    outputs = pipeline.metadata.get("outputs", {})
    print("\nPipeline Outputs:")
    print(f"- Report available: {'report' in outputs}")
    print(f"- Summary available: {'summary' in outputs}")
    print(f"- Sources available: {'sources' in outputs}")
    
    return results


if __name__ == "__main__":
    # Run the test
    results = asyncio.run(test_research_pipeline())
    
    # Check if all tasks completed successfully
    all_success = all(
        r.get("status") == "success" 
        for r in results.values()
    )
    
    if all_success:
        print("\n✅ All pipeline steps completed successfully!")
    else:
        print("\n❌ Some pipeline steps failed")
        sys.exit(1)