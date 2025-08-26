#!/usr/bin/env python3
"""Test MCP integration pipeline with multiple queries and display results."""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import Orchestrator, init_models


async def run_mcp_search(query: str):
    """Run MCP integration pipeline for a single query."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    # Initialize models
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Load pipeline
    pipeline_path = Path(__file__).parent.parent / "examples/mcp_integration_pipeline.yaml"
    yaml_content = pipeline_path.read_text()
    
    # Run pipeline
    try:
        results = await orchestrator.execute_yaml(
            yaml_content,
            {"search_query": query}
        )
        
        # Extract search results
        if results and "steps" in results:
            search_results = results["steps"].get("search_web", {}).get("result", {})
            
            if search_results and "results" in search_results:
                print(f"\nFound {search_results.get('total', 0)} results:")
                print("-" * 40)
                
                for i, result in enumerate(search_results["results"], 1):
                    print(f"\n{i}. {result.get('title', 'No title')}")
                    print(f"   URL: {result.get('url', 'No URL')}")
                    print(f"   Snippet: {result.get('snippet', 'No snippet')[:150]}...")
                
                # Save properly formatted results
                output_dir = Path("examples/outputs/mcp_integration")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create filename from query
                safe_query = query[:25].replace(" ", "-").replace("/", "-").lower()
                filename = f"{safe_query}_results.json"
                output_file = output_dir / filename
                
                # Save formatted results
                formatted_data = {
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "total_results": search_results.get("total", 0),
                    "results": search_results.get("results", []),
                    "pipeline": "mcp_integration"
                }
                
                with open(output_file, "w") as f:
                    json.dump(formatted_data, f, indent=2)
                
                print(f"\n✅ Results saved to: {output_file}")
            else:
                print("❌ No search results returned")
        else:
            print("❌ Pipeline execution failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run multiple test queries."""
    queries = [
        "machine learning algorithms",
        "quantum computing breakthroughs 2024",
        "climate change renewable energy solutions",
        "artificial intelligence ethics governance",
        "Python web frameworks Django Flask",
        "blockchain cryptocurrency applications",
        "space exploration Mars colonization",
        "cybersecurity best practices 2024",
        "sustainable agriculture techniques",
        "medical AI diagnosis systems"
    ]
    
    print("\n" + "="*60)
    print("MCP Integration Pipeline Test Suite")
    print("Testing with real DuckDuckGo searches")
    print("="*60)
    
    for query in queries:
        await run_mcp_search(query)
        await asyncio.sleep(1)  # Be polite to the search service
    
    print("\n" + "="*60)
    print("All queries completed!")
    print("="*60)
    
    # List all output files
    output_dir = Path("examples/outputs/mcp_integration")
    if output_dir.exists():
        files = list(output_dir.glob("*.json"))
        print(f"\n📁 Generated {len(files)} output files in {output_dir}")
        for f in sorted(files):
            print(f"   - {f.name}")


if __name__ == "__main__":
    asyncio.run(main())