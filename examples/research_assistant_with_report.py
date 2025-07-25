"""Research Assistant with Report Generation.

This module provides a research assistant that conducts web searches,
analyzes results, and generates comprehensive PDF reports.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from orchestrator.tools.web_tools import WebSearchTool
from orchestrator.tools.report_tools import ReportGeneratorTool, PDFCompilerTool


class ResearchAssistantWithReport:
    """Research assistant that generates comprehensive reports."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize research assistant with configuration."""
        self.config = config
        self.web_search_tool = WebSearchTool(config)
        self.report_generator = ReportGeneratorTool()
        self.pdf_compiler = PDFCompilerTool()

    async def conduct_research_with_report(
        self,
        query: str,
        context: Optional[str] = None,
        output_dir: str = ".",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """Conduct research and generate a comprehensive report.
        
        Args:
            query: The research query
            context: Additional context for the research
            output_dir: Directory to save the report
            max_results: Maximum number of search results
            
        Returns:
            Dictionary with report metadata and paths
        """
        try:
            # Step 1: Web search
            search_results = await self.web_search_tool.execute(
                query=query,
                max_results=max_results
            )
            
            # Step 2: Extract and analyze content
            extraction_results = self._extract_content(search_results)
            
            # Step 3: Generate findings and recommendations
            findings = self._generate_findings(search_results, extraction_results)
            recommendations = self._generate_recommendations(findings)
            
            # Step 4: Calculate quality score
            quality_score = self._calculate_quality_score(
                search_results, extraction_results
            )
            
            # Step 5: Generate markdown report
            report_result = await self.report_generator.execute(
                title=f"Research Report: {query}",
                query=query,
                context=context,
                search_results=search_results,
                extraction_results=extraction_results,
                findings=findings,
                recommendations=recommendations,
                quality_score=quality_score
            )
            
            if not report_result["success"]:
                return {
                    "success": False,
                    "error": report_result.get("error", "Failed to generate report")
                }
            
            # Step 6: Save markdown report
            markdown_path = Path(output_dir) / f"research_report_{query[:30].replace(' ', '_')}.md"
            with open(markdown_path, "w") as f:
                f.write(report_result["markdown"])
            
            # Step 7: Generate PDF (optional)
            pdf_path = None
            pdf_result = None
            try:
                pdf_path = Path(output_dir) / f"research_report_{query[:30].replace(' ', '_')}.pdf"
                pdf_result = await self.pdf_compiler.execute(
                    markdown_content=report_result["markdown"],
                    output_path=str(pdf_path),
                    title=f"Research Report: {query}",
                    author="Research Assistant",
                    install_if_missing=True
                )
            except Exception as e:
                # PDF generation is optional
                print(f"PDF generation failed: {e}")
            
            # Return results
            result = {
                "success": True,
                "markdown_path": str(markdown_path),
                "word_count": report_result["word_count"],
                "quality_score": quality_score,
                "findings_count": len(findings),
                "recommendations_count": len(recommendations)
            }
            
            if pdf_result and pdf_result.get("success"):
                result["pdf_path"] = str(pdf_path)
                result["file_size"] = pdf_result.get("file_size", 0)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_content(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from search results."""
        # Simulate content extraction
        total_words = 0
        extracted_text = []
        
        for result in search_results.get("results", []):
            snippet = result.get("snippet", "")
            extracted_text.append(snippet)
            total_words += len(snippet.split())
        
        return {
            "success": True,
            "text": " ".join(extracted_text),
            "word_count": total_words,
            "sources": len(search_results.get("results", []))
        }
    
    def _generate_findings(
        self, 
        search_results: Dict[str, Any], 
        extraction_results: Dict[str, Any]
    ) -> List[str]:
        """Generate findings from search and extraction results."""
        findings = []
        
        results = search_results.get("results", [])
        if results:
            findings.append(f"Found {len(results)} relevant sources for the query")
            
            # Analyze source types
            domains = set()
            for result in results:
                url = result.get("url", "")
                if url:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain:
                        domains.add(domain)
            
            if len(domains) > 1:
                findings.append(f"Information gathered from {len(domains)} different domains")
            
            # Check relevance scores
            high_relevance = sum(1 for r in results if r.get("relevance", 0) > 0.8)
            if high_relevance > 0:
                findings.append(f"{high_relevance} sources have high relevance scores")
            
            # Content analysis
            word_count = extraction_results.get("word_count", 0)
            if word_count > 1000:
                findings.append("Substantial content available for comprehensive analysis")
            elif word_count > 500:
                findings.append("Moderate amount of content found for analysis")
            else:
                findings.append("Limited content available, consider expanding search")
        
        return findings
    
    def _generate_recommendations(self, findings: List[str]) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        # Base recommendations on findings
        for finding in findings:
            if "Limited content" in finding:
                recommendations.append("Consider broadening search terms or using alternative sources")
            elif "high relevance" in finding:
                recommendations.append("Focus on high-relevance sources for detailed analysis")
            elif "different domains" in finding:
                recommendations.append("Cross-reference information across multiple sources for accuracy")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Continue monitoring this topic for new developments")
            recommendations.append("Consider conducting follow-up research on specific subtopics")
        
        return recommendations
    
    def _calculate_quality_score(
        self, 
        search_results: Dict[str, Any], 
        extraction_results: Dict[str, Any]
    ) -> float:
        """Calculate quality score for the research."""
        score = 0.0
        
        # Factor 1: Number of results (max 0.3)
        num_results = len(search_results.get("results", []))
        results_score = min(num_results / 10, 1.0) * 0.3
        score += results_score
        
        # Factor 2: Content volume (max 0.3)
        word_count = extraction_results.get("word_count", 0)
        content_score = min(word_count / 2000, 1.0) * 0.3
        score += content_score
        
        # Factor 3: Average relevance (max 0.2)
        results = search_results.get("results", [])
        if results:
            avg_relevance = sum(r.get("relevance", 0) for r in results) / len(results)
            score += avg_relevance * 0.2
        
        # Factor 4: Source diversity (max 0.2)
        domains = set()
        for result in results:
            url = result.get("url", "")
            if url:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc
                if domain:
                    domains.add(domain)
        
        diversity_score = min(len(domains) / 5, 1.0) * 0.2
        score += diversity_score
        
        return round(score, 2)


# Example usage
async def main():
    """Example usage of the research assistant."""
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    assistant = ResearchAssistantWithReport(config)
    
    result = await assistant.conduct_research_with_report(
        query="Python asyncio best practices",
        context="Focus on performance optimization",
        output_dir="./reports"
    )
    
    if result["success"]:
        print(f"Report generated successfully!")
        print(f"Markdown: {result['markdown_path']}")
        print(f"Word count: {result['word_count']}")
        print(f"Quality score: {result['quality_score']}")
        if result.get("pdf_path"):
            print(f"PDF: {result['pdf_path']}")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())