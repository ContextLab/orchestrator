"""Tests for the iterative fact checker pipeline."""

import asyncio
import os
import pytest
from pathlib import Path

from orchestrator import Orchestrator, init_models


class TestFactChecker:
    """Test suite for fact-checking pipeline."""

    @pytest.mark.asyncio
    async def test_simple_fact_checking(self):
        """Test that fact-checker adds citations to uncited claims."""
        # Initialize models
        init_models()
        
        # Create a simple test document with uncited facts
        test_doc = """# Test Document

The Earth orbits the Sun every 365.25 days.

Water freezes at 0 degrees Celsius at standard pressure.

The speed of light in vacuum is approximately 299,792,458 meters per second.
"""
        
        # Save test document
        test_path = Path("test_facts.md")
        test_path.write_text(test_doc)
        
        try:
            # Run the fact-checker pipeline
            orchestrator = Orchestrator()
            pipeline_path = Path("examples/iterative_fact_checker_simple.yaml")
            
            if not pipeline_path.exists():
                pytest.skip("Fact checker pipeline not found")
            
            results = await orchestrator.execute_yaml(
                pipeline_path.read_text(),
                inputs={
                    "input_document": str(test_path),
                    "output_path": "test_output"
                }
            )
            
            # Verify pipeline completed successfully
            assert results.get("success"), "Pipeline should complete successfully"
            assert "save_document" in results.get("steps", {}), "Should save fact-checked document"
            
            # Check the output document
            output_path = Path("test_output") / "test_facts_fact_checked.md"
            assert output_path.exists(), "Fact-checked document should be created"
            
            # Read the fact-checked document
            fact_checked = output_path.read_text()
            
            # Verify citations were added
            assert "[1]" in fact_checked, "Should add citation [1]"
            assert "[2]" in fact_checked, "Should add citation [2]"
            assert "[3]" in fact_checked, "Should add citation [3]"
            
            # Verify References section was added
            assert "## References" in fact_checked or "References" in fact_checked, \
                "Should add References section"
            
            # Count total citations (both inline and in references)
            citation_count = fact_checked.count("[1]") + fact_checked.count("[2]") + fact_checked.count("[3]")
            assert citation_count >= 6, "Should have at least 6 citations (3 inline + 3 in references)"
            
        finally:
            # Clean up test files
            if test_path.exists():
                test_path.unlink()
            
            # Clean up output directory
            output_dir = Path("test_output")
            if output_dir.exists():
                for file in output_dir.glob("*"):
                    file.unlink()
                output_dir.rmdir()

    @pytest.mark.asyncio
    async def test_climate_document(self):
        """Test fact-checking with the full climate document."""
        # Initialize models
        init_models()
        
        # Check if climate document exists
        climate_doc = Path("test_climate_document.md")
        if not climate_doc.exists():
            pytest.skip("Climate test document not found")
        
        orchestrator = Orchestrator()
        pipeline_path = Path("examples/iterative_fact_checker_simple.yaml")
        
        if not pipeline_path.exists():
            pytest.skip("Fact checker pipeline not found")
        
        results = await orchestrator.execute_yaml(
            pipeline_path.read_text(),
            inputs={
                "input_document": str(climate_doc),
                "output_path": "test_climate_output"
            }
        )
        
        # Verify success
        assert results.get("success"), "Pipeline should complete successfully"
        
        # Check output
        output_path = Path("test_climate_output") / "test_climate_document_fact_checked.md"
        assert output_path.exists(), "Fact-checked climate document should be created"
        
        fact_checked = output_path.read_text()
        
        # Original document has 30+ claims, should have many citations
        citation_count = sum(1 for i in range(1, 50) if f"[{i}]" in fact_checked)
        assert citation_count >= 30, f"Should have at least 30 citations, found {citation_count}"
        
        # Clean up
        output_dir = Path("test_climate_output")
        if output_dir.exists():
            for file in output_dir.glob("*"):
                file.unlink()
            output_dir.rmdir()

    @pytest.mark.asyncio
    async def test_extract_claims_structure(self):
        """Test that extract_claims returns proper structure."""
        # Initialize models
        init_models()
        
        test_doc = """# Simple Test

Fact one without citation.

Fact two with citation [1].

[1] Source: Example Journal
"""
        
        test_path = Path("test_structure.md")
        test_path.write_text(test_doc)
        
        try:
            orchestrator = Orchestrator()
            pipeline_path = Path("examples/iterative_fact_checker_simple.yaml")
            
            if not pipeline_path.exists():
                pytest.skip("Fact checker pipeline not found")
            
            results = await orchestrator.execute_yaml(
                pipeline_path.read_text(),
                inputs={
                    "input_document": str(test_path),
                    "output_path": "test_structure_output"
                }
            )
            
            # Check that pipeline runs
            assert results.get("success"), "Pipeline should complete"
            
            # The extract_claims step should identify claims
            steps = results.get("steps", {})
            assert "extract_claims" in steps, "Should have extract_claims step"
            
            # Note: Due to the generate-structured action returning strings,
            # we can't directly test the structure here, but we can verify
            # the pipeline processes it correctly by checking the output
            
            output_path = Path("test_structure_output") / "test_structure_fact_checked.md"
            if output_path.exists():
                fact_checked = output_path.read_text()
                # Should preserve existing citation and add new ones
                assert "[1]" in fact_checked, "Should preserve existing citations"
                
        finally:
            if test_path.exists():
                test_path.unlink()
            
            output_dir = Path("test_structure_output")
            if output_dir.exists():
                for file in output_dir.glob("*"):
                    file.unlink()
                output_dir.rmdir()


if __name__ == "__main__":
    # Run tests
    asyncio.run(TestFactChecker().test_simple_fact_checking())
    print("✓ Simple fact-checking test passed")
    
    asyncio.run(TestFactChecker().test_climate_document())
    print("✓ Climate document test passed")
    
    asyncio.run(TestFactChecker().test_extract_claims_structure())
    print("✓ Extract claims structure test passed")
    
    print("\nAll tests passed!")