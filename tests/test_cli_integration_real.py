"""
Real CLI Integration Tests - Issue #153 Phase 3

Tests actual CLI usage patterns with dependency fixes applied to validate
template rendering works in real user workflows.
"""

import pytest
import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from functools import wraps


def cost_controlled_test(timeout=180):
    """Decorator for cost-controlled CLI tests."""
    def decorator(test_func):
        @wraps(test_func)
        async def wrapper(*args, **kwargs):
            try:
                result = await asyncio.wait_for(
                    test_func(*args, **kwargs), 
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                pytest.fail(f"CLI test timed out after {timeout} seconds")
        return wrapper
    return decorator


@pytest.fixture(scope="session")
def api_keys_available():
    """Check if API keys are available for real testing."""
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
    available_keys = [key for key in required_keys if os.getenv(key)]
    
    if not available_keys:
        pytest.skip(f"Skipping CLI integration tests - no API keys found. Set one of: {required_keys}")
    
    print(f"Using real APIs with keys: {available_keys}")
    return available_keys


class TestCLIIntegrationReal:
    """Test CLI integration with real APIs and dependency fixes."""
    
    @cost_controlled_test(timeout=240)
    async def test_research_basic_cli_fixed_dependencies(self, api_keys_available):
        """Test research_basic.yaml via CLI with dependency fixes applied."""
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Test the basic research pipeline with CLI
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py",
                "examples/research_basic.yaml", 
                "-i", "topic=artificial intelligence ethics",
                "-i", f"output_path={output_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            stdout, stderr = await proc.communicate()
            
            # Validate CLI execution success
            if proc.returncode != 0:
                error_msg = f"CLI failed with return code {proc.returncode}\n"
                error_msg += f"STDOUT: {stdout.decode()}\n"
                error_msg += f"STDERR: {stderr.decode()}"
                pytest.fail(error_msg)
            
            print(f"CLI execution successful. Output: {stdout.decode()[:500]}")
            
            # Validate output files were created
            output_files = list(output_path.glob("*.md"))
            assert len(output_files) > 0, f"No output files found in {output_path}"
            
            # Validate template rendering in all output files
            unrendered_templates = []
            for file_path in output_files:
                content = file_path.read_text()
                
                # Check for unrendered templates
                if '{{' in content or '{%' in content:
                    unrendered_templates.append((file_path.name, content[:200]))
                
                # Validate expected content
                assert 'artificial intelligence ethics' in content.lower()
                assert len(content) > 100, f"Content too short in {file_path.name}"
                
            if unrendered_templates:
                error_msg = "Found unrendered templates:\n"
                for filename, content_sample in unrendered_templates:
                    error_msg += f"  {filename}: {content_sample}\n"
                pytest.fail(error_msg)
            
            print(f"✅ All {len(output_files)} output files have properly rendered templates")

    @cost_controlled_test(timeout=300)  
    async def test_research_advanced_tools_cli_fixed(self, api_keys_available):
        """Test research_advanced_tools.yaml with dependency fixes via CLI."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            
            # Test advanced research pipeline
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py",
                "examples/research_advanced_tools.yaml",
                "-i", "topic=renewable energy storage",
                "-i", f"output_path={output_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                pytest.fail(f"Advanced research CLI failed: {stderr.decode()}")
            
            # Validate comprehensive output
            output_files = list(output_path.glob("*.md"))
            assert len(output_files) > 0, "No output files created"
            
            # Check for complex template rendering
            for file_path in output_files:
                content = file_path.read_text()
                
                # Ensure no template artifacts remain
                assert '{{' not in content, f"Unrendered templates in {file_path.name}"
                assert '{%' not in content, f"Unrendered control structures in {file_path.name}"
                
                # Validate rich content generation
                assert 'renewable energy storage' in content.lower()
                
                # Check for evidence of multi-step processing
                if 'analysis' in content.lower():
                    print(f"✅ {file_path.name} contains analysis content")
                    
            print(f"✅ Advanced research pipeline generated {len(output_files)} files successfully")

    @cost_controlled_test(timeout=200)
    async def test_concurrent_cli_execution(self, api_keys_available):
        """Test multiple CLI pipelines running concurrently."""
        
        async def run_pipeline_cli(pipeline_name, topic, output_subdir):
            """Run single pipeline via CLI."""
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / output_subdir
                output_path.mkdir(parents=True, exist_ok=True)
                
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "scripts/run_pipeline.py",
                    f"examples/{pipeline_name}",
                    "-i", f"topic={topic}",
                    "-i", f"output_path={output_path}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, 'PYTHONPATH': 'src'},
                    cwd=Path.cwd()
                )
                
                stdout, stderr = await proc.communicate()
                
                return {
                    'pipeline': pipeline_name,
                    'returncode': proc.returncode,
                    'output_files': list(output_path.glob("*.md")) if output_path.exists() else [],
                    'stdout': stdout.decode()[:1000],  # First 1000 chars
                    'stderr': stderr.decode()[:1000] if stderr else ""
                }
        
        # Run multiple pipelines concurrently
        concurrent_tests = [
            ("research_minimal.yaml", "quantum computing", "quantum"),
            ("research_basic.yaml", "machine learning", "ml"),
            ("test_validation_pipeline.yaml", "data science", "datasci"),
        ]
        
        # Execute all pipelines concurrently
        tasks = [
            run_pipeline_cli(pipeline, topic, subdir) 
            for pipeline, topic, subdir in concurrent_tests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate all concurrent executions  
        successful_runs = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Concurrent execution failed with exception: {result}")
                continue
                
            pipeline_name = result['pipeline']
            
            if result['returncode'] == 0:
                successful_runs += 1
                print(f"✅ {pipeline_name} executed successfully")
                
                # Validate output files
                if result['output_files']:
                    print(f"   Generated {len(result['output_files'])} output files")
                else:
                    print(f"   ⚠️ No output files found for {pipeline_name}")
            else:
                print(f"❌ {pipeline_name} failed with return code {result['returncode']}")
                if result['stderr']:
                    print(f"   Error: {result['stderr'][:200]}")
        
        # Require at least 2 successful concurrent executions
        assert successful_runs >= 2, f"Only {successful_runs} pipelines succeeded concurrently"
        print(f"✅ {successful_runs}/{len(concurrent_tests)} pipelines executed successfully concurrently")

    @cost_controlled_test(timeout=180)
    async def test_cli_with_invalid_pipeline_graceful_handling(self, api_keys_available):
        """Test CLI gracefully handles pipelines with remaining template issues."""
        
        # Create a test pipeline with intentionally missing dependencies
        broken_pipeline_content = """
name: Intentionally Broken Pipeline
parameters:
  test_topic: "testing"
steps:
  - id: generate_content
    action: generate_text
    parameters:
      prompt: "Write about {{ test_topic }}"
      max_tokens: 50
      
  - id: save_broken  # Missing dependencies intentionally
    tool: filesystem
    action: write
    parameters:
      path: "/tmp/broken_test.md"  
      content: |
        # Test Results
        Content: {{ generate_content.result }}
        Topic: {{ test_topic }}
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write broken pipeline
            broken_pipeline_path = Path(temp_dir) / "broken.yaml"
            broken_pipeline_path.write_text(broken_pipeline_content)
            
            # Run the broken pipeline
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py",
                str(broken_pipeline_path),
                "-i", "test_topic=broken pipeline test",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            stdout, stderr = await proc.communicate()
            
            # Pipeline should complete but produce unrendered templates
            assert proc.returncode == 0, "Pipeline should complete even with template issues"
            
            # Check the output file
            output_file = Path("/tmp/broken_test.md")
            if output_file.exists():
                content = output_file.read_text()
                
                # Should have rendered pipeline parameters but not step results
                assert "broken pipeline test" in content  # Parameter rendered
                assert "{{ generate_content.result }}" in content  # Step result NOT rendered
                
                print("✅ CLI gracefully handled pipeline with missing dependencies")
                
                # Cleanup
                output_file.unlink()
            else:
                pytest.fail("Output file was not created for broken pipeline")

    async def test_cli_help_and_validation(self, api_keys_available):
        """Test CLI help and input validation."""
        
        # Test help command
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "scripts/run_pipeline.py", "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        assert proc.returncode == 0, "Help command should succeed"
        help_output = stdout.decode()
        
        # Validate help content
        assert "-i" in help_output or "--input" in help_output, "Help should mention input flag"
        assert "pipeline" in help_output.lower(), "Help should mention pipeline"
        
        print("✅ CLI help command works correctly")
        
        # Test invalid pipeline path
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "scripts/run_pipeline.py", 
            "nonexistent_pipeline.yaml",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, 'PYTHONPATH': 'src'}
        )
        
        stdout, stderr = await proc.communicate()
        
        # Should fail gracefully
        assert proc.returncode != 0, "Should fail for nonexistent pipeline"
        error_output = (stdout.decode() + stderr.decode()).lower()
        assert "not found" in error_output or "error" in error_output
        
        print("✅ CLI properly handles invalid pipeline paths")


class TestRealUserWorkflows:
    """Test complete user workflows end-to-end."""
    
    @cost_controlled_test(timeout=300)
    async def test_new_user_onboarding_workflow(self, api_keys_available):
        """Test complete new user onboarding experience."""
        
        with tempfile.TemporaryDirectory() as temp_workspace:
            workspace = Path(temp_workspace)
            
            # Step 1: New user tries research_minimal.yaml (simplest case)
            print("🔄 Step 1: Testing research_minimal.yaml...")
            
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py",
                "examples/research_minimal.yaml",
                "-i", "topic=getting started with orchestrator",
                "-i", f"output_path={workspace / 'minimal'}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            stdout, stderr = await proc.communicate()
            assert proc.returncode == 0, f"Minimal research failed: {stderr.decode()}"
            
            minimal_files = list((workspace / 'minimal').glob("*.md"))
            assert len(minimal_files) > 0, "No output files from minimal research"
            
            # Validate content quality
            for file_path in minimal_files:
                content = file_path.read_text()
                assert '{{' not in content, f"Unrendered templates in {file_path.name}"
                assert 'getting started' in content.lower()
                
            print(f"✅ Step 1 complete: Generated {len(minimal_files)} files")
            
            # Step 2: User progresses to research_basic.yaml
            print("🔄 Step 2: Testing research_basic.yaml...")
            
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py",
                "examples/research_basic.yaml",
                "-i", "topic=advanced orchestrator features",
                "-i", f"output_path={workspace / 'basic'}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            stdout, stderr = await proc.communicate()
            assert proc.returncode == 0, f"Basic research failed: {stderr.decode()}"
            
            basic_files = list((workspace / 'basic').glob("*.md"))
            assert len(basic_files) > 0, "No output files from basic research"
            
            print(f"✅ Step 2 complete: Generated {len(basic_files)} files")
            
            # Step 3: Advanced user tries research_advanced_tools.yaml
            print("🔄 Step 3: Testing research_advanced_tools.yaml...")
            
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py", 
                "examples/research_advanced_tools.yaml",
                "-i", "topic=complex workflow automation",
                "-i", f"output_path={workspace / 'advanced'}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                advanced_files = list((workspace / 'advanced').glob("*.md"))
                print(f"✅ Step 3 complete: Generated {len(advanced_files)} files")
            else:
                print(f"⚠️ Step 3 advanced pipeline had issues: {stderr.decode()[:200]}")
                # Don't fail the test - advanced tools might have external dependencies
            
            # Validate overall user experience
            total_files = len(minimal_files) + len(basic_files)
            assert total_files >= 2, f"User workflow only generated {total_files} total files"
            
            print(f"✅ Complete user workflow successful: {total_files} files generated across progression")

    @cost_controlled_test(timeout=240)
    async def test_real_research_workflow_validation(self, api_keys_available):
        """Test realistic research workflow that users actually perform."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            research_dir = Path(temp_dir) / "research_project"
            research_dir.mkdir()
            
            # User wants to research a specific topic comprehensively
            research_topic = "sustainable urban transportation"
            
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py",
                "examples/research_basic.yaml",
                "-i", f"topic={research_topic}",
                "-i", f"output_path={research_dir}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            stdout, stderr = await proc.communicate()
            
            assert proc.returncode == 0, f"Research workflow failed: {stderr.decode()}"
            
            # Validate research deliverables
            output_files = list(research_dir.glob("*.md"))
            assert len(output_files) > 0, "No research deliverables created"
            
            research_quality_checks = {
                'topic_coverage': False,
                'structured_content': False, 
                'no_template_artifacts': True,
                'sufficient_length': False
            }
            
            total_content_length = 0
            
            for file_path in output_files:
                content = file_path.read_text()
                total_content_length += len(content)
                
                # Check topic coverage
                if research_topic.lower() in content.lower():
                    research_quality_checks['topic_coverage'] = True
                    
                # Check for structured content
                if any(marker in content for marker in ['#', '##', '###', '*', '-', '1.']):
                    research_quality_checks['structured_content'] = True
                    
                # Check for template artifacts
                if '{{' in content or '{%' in content:
                    research_quality_checks['no_template_artifacts'] = False
                    print(f"❌ Found template artifacts in {file_path.name}")
            
            # Check sufficient content length
            if total_content_length > 1000:  # At least 1000 chars total
                research_quality_checks['sufficient_length'] = True
                
            # Validate research quality
            failed_checks = [check for check, passed in research_quality_checks.items() if not passed]
            
            assert len(failed_checks) == 0, f"Research quality checks failed: {failed_checks}"
            
            print(f"✅ Research workflow validation complete:")
            print(f"   - Files generated: {len(output_files)}")
            print(f"   - Total content: {total_content_length} characters")
            print(f"   - Quality checks: All passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])