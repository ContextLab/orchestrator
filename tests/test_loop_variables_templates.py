"""Tests for while loop variable availability in templates (Issue #219).

NO MOCKS OR SIMULATIONS - All tests use real file operations and API calls.
"""

import asyncio
import pytest
from pathlib import Path
import shutil
import tempfile
import json
import yaml
from typing import Dict, Any

from src.orchestrator.orchestrator import Orchestrator


class TestWhileLoopVariableTemplates:
    """Test while loop variables are available in templates."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance with test config."""
        orch = Orchestrator()
        yield orch
        # Cleanup
        await orch.shutdown()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_path = tempfile.mkdtemp(prefix="test_loop_vars_")
        yield temp_path
        # Cleanup
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_basic_while_loop_iteration_variables(self, orchestrator, temp_dir):
        """Test that $iteration and iteration are available in templates."""
        # Create pipeline with while loop using iteration in templates
        pipeline_yaml = f"""
name: test_iteration_variables
description: Test iteration variables in while loop templates

tasks:
  - id: test_loop
    while: "{{{{ iteration < 3 }}}}"
    max_iterations: 3
    steps:
      - id: save_file
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/iteration_{{{{ iteration }}}}.txt"
          content: "This is iteration {{{{ iteration }}}}"
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_pipeline.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Verify files were created with correct names and content
        for i in range(3):
            file_path = Path(temp_dir) / f"iteration_{i}.txt"
            assert file_path.exists(), f"File iteration_{i}.txt should exist"
            
            content = file_path.read_text()
            assert content == f"This is iteration {i}", f"Content should match iteration {i}"
    
    @pytest.mark.asyncio
    async def test_conditional_logic_with_iteration(self, orchestrator, temp_dir):
        """Test conditional logic based on iteration number."""
        pipeline_yaml = f"""
name: test_conditional_iteration
description: Test conditional logic with iteration variables

tasks:
  - id: conditional_loop
    while: "true"
    max_iterations: 5
    steps:
      - id: save_data
        tool: filesystem
        action: write
        parameters:
          path: |
            {{% if iteration == 0 %}}
            {temp_dir}/initial.txt
            {{% else %}}
            {temp_dir}/iteration_{{{{ iteration }}}}.txt
            {{% endif %}}
          content: "Iteration {{{{ iteration }}}} data"
      - id: check_stop
        action: evaluate_condition
        parameters:
          condition: "{{{{ iteration >= 4 }}}}"
          break_on_true: true
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_conditional.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Verify initial.txt exists with iteration 0
        initial_file = Path(temp_dir) / "initial.txt"
        assert initial_file.exists(), "initial.txt should exist for iteration 0"
        assert initial_file.read_text() == "Iteration 0 data"
        
        # Verify other iteration files exist
        for i in range(1, 5):
            file_path = Path(temp_dir) / f"iteration_{i}.txt"
            assert file_path.exists(), f"iteration_{i}.txt should exist"
            assert file_path.read_text() == f"Iteration {i} data"
    
    @pytest.mark.asyncio
    async def test_loop_state_persistence(self, orchestrator, temp_dir):
        """Test that loop_state is maintained across iterations."""
        pipeline_yaml = f"""
name: test_loop_state
description: Test loop state persistence across iterations

tasks:
  - id: stateful_loop
    while: "{{{{ (loop_state.counter | default(0)) < 3 }}}}"
    max_iterations: 5
    steps:
      - id: increment_counter
        action: python_executor
        parameters:
          code: |
            counter = {{{{ loop_state.counter | default(0) }}}} + 1
            print(counter)
      - id: save_state
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/state_{{{{ iteration }}}}.json"
          content: |
            {{
              "iteration": {{{{ iteration }}}},
              "counter": {{{{ increment_counter.result | default(1) }}}},
              "timestamp": "{{{{ now() }}}}"
            }}
      - id: update_loop_state
        action: update_loop_state
        parameters:
          counter: "{{{{ increment_counter.result }}}}"
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_state.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Verify state files were created and contain correct data
        for i in range(3):
            file_path = Path(temp_dir) / f"state_{i}.json"
            assert file_path.exists(), f"state_{i}.json should exist"
            
            data = json.loads(file_path.read_text())
            assert data["iteration"] == i, f"Iteration should be {i}"
            assert data["counter"] == i + 1, f"Counter should be {i + 1}"
            assert "timestamp" in data, "Timestamp should be present"
    
    @pytest.mark.asyncio
    async def test_nested_loops_with_iteration_variables(self, orchestrator, temp_dir):
        """Test nested while loops with multiple iteration variables."""
        pipeline_yaml = f"""
name: test_nested_loops
description: Test nested loops with iteration variables

tasks:
  - id: outer_loop
    while: "{{{{ iteration < 2 }}}}"
    max_iterations: 2
    loop_name: outer
    steps:
      - id: inner_loop
        while: "{{{{ iteration < 3 }}}}"
        max_iterations: 3
        loop_name: inner
        steps:
          - id: save_nested
            tool: filesystem
            action: write
            parameters:
              path: "{temp_dir}/outer_{{{{ outer.iteration }}}}_inner_{{{{ inner.iteration }}}}.txt"
              content: "Outer: {{{{ outer.iteration }}}}, Inner: {{{{ inner.iteration }}}}"
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_nested.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Verify all nested files were created
        for outer in range(2):
            for inner in range(3):
                file_path = Path(temp_dir) / f"outer_{outer}_inner_{inner}.txt"
                assert file_path.exists(), f"outer_{outer}_inner_{inner}.txt should exist"
                
                content = file_path.read_text()
                expected = f"Outer: {outer}, Inner: {inner}"
                assert content == expected, f"Content should be '{expected}'"
    
    @pytest.mark.asyncio
    async def test_complex_templates_with_iteration(self, orchestrator, temp_dir):
        """Test complex template expressions using iteration."""
        pipeline_yaml = f"""
name: test_complex_templates
description: Test complex template expressions with iteration

parameters:
  base_name: "test"
  quality_threshold: 0.8

tasks:
  - id: quality_loop
    while: "{{{{ (quality_score | default(0.0)) < parameters.quality_threshold }}}}"
    max_iterations: 5
    steps:
      - id: generate_quality
        action: python_executor
        parameters:
          code: |
            import random
            # Simulate quality improving with iterations
            quality = 0.2 + ({{{{ iteration }}}} * 0.2) + random.random() * 0.1
            print(min(quality, 1.0))
      
      - id: save_results
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/{{{{ parameters.base_name }}}}_{{{{ 'initial' if iteration == 0 else 'iteration_' + (iteration | string) }}}}.json"
          content: |
            {{
              "iteration": {{{{ iteration }}}},
              "quality": {{{{ generate_quality.result }}}},
              "is_first": {{{{ 'true' if iteration == 0 else 'false' }}}},
              "is_final": {{{{ 'true' if generate_quality.result >= parameters.quality_threshold else 'false' }}}},
              "message": "{{{{ 'Starting quality check' if iteration == 0 else 'Iteration ' + (iteration | string) + ' quality: ' + (generate_quality.result | string) }}}}"
            }}
      
      - id: update_quality
        action: update_loop_state
        parameters:
          quality_score: "{{{{ generate_quality.result }}}}"
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_complex.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Check that files were created with correct naming pattern
        files = list(Path(temp_dir).glob("test_*.json"))
        assert len(files) > 0, "At least one result file should be created"
        
        # Verify initial file
        initial_file = Path(temp_dir) / "test_initial.json"
        if initial_file.exists():
            data = json.loads(initial_file.read_text())
            assert data["iteration"] == 0
            assert data["is_first"] == "true" or data["is_first"] is True
            assert "Starting quality check" in data["message"]
        
        # Verify iteration files have correct structure
        for file_path in files:
            if "iteration_" in file_path.name:
                data = json.loads(file_path.read_text())
                assert "iteration" in data
                assert "quality" in data
                assert "message" in data
                assert data["is_first"] == "false" or data["is_first"] is False
    
    @pytest.mark.asyncio
    async def test_iteration_in_paths_and_content(self, orchestrator, temp_dir):
        """Test iteration variables work in both file paths and content."""
        pipeline_yaml = f"""
name: test_paths_and_content
description: Test iteration in both paths and content

tasks:
  - id: path_content_loop
    while: "{{{{ iteration < 3 }}}}"
    max_iterations: 3
    steps:
      - id: create_directory
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/dir_{{{{ iteration }}}}/file_{{{{ iteration }}}}.txt"
          content: |
            Directory: dir_{{{{ iteration }}}}
            File: file_{{{{ iteration }}}}.txt
            Full path: {temp_dir}/dir_{{{{ iteration }}}}/file_{{{{ iteration }}}}.txt
            Iteration number: {{{{ iteration }}}}
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_paths.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Verify directories and files were created correctly
        for i in range(3):
            dir_path = Path(temp_dir) / f"dir_{i}"
            assert dir_path.exists(), f"Directory dir_{i} should exist"
            
            file_path = dir_path / f"file_{i}.txt"
            assert file_path.exists(), f"File dir_{i}/file_{i}.txt should exist"
            
            content = file_path.read_text()
            assert f"Directory: dir_{i}" in content
            assert f"File: file_{i}.txt" in content
            assert f"Iteration number: {i}" in content
    
    @pytest.mark.asyncio
    async def test_iteration_with_zero(self, orchestrator, temp_dir):
        """Test that iteration starting at 0 works correctly."""
        pipeline_yaml = f"""
name: test_zero_iteration
description: Test iteration starting at zero

tasks:
  - id: zero_loop
    while: "{{{{ iteration < 1 }}}}"
    max_iterations: 1
    steps:
      - id: save_zero
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/zero_{{{{ iteration }}}}.txt"
          content: "Iteration is {{{{ iteration }}}}, which should be 0"
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_zero.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Verify file was created with iteration 0
        file_path = Path(temp_dir) / "zero_0.txt"
        assert file_path.exists(), "zero_0.txt should exist"
        
        content = file_path.read_text()
        assert content == "Iteration is 0, which should be 0"
    
    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, orchestrator, temp_dir):
        """Test that max_iterations properly limits the loop."""
        pipeline_yaml = f"""
name: test_max_iterations
description: Test max iterations limit

tasks:
  - id: max_loop
    while: "true"  # Always true, rely on max_iterations
    max_iterations: 3
    steps:
      - id: save_file
        tool: filesystem
        action: write
        parameters:
          path: "{temp_dir}/max_{{{{ iteration }}}}.txt"
          content: "Iteration {{{{ iteration }}}} of max 3"
"""
        
        # Save pipeline to temp file
        pipeline_file = Path(temp_dir) / "test_max.yaml"
        pipeline_file.write_text(pipeline_yaml)
        
        # Run the pipeline
        result = await orchestrator.run(
            pipeline_path=str(pipeline_file),
            inputs={},
            output_path=temp_dir
        )
        
        # Verify exactly 3 files were created (0, 1, 2)
        for i in range(3):
            file_path = Path(temp_dir) / f"max_{i}.txt"
            assert file_path.exists(), f"max_{i}.txt should exist"
        
        # Verify no 4th file was created
        file_path = Path(temp_dir) / "max_3.txt"
        assert not file_path.exists(), "max_3.txt should NOT exist (max iterations reached)"