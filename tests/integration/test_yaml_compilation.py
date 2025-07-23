#!/usr/bin/env python3
"""Test YAML compilation with AUTO tags."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def test_yaml_compilation():
    """Test YAML compilation with AUTO resolution."""
    print("📄 Testing YAML Compilation with AUTO")
    print("=" * 40)

    try:
        # Step 1: Test YAML compiler creation
        print("1️⃣ Creating YAML compiler...")
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
        from orchestrator.integrations.ollama_model import OllamaModel

        model = OllamaModel(model_name="llama3.2:1b", timeout=10)
        compiler = YAMLCompiler()
        compiler.ambiguity_resolver.model = model

        print(f"   Compiler created with model: {compiler.ambiguity_resolver.model.name}")

        # Step 2: Test simple YAML without AUTO
        print("2️⃣ Testing simple YAML...")
        simple_yaml = """
name: "test"
description: "Simple test"
steps:
  - id: test_step
    action: process
    parameters:
      format: "json"
"""
        pipeline = await compiler.compile_yaml(simple_yaml)
        print(f"   Simple YAML compiled: {len(pipeline.tasks)} tasks")

        # Step 3: Test YAML with AUTO tags
        print("3️⃣ Testing YAML with AUTO...")
        auto_yaml = """
name: "auto_test"
description: "AUTO test"
steps:
  - id: auto_step
    action: process
    parameters:
      format: <AUTO>json or csv</AUTO>
"""

        print("   Compiling AUTO YAML...")
        pipeline = await compiler.compile_yaml(auto_yaml)
        print(f"   AUTO YAML compiled: {len(pipeline.tasks)} tasks")

        # Check the resolved parameters
        task = pipeline.tasks[0]
        print(f"   Resolved format: {task.parameters.get('format', 'UNKNOWN')}")

        print("✅ YAML compilation successful!")
        return True

    except Exception as e:
        print(f"❌ YAML compilation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_yaml_compilation())
    sys.exit(0 if success else 1)
