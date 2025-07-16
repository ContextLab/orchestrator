"""Working tests for documentation code snippets - Batch 18."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_model_configuration_lines_16_61_0():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 16-61."""
    # Description: ~~~~~~~~~~~~~~~~~~~~~~~
    import yaml
    
    content = 'models:\n  # Ollama models (automatically installed if not present)\n  - source: ollama\n    name: gemma2:27b\n    expertise:\n      - general\n      - reasoning\n      - analysis\n    size: 27b\n\n  - source: ollama\n    name: codellama:7b\n    expertise:\n      - code\n      - programming\n    size: 7b\n\n  # HuggingFace models (automatically downloaded)\n  - source: huggingface\n    name: microsoft/phi-2\n    expertise:\n      - reasoning\n      - code\n    size: 2.7b\n\n  # Cloud models (require API keys)\n  - source: openai\n    name: gpt-4o\n    expertise:\n      - general\n      - reasoning\n      - code\n      - analysis\n      - vision\n    size: 1760b\n\ndefaults:\n  expertise_preferences:\n    code: codellama:7b\n    reasoning: gemma2:27b\n    fast: llama3.2:1b\n  fallback_chain:\n    - gemma2:27b\n    - llama3.2:1b\n    - TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_model_configuration_lines_104_112_1():
    """Test Python snippet from docs/user_guide/model_configuration.rst lines 104-112."""
    # Description: The framework uses lazy loading for both Ollama and HuggingFace models to avoid downloading large models until they're actually needed:
    content = 'import orchestrator as orc\n\n# This registers models but doesn\'t download them yet\nregistry = orc.init_models()\n\n# Models are downloaded only when first used by a pipeline\npipeline = orc.compile("my_pipeline.yaml")\nresult = pipeline.run()  # Model downloads happen here if needed'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    try:
        compile(content, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_model_configuration_lines_130_133_2():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 130-133."""
    # Description: HuggingFace models are also downloaded on first use:
    import yaml
    
    content = '- source: huggingface\n  name: microsoft/Phi-3.5-mini-instruct\n  expertise: [reasoning, code]'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_model_configuration_lines_156_162_3():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 156-162."""
    # Description: Specify a model by name:
    import yaml
    
    content = 'steps:\n  - id: summarize\n    action: generate_text\n    parameters:\n      prompt: "Summarize this text..."\n    requires_model: gemma2:27b  # Use specific model'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_model_configuration_lines_170_178_4():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 170-178."""
    # Description: Specify requirements and let the framework choose:
    import yaml
    
    content = 'steps:\n  - id: generate_code\n    action: generate_text\n    parameters:\n      prompt: "Write a Python function..."\n    requires_model:\n      expertise: code\n      min_size: 7b  # At least 7B parameters'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_model_configuration_lines_186_196_5():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 186-196."""
    # Description: Specify multiple expertise areas (any match will qualify):
    import yaml
    
    content = 'steps:\n  - id: analyze\n    action: analyze\n    parameters:\n      content: "{input_data}"\n    requires_model:\n      expertise:\n        - reasoning\n        - analysis\n      min_size: 20b'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_model_configuration_lines_204_240_6():
    """Test YAML snippet from docs/user_guide/model_configuration.rst lines 204-240."""
    # Description: Here's a complete pipeline demonstrating model requirements:
    import yaml
    
    content = 'id: multi_model_pipeline\nname: Multi-Model Processing Pipeline\n\ninputs:\n  - name: topic\n    type: string\n\nsteps:\n  # Fast task with small model\n  - id: quick_check\n    action: generate_text\n    parameters:\n      prompt: "Is this topic related to programming: {topic}?"\n    requires_model:\n      expertise: fast\n      min_size: 0  # Any size\n\n  # Code generation with specialized model\n  - id: code_example\n    action: generate_text\n    parameters:\n      prompt: "Generate example code for: {topic}"\n    requires_model:\n      expertise: code\n      min_size: 7b\n    dependencies: [quick_check]\n\n  # Complex reasoning with large model\n  - id: deep_analysis\n    action: analyze\n    parameters:\n      content: "{topic} with code: {code_example.result}"\n    requires_model:\n      expertise: [reasoning, analysis]\n      min_size: 27b\n    dependencies: [code_example]'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        assert data is not None
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                assert 'id' in step, "Each step should have an id"


def test_model_configuration_lines_280_291_7():
    """Test Python snippet from docs/user_guide/model_configuration.rst lines 280-291."""
    # Description: Check which models are being used:
    content = 'import orchestrator as orc\n\n# Initialize and list available models\nregistry = orc.init_models()\nprint("Available models:")\nfor model_key in registry.list_models():\n    print(f"  - {model_key}")\n\n# Run pipeline and check model selection\npipeline = orc.compile("pipeline.yaml")\nresult = pipeline.run(topic="AI agents")'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    try:
        compile(content, '<string>', 'exec')
    except SyntaxError as e:
        pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_model_configuration_lines_296_299_8():
    """Test text snippet from docs/user_guide/model_configuration.rst lines 296-299."""
    # Description: The framework logs model selection decisions:
    content = ">> Using model for task 'quick_check': ollama:llama3.2:1b (fast, 1B params)\n>> Using model for task 'code_example': ollama:codellama:7b (code, 7B params)\n>> Using model for task 'deep_analysis': ollama:gemma2:27b (reasoning, 27B params)"
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_model_configuration_lines_337_338_9():
    """Test text snippet from docs/user_guide/model_configuration.rst lines 337-338."""
    # Description: **Model Installation Fails**:
    content = '>> ❌ Failed to install gemma2:27b: connection timeout'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"
