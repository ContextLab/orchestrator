"""Working tests for documentation code snippets - Batch 1."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_CLAUDE_lines_49_59_0():
    """Test text snippet from CLAUDE.md lines 49-59."""
    # Description: - Redis - For caching and session management
    content = 'orchestrator/\n├── src/orchestrator/          # Core library code\n│   ├── compiler/             # YAML parsing and compilation\n│   ├── models/               # Model abstractions and registry\n│   ├── adapters/             # Control system adapters\n│   ├── executor/             # Sandboxed execution\n│   └── state/                # State management\n├── tests/                    # Unit and integration tests\n├── examples/                 # Example pipeline definitions\n├── docs/                     # Documentation\n└── config/                   # Configuration schemas'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_CLAUDE_lines_67_73_1():
    """Test YAML snippet from CLAUDE.md lines 67-73."""
    # Description: Pipelines use <AUTO> tags for LLM-resolved ambiguities:
    import yaml
    
    content = 'steps:\n  - id: analyze_data\n    action: analyze\n    parameters:\n      data: "{{ input_data }}"\n      method: <AUTO>Choose best analysis method for this data type</AUTO>\n      depth: <AUTO>Determine analysis depth based on data complexity</AUTO>'
    
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


def test_README_lines_32_32_2():
    """Test Bash snippet from README.md lines 32-32."""
    # Description: - 💾 Lazy Model Loading: Models are downloaded only when needed, saving disk space
    content = 'pip install py-orc'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Special handling for pip install commands
    if 'pip install' in content:
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('pip install'), f"Expected pip install command: {line}"
        return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"


def test_README_lines_37_40_3():
    """Test Bash snippet from README.md lines 37-40."""
    # Description: For additional features:
    content = 'pip install py-orc[ollama]      # Ollama model support\npip install py-orc[cloud]        # Cloud model providers\npip install py-orc[dev]          # Development tools\npip install py-orc[all]          # Everything'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Special handling for pip install commands
    if 'pip install' in content:
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                assert line.startswith('pip install'), f"Expected pip install command: {line}"
        return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"


def test_README_lines_48_66_4():
    """Test YAML snippet from README.md lines 48-66."""
    # Description: 1. Create a simple pipeline (hello_world.yaml):
    import yaml
    
    content = 'id: hello_world\nname: Hello World Pipeline\ndescription: A simple example pipeline\n\nsteps:\n  - id: greet\n    action: generate_text\n    parameters:\n      prompt: "Say hello to the world in a creative way!"\n\n  - id: translate\n    action: generate_text\n    parameters:\n      prompt: "Translate this greeting to Spanish: {{ greet.result }}"\n    dependencies: [greet]\n\noutputs:\n  greeting: "{{ greet.result }}"\n  spanish: "{{ translate.result }}"'
    
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


def test_README_lines_72_81_5():
    """Test Python snippet from README.md lines 72-81."""
    # Description: 2. Run the pipeline:
    content = 'import orchestrator as orc\n\n# Initialize models (auto-detects available models)\norc.init_models()\n\n# Compile and run the pipeline\npipeline = orc.compile("hello_world.yaml")\nresult = pipeline.run()\n\nprint(result)'
    
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


def test_README_lines_89_95_6():
    """Test YAML snippet from README.md lines 89-95."""
    # Description: Orchestrator's <AUTO> tags let AI decide configuration details:
    import yaml
    
    content = 'steps:\n  - id: analyze_data\n    action: analyze\n    parameters:\n      data: "{{ input_data }}"\n      method: <AUTO>Choose the best analysis method for this data type</AUTO>\n      visualization: <AUTO>Decide if we should create a chart</AUTO>'
    
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


def test_README_lines_103_125_7():
    """Test YAML snippet from README.md lines 103-125."""
    # Description: Configure available models in models.yaml:
    import yaml
    
    content = 'models:\n  # Local models (via Ollama) - downloaded on first use\n  - source: ollama\n    name: llama3.1:8b\n    expertise: [general, reasoning, multilingual]\n    size: 8b\n\n  - source: ollama\n    name: qwen2.5-coder:7b\n    expertise: [code, programming]\n    size: 7b\n\n  # Cloud models\n  - source: openai\n    name: gpt-4o\n    expertise: [general, reasoning, code, analysis, vision]\n    size: 1760b  # Estimated\n\ndefaults:\n  expertise_preferences:\n    code: qwen2.5-coder:7b\n    reasoning: deepseek-r1:8b\n    fast: llama3.2:1b'
    
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


def test_README_lines_135_192_8():
    """Test YAML snippet from README.md lines 135-192."""
    # Description: Here's a more complex example showing model requirements and parallel execution:
    import yaml
    
    content = 'id: research_pipeline\nname: AI Research Pipeline\ndescription: Research a topic and create a comprehensive report\n\ninputs:\n  - name: topic\n    type: string\n    description: Research topic\n\n  - name: depth\n    type: string\n    default: <AUTO>Determine appropriate research depth</AUTO>\n\nsteps:\n  # Parallel research from multiple sources\n  - id: web_search\n    action: search_web\n    parameters:\n      query: "{{ topic }} latest research 2025"\n      count: <AUTO>Decide how many results to fetch</AUTO>\n    requires_model:\n      expertise: [research, web]\n\n  - id: academic_search\n    action: search_academic\n    parameters:\n      query: "{{ topic }}"\n      filters: <AUTO>Set appropriate academic filters</AUTO>\n    requires_model:\n      expertise: [research, academic]\n\n  # Analyze findings with specialized model\n  - id: analyze_findings\n    action: analyze\n    parameters:\n      web_results: "{{ web_search.results }}"\n      academic_results: "{{ academic_search.results }}"\n      analysis_focus: <AUTO>Determine key aspects to analyze</AUTO>\n    dependencies: [web_search, academic_search]\n    requires_model:\n      expertise: [analysis, reasoning]\n      min_size: 20b  # Require large model for complex analysis\n\n  # Generate report\n  - id: write_report\n    action: generate_document\n    parameters:\n      topic: "{{ topic }}"\n      analysis: "{{ analyze_findings.result }}"\n      style: <AUTO>Choose appropriate writing style</AUTO>\n      length: <AUTO>Determine optimal report length</AUTO>\n    dependencies: [analyze_findings]\n    requires_model:\n      expertise: [writing, general]\n\noutputs:\n  report: "{{ write_report.document }}"\n  summary: "{{ analyze_findings.summary }}"'
    
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


def test_README_lines_200_274_9():
    """Test YAML snippet from README.md lines 200-274."""
    # Description: Here's a fully functional pipeline that generates research reports:
    import yaml
    
    content = '# research_report.yaml\nid: research_report\nname: Research Report Generator\ndescription: Generate comprehensive research reports with citations\n\ninputs:\n  - name: topic\n    type: string\n    description: Research topic\n  - name: instructions\n    type: string\n    description: Additional instructions for the report\n\noutputs:\n  - pdf: <AUTO>Generate appropriate filename for the research report PDF</AUTO>\n\nsteps:\n  - id: search\n    name: Web Search\n    action: search_web\n    parameters:\n      query: <AUTO>Create effective search query for {topic} with {instructions}</AUTO>\n      max_results: 10\n    requires_model:\n      expertise: fast\n\n  - id: compile_notes\n    name: Compile Research Notes\n    action: generate_text\n    parameters:\n      prompt: |\n        Compile comprehensive research notes from these search results:\n        {{ search.results }}\n\n        Topic: {{ topic }}\n        Instructions: {{ instructions }}\n\n        Create detailed notes with:\n        - Key findings\n        - Important quotes\n        - Source citations\n        - Relevant statistics\n    dependencies: [search]\n    requires_model:\n      expertise: [analysis, reasoning]\n      min_size: 7b\n\n  - id: write_report\n    name: Write Report\n    action: generate_document\n    parameters:\n      content: |\n        Write a comprehensive research report on "{{ topic }}"\n\n        Research notes:\n        {{ compile_notes.result }}\n\n        Requirements:\n        - Professional academic style\n        - Include introduction, body sections, and conclusion\n        - Cite sources properly\n        - {{ instructions }}\n      format: markdown\n    dependencies: [compile_notes]\n    requires_model:\n      expertise: [writing, general]\n      min_size: 20b\n\n  - id: create_pdf\n    name: Create PDF\n    action: convert_to_pdf\n    parameters:\n      markdown: "{{ write_report.document }}"\n      filename: "{{ outputs.pdf }}"\n    dependencies: [write_report]'
    
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
