"""
Test suite for control_flow_conditional pipeline (Issue #160).
"""

import pytest
import asyncio
from pathlib import Path
import shutil
import sys

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


@pytest.fixture
def orchestrator():
    """Create orchestrator instance."""
    return Orchestrator(model_registry=init_models())


@pytest.fixture
def pipeline_yaml():
    """Load the control_flow_conditional pipeline."""
    with open('examples/control_flow_conditional.yaml', 'r') as f:
        return f.read()


@pytest.fixture
def test_dir():
    """Create and cleanup test directory."""
    test_path = Path('data/test_control_flow')
    test_path.mkdir(exist_ok=True, parents=True)
    yield test_path
    shutil.rmtree(test_path, ignore_errors=True)


@pytest.fixture
def output_dir():
    """Get output directory path."""
    return Path('examples/outputs/control_flow_conditional')


@pytest.mark.asyncio
async def test_empty_file_handling(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that empty files are handled correctly without conversational output."""
    # Create empty test file
    test_file = test_dir / 'empty.txt'
    test_file.write_text('')
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_empty.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify markdown format
    assert content.startswith("# Processed File"), "Not a proper markdown file"
    
    # Verify processing type
    assert "Processing type: Empty file" in content, "Wrong processing type"
    
    # Verify correct empty message
    assert "The input file was empty. No content to process." in content, "Wrong empty message"
    
    # Verify no template placeholders
    assert "{{" not in content and "}}" not in content, "Template placeholders found"
    
    # Verify no conversational output
    conversational = ["let's", "let me", "okay", "here's", "i'll", "please provide"]
    for word in conversational:
        assert word not in content.lower(), f"Conversational output detected: {word}"


@pytest.mark.asyncio
async def test_small_file_expansion(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that small files are expanded correctly."""
    # Create small test file
    test_content = "This is a small test file for expansion."
    test_file = test_dir / 'small.txt'
    test_file.write_text(test_content)
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_small.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify markdown format
    assert content.startswith("# Processed File"), "Not a proper markdown file"
    
    # Verify processing type
    assert "Processing type: Expanded" in content, "Wrong processing type"
    
    # Verify size is correct
    size = len(test_content)
    assert f"Original size: {size} bytes" in content, "Wrong size reported"
    
    # Verify content was expanded (should be longer than original)
    assert len(content) > len(test_content) * 5, "Content not properly expanded"
    
    # Verify no template placeholders
    assert "{{" not in content and "}}" not in content, "Template placeholders found"


@pytest.mark.asyncio
async def test_large_file_compression(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that large files are compressed with bullet points."""
    # Create large test file
    test_content = "This is a large test file. " * 100  # Makes it over 1000 bytes
    test_file = test_dir / 'large.txt'
    test_file.write_text(test_content)
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_large.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify markdown format
    assert content.startswith("# Processed File"), "Not a proper markdown file"
    
    # Verify processing type
    assert "Processing type: Compressed" in content, "Wrong processing type"
    
    # Verify bullet points exist (either • or * format)
    has_bullets = "•" in content or "*   " in content
    assert has_bullets, "No bullet points found"
    
    bullet_count = content.count("•") + content.count("*   ")
    assert bullet_count >= 3, "Should have at least 3 bullet points"
    
    # Verify bullet points have content
    lines = content.split('\n')
    bullet_lines = [l for l in lines if l.strip().startswith("•") or l.strip().startswith("*")]
    for line in bullet_lines:
        assert len(line.strip()) > 5, f"Empty or too short bullet point: {line}"
    
    # Verify no template placeholders
    assert "{{" not in content and "}}" not in content, "Template placeholders found"


@pytest.mark.asyncio
async def test_exact_threshold_handling(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test file exactly at threshold (1000 bytes)."""
    # Create file exactly at threshold
    test_content = "X" * 1000
    test_file = test_dir / 'threshold.txt'
    test_file.write_text(test_content)
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_threshold.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify processing type (should be expanded at exactly 1000)
    assert "Processing type: Expanded" in content, "Wrong processing type for threshold"
    
    # Verify repetitive content is explained, not just output
    assert "repetitive" in content.lower() or "pattern" in content.lower(), "Repetitive pattern not explained"
    assert content.count('X' * 50) == 0, "Should not output raw X's"


@pytest.mark.asyncio
async def test_special_characters_handling(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that special characters don't trigger safety filters."""
    # Create file with special characters
    test_content = "Test with émojis 🎉 and spëcial çharacters: @#$%^&*()"
    test_file = test_dir / 'special.txt'
    test_file.write_text(test_content)
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_special.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify processing type
    assert "Processing type: Expanded" in content, "Wrong processing type"
    
    # Verify special characters are handled (not rejected)
    assert "cannot" not in content.lower() and "refuse" not in content.lower(), "Content was rejected"
    
    # Verify content discusses the special characters
    assert "special" in content.lower() or "émoji" in content.lower(), "Special characters not discussed"


@pytest.mark.asyncio
async def test_repetitive_content_handling(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that repetitive content is properly analyzed."""
    # Create file with repetitive content
    test_content = "A" * 2000
    test_file = test_dir / 'repetitive.txt'
    test_file.write_text(test_content)
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_repetitive.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify processing type (should be compressed)
    assert "Processing type: Compressed" in content, "Wrong processing type"
    
    # Verify correct count is mentioned
    assert "2000" in content or "2,000" in content, "Wrong character count"
    
    # Verify it describes the pattern
    assert "•" in content, "No bullet points for compression"
    assert "character 'A'" in content or "A'" in content or "'A" in content, "Pattern not described"


@pytest.mark.asyncio
async def test_multiline_content_handling(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that multiline content is handled correctly."""
    # Create multiline file
    lines = [f"Line {i}: This is test content for line number {i}." for i in range(1, 11)]
    test_content = '\n'.join(lines)
    test_file = test_dir / 'multiline.txt'
    test_file.write_text(test_content)
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_multiline.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify processing type
    assert "Processing type: Expanded" in content, "Wrong processing type"
    
    # Verify multiline nature is discussed
    assert "line" in content.lower(), "Multiline nature not discussed"
    
    # Verify expansion occurred
    assert len(content) > len(test_content) * 3, "Content not properly expanded"


@pytest.mark.asyncio
async def test_file_extension_is_markdown(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that all output files have .md extension."""
    # Create test file
    test_file = test_dir / 'extension_test.txt'
    test_file.write_text('Test content')
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output has .md extension
    output_file = output_dir / 'processed_extension_test.md'
    assert output_file.exists(), "Output file with .md extension not created"
    
    # Ensure no .txt.md double extension
    wrong_file = output_dir / 'processed_extension_test.txt.md'
    assert not wrong_file.exists(), "File with double extension created"
    
    # Verify it's proper markdown
    content = output_file.read_text()
    assert content.startswith("# Processed File"), "Not a proper markdown file"
    assert "## Result" in content, "Missing markdown headers"


@pytest.mark.asyncio
async def test_no_conversational_output(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that no conversational language appears in any output."""
    test_cases = [
        ('empty_conv.txt', ''),
        ('small_conv.txt', 'Test'),
        ('large_conv.txt', 'X' * 1500),
    ]
    
    conversational_phrases = [
        "let's", "let me", "i'll help", "okay,", "sure,",
        "please provide", "i can help", "would you like", "shall we"
    ]
    
    for filename, content in test_cases:
        test_file = test_dir / filename
        test_file.write_text(content)
        
        context = {
            'input_file': str(test_file),
            'size_threshold': 1000
        }
        
        result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
        
        output_name = filename.replace('.txt', '')
        output_file = output_dir / f'processed_{output_name}.md'
        assert output_file.exists(), f"Output file not created for {filename}"
        
        output_content = output_file.read_text().lower()
        
        for phrase in conversational_phrases:
            assert phrase not in output_content, f"Conversational phrase '{phrase}' found in {filename} output"


@pytest.mark.asyncio 
async def test_byte_size_accuracy(orchestrator, pipeline_yaml, test_dir, output_dir):
    """Test that byte sizes are reported accurately, not confused with kilobytes."""
    # Create file of specific size
    test_content = "B" * 448  # Specific size that was problematic before
    test_file = test_dir / 'size_test.txt'
    test_file.write_text(test_content)
    
    # Run pipeline
    context = {
        'input_file': str(test_file),
        'size_threshold': 1000
    }
    
    result = await orchestrator.execute_yaml(pipeline_yaml, context=context)
    
    # Check output
    output_file = output_dir / 'processed_size_test.md'
    assert output_file.exists(), "Output file not created"
    
    content = output_file.read_text()
    
    # Verify correct byte size
    assert "Original size: 448 bytes" in content, "Wrong size reported"
    
    # Ensure no confusion with kilobytes
    assert "kilobyte" not in content.lower() and "kb" not in content.lower(), "Size confused with kilobytes"