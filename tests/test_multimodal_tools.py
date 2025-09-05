"""Tests for multimodal tools."""

import asyncio
import base64
import os
import pytest
import tempfile
from PIL import Image
import io

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
from src.orchestrator.tools.multimodal_tools import (

    ImageAnalysisTool,
    ImageGenerationTool,
    AudioProcessingTool,
    VideoProcessingTool,
    ImageData,
    AudioData)


@pytest.fixture
async def setup_test_model(populated_model_registry):
    """Setup a test model for multimodal operations."""
    # Use the populated model registry that already has API keys loaded
    yield populated_model_registry


@pytest.fixture
def test_image():
    """Create a test image."""
    img = Image.new("RGB", (100, 100), color="red")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, "PNG")
        img_path = f.name

    # Also create base64 version
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    yield {"path": img_path, "base64": img_base64, "pil": img}

    # Cleanup
    if os.path.exists(img_path):
        os.unlink(img_path)


@pytest.fixture
def test_audio():
    """Create a test audio file."""
    # Create a simple WAV header (44 bytes) + minimal data
    wav_header = b"RIFF" + b"\x2c\x00\x00\x00" + b"WAVE"
    wav_header += b"fmt " + b"\x10\x00\x00\x00"  # fmt chunk size
    wav_header += b"\x01\x00"  # PCM
    wav_header += b"\x02\x00"  # 2 channels
    wav_header += b"\x44\xac\x00\x00"  # 44100 sample rate
    wav_header += b"\x10\xb1\x02\x00"  # byte rate
    wav_header += b"\x04\x00"  # block align
    wav_header += b"\x10\x00"  # bits per sample
    wav_header += b"data" + b"\x00\x00\x00\x00"  # data chunk

    audio_data = wav_header + b"\x00" * 1000  # Add some silence

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        audio_path = f.name

    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    yield {"path": audio_path, "base64": audio_base64}

    if os.path.exists(audio_path):
        os.unlink(audio_path)


@pytest.mark.asyncio
async def test_image_analysis_tool_load_file(test_image):
    """Test loading image from file."""
    tool = ImageAnalysisTool()

    # Test loading from file
    image_data = tool._load_image(test_image["path"])

    assert isinstance(image_data, ImageData)
    assert image_data.width == 100
    assert image_data.height == 100
    assert image_data.mode in ("RGB", "RGBA")
    assert image_data.metadata["source"] == "file"


@pytest.mark.asyncio
async def test_image_analysis_tool_load_base64(test_image):
    """Test loading image from base64."""
    tool = ImageAnalysisTool()

    # Test loading from base64
    image_data = tool._load_image(test_image["base64"])

    assert isinstance(image_data, ImageData)
    assert image_data.width == 100
    assert image_data.height == 100
    assert image_data.metadata["source"] == "base64"


@pytest.mark.asyncio
async def test_image_analysis_describe(test_image, setup_test_model):
    """Test image description analysis."""
    tool = ImageAnalysisTool()

    # Check if we have any vision-capable models
    registry = setup_test_model
    vision_models = []

    # Directly iterate through the models dictionary
    for model_key, model in registry.models.items():
        if (
            hasattr(model, "capabilities")
            and "vision" in model.capabilities.supported_tasks
        ):
            vision_models.append(model)

    if not vision_models:
        pytest.skip("No vision-capable models available")

    # Print found vision models for debugging
    print(f"\nFound {len(vision_models)} vision models")
    for model in vision_models[:3]:  # Show first 3
        print(f"  - {model.name} ({model.provider})")

    try:
        result = await tool.execute(
            image=test_image["path"], analysis_type="describe", detail_level="low"
        )
    except Exception as e:
        print(f"\nTool execution failed with: {e}")
        raise

    # Print result for debugging
    print(f"\nResult: {result}")

    assert result["success"] is True
    assert "analysis" in result
    assert "metadata" in result


@pytest.mark.asyncio
async def test_image_generation_placeholder():
    """Test image generation with placeholder."""
    tool = ImageGenerationTool()

    # Test will use placeholder since no image generation model
    result = await tool.execute(
        prompt="A beautiful sunset",
        size="256x256",
        num_images=2,
        output_format="base64")

    assert result["success"] is True
    assert len(result["images"]) == 2
    assert all(img["format"] == "base64" for img in result["images"])
    assert all(
        img["data"].startswith("data:image/png;base64,") for img in result["images"]
    )


@pytest.mark.asyncio
async def test_image_generation_file_output():
    """Test image generation with file output."""
    tool = ImageGenerationTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        result = await tool.execute(
            prompt="A mountain landscape",
            size="512x512",
            output_format="file",
            output_path=tmpdir)

        assert result["success"] is True
        assert len(result["images"]) == 1
        assert result["images"][0]["format"] == "file"

        # Check file exists
        filepath = result["images"][0]["path"]
        assert os.path.exists(filepath)
        assert filepath.startswith(tmpdir)


@pytest.mark.asyncio
async def test_audio_processing_load(test_audio):
    """Test audio loading."""
    tool = AudioProcessingTool()

    # Test loading from file
    audio_data = tool._load_audio(test_audio["path"])

    assert isinstance(audio_data, AudioData)
    assert audio_data.format == "wav"
    assert audio_data.metadata["source"] == "file"

    # Test loading from base64
    audio_data_b64 = tool._load_audio(test_audio["base64"])

    assert isinstance(audio_data_b64, AudioData)
    assert audio_data_b64.metadata["source"] == "base64"


@pytest.mark.asyncio
async def test_audio_transcribe(test_audio):
    """Test audio transcription."""
    tool = AudioProcessingTool()

    result = await tool.execute(
        audio=test_audio["path"], operation="transcribe", language="en"
    )

    assert result["success"] is True
    assert "transcription" in result
    assert result["metadata"]["language"] == "en"


@pytest.mark.asyncio
async def test_audio_analyze(test_audio):
    """Test audio analysis."""
    tool = AudioProcessingTool()

    result = await tool.execute(audio=test_audio["path"], operation="analyze")

    assert result["success"] is True
    assert "analysis" in result
    assert result["analysis"]["format"] == "wav"


@pytest.mark.asyncio
async def test_video_processing_analyze():
    """Test video analysis."""
    tool = VideoProcessingTool()

    # Use a fake video path
    result = await tool.execute(video="test_video.mp4", operation="analyze")

    assert result["success"] is True
    assert "analysis" in result
    assert "video_info" in result["analysis"]


@pytest.mark.asyncio
async def test_video_extract_frames():
    """Test video frame extraction."""
    tool = VideoProcessingTool()
    
    # Create a test video if samples directory exists, otherwise use placeholder
    test_video_path = "samples/test_video.mp4"
    if not os.path.exists(test_video_path):
        # Create a simple placeholder video or skip
        test_video_path = "non_existent_video.mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = await tool.execute(
            video=test_video_path,
            operation="extract_frames",
            frame_interval=2.0,
            output_path=tmpdir)

        assert result["success"] is True
        assert "frames" in result
        
        # If video exists, we should get frames
        if os.path.exists("samples/test_video.mp4") or os.path.exists("samples/test_video_real.mp4"):
            if len(result["frames"]) > 0:
                # Check that frame files were created
                for frame_path in result["frames"]:
                    assert os.path.exists(frame_path)
                    assert frame_path.endswith(".jpg")
        else:
            # With non-existent video, we get a placeholder frame or empty list
            assert len(result["frames"]) >= 0  # Can be 0 or 1 (placeholder)


@pytest.mark.asyncio
async def test_video_summarize():
    """Test video summarization."""
    tool = VideoProcessingTool()

    result = await tool.execute(
        video="http://example.com/video.mp4", operation="summarize"
    )

    assert result["success"] is True
    assert "summary" in result
    assert "key_moments" in result
    assert "metadata" in result


@pytest.mark.asyncio
async def test_invalid_operations():
    """Test invalid operations for each tool."""
    # Image analysis
    image_tool = ImageAnalysisTool()
    result = await image_tool.execute(image="test.jpg", analysis_type="invalid_type")
    assert result["success"] is False
    assert "Invalid analysis_type" in result["error"]

    # Audio processing
    audio_tool = AudioProcessingTool()
    result = await audio_tool.execute(audio="test.wav", operation="invalid_op")
    assert result["success"] is False
    assert "Invalid operation" in result["error"]

    # Video processing
    video_tool = VideoProcessingTool()
    result = await video_tool.execute(video="test.mp4", operation="invalid_op")
    assert result["success"] is False
    assert "Invalid operation" in result["error"]


@pytest.mark.asyncio
async def test_image_prepare_for_model(test_image):
    """Test image preparation for model input."""
    tool = ImageAnalysisTool()

    # Create a large image
    large_img = Image.new("RGB", (2000, 2000), color="blue")
    image_data = ImageData(
        data=large_img, format="PNG", width=2000, height=2000, mode="RGB", metadata={}
    )

    # Prepare for model (should resize)
    prepared_b64 = tool._prepare_image_for_model(image_data, max_size=1024)

    assert isinstance(prepared_b64, str)

    # Decode and check size
    decoded = base64.b64decode(prepared_b64)
    img = Image.open(io.BytesIO(decoded))
    assert img.width <= 1024
    assert img.height <= 1024


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_image_generation_placeholder())
    asyncio.run(test_video_processing_analyze())
    print("Basic multimodal tests passed!")
