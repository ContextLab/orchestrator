"""Real integration tests for multimodal tools - NO MOCKS."""

import asyncio
import base64
import os
import pytest
import tempfile
import shutil
import numpy as np
from PIL import Image
import cv2

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
from src.orchestrator.tools.multimodal_tools import (

    ImageAnalysisTool,
    ImageGenerationTool,
    AudioProcessingTool,
    VideoProcessingTool,
)


@pytest.fixture
def real_test_image():
    """Create a real test image with actual content."""
    # Create an image with text and shapes
    img = Image.new("RGB", (512, 512), color="white")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 200, 200], fill="red", outline="black", width=3)
    draw.ellipse([250, 50, 400, 200], fill="blue", outline="black", width=3)
    draw.polygon([(100, 300), (200, 450), (50, 450)], fill="green", outline="black", width=3)
    
    # Add text
    draw.text((150, 250), "Test Image", fill="black")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f, "JPEG")
        img_path = f.name
    
    yield {"path": img_path, "image": img}
    
    # Cleanup
    if os.path.exists(img_path):
        os.unlink(img_path)


@pytest.fixture
def real_test_audio():
    """Create a real audio file with speech."""
    # Use system TTS to create real speech audio
    text = "This is a real test of the audio transcription system."
    
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        aiff_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    
    # Create audio with macOS say command
    os.system(f'say -o {aiff_path} "{text}"')
    
    # Convert to WAV
    os.system(f'ffmpeg -i {aiff_path} -acodec pcm_s16le -ar 44100 {wav_path} -y 2>/dev/null')
    
    yield {"path": wav_path, "text": text}
    
    # Cleanup
    for path in [aiff_path, wav_path]:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def real_test_video():
    """Create a real video file with actual content."""
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    
    # Video parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    width, height = 640, 480
    duration = 2  # seconds
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames with actual content
    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Scene 1: 0-0.5s - Red background with circle
        if i < fps * 0.5:
            frame[:] = (0, 0, 100)  # Red background
            cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
            cv2.putText(frame, "Scene 1", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Scene 2: 0.5-1s - Green background with rectangle
        elif i < fps * 1:
            frame[:] = (0, 100, 0)  # Green background
            cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
            cv2.putText(frame, "Scene 2", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Scene 3: 1-1.5s - Blue background with triangle
        elif i < fps * 1.5:
            frame[:] = (100, 0, 0)  # Blue background
            pts = np.array([[320, 150], [220, 330], [420, 330]], np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 255))
            cv2.putText(frame, "Scene 3", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Scene 4: 1.5-2s - Yellow background with text
        else:
            frame[:] = (0, 200, 200)  # Yellow background
            cv2.putText(frame, "Final Scene", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        # Add frame counter
        cv2.putText(frame, f"Frame {i}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    
    yield {"path": output_path, "duration": duration, "fps": fps, "scenes": 4}
    
    # Cleanup
    if os.path.exists(output_path):
        os.unlink(output_path)


class TestRealImageAnalysis:
    """Test real image analysis without mocks."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_image_description(self, real_test_image):
        """Test real image description with vision models."""
        tool = ImageAnalysisTool()
        
        result = await tool.execute(
            image=real_test_image["path"],
            analysis_type="describe"
        )
        
        assert result["success"] is True
        assert "analysis" in result
        
        # Verify real description contains expected elements
        description = result["analysis"]["result"].lower()
        
        # Should mention colors or shapes
        assert any(word in description for word in ["red", "blue", "green", "rectangle", "circle", "triangle", "shape"])
        
        # Should not be a placeholder
        assert "placeholder" not in description
        assert "[" not in description
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_object_detection(self, real_test_image):
        """Test real object detection in images."""
        tool = ImageAnalysisTool()
        
        result = await tool.execute(
            image=real_test_image["path"],
            analysis_type="detect_objects"
        )
        
        assert result["success"] is True
        detection_result = result["analysis"]["result"].lower()
        
        # Should detect shapes or mention geometric forms
        assert any(word in detection_result for word in ["rectangle", "circle", "triangle", "shape", "geometric"])


class TestRealAudioProcessing:
    """Test real audio processing without mocks."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_audio_transcription(self, real_test_audio):
        """Test real audio transcription with speech-to-text."""
        tool = AudioProcessingTool()
        
        result = await tool.execute(
            audio=real_test_audio["path"],
            operation="transcribe",
            language="en"
        )
        
        assert result["success"] is True
        assert "transcription" in result
        
        transcription = result["transcription"].lower()
        
        # Should contain key words from the original text
        assert "test" in transcription or "audio" in transcription or "transcription" in transcription
        
        # Should not be a placeholder
        assert "[" not in transcription
        assert "placeholder" not in transcription
        
        # Check confidence
        assert result["metadata"]["confidence"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_audio_analysis(self, real_test_audio):
        """Test real audio analysis with librosa."""
        tool = AudioProcessingTool()
        
        result = await tool.execute(
            audio=real_test_audio["path"],
            operation="analyze"
        )
        
        assert result["success"] is True
        assert "analysis" in result
        
        analysis = result["analysis"]
        
        # Check real audio properties
        assert analysis["duration"] > 0  # Should have non-zero duration
        assert analysis["sample_rate"] == 44100  # We set this in fixture
        assert "analysis" in analysis
        
        detailed = analysis["analysis"]
        
        # Check real audio features
        assert "tempo_bpm" in detailed
        assert detailed["tempo_bpm"] > 0
        
        assert "spectral_centroid_hz" in detailed
        assert detailed["spectral_centroid_hz"] > 0
        
        assert "peak_amplitude" in detailed
        assert 0 < detailed["peak_amplitude"] <= 1
        
        assert "volume_level" in detailed
        assert detailed["volume_level"] in ["very_quiet", "quiet", "normal", "loud", "very_loud"]


class TestRealVideoProcessing:
    """Test real video processing without mocks."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_video_metadata(self, real_test_video):
        """Test real video metadata extraction with OpenCV."""
        tool = VideoProcessingTool()
        
        result = await tool.execute(
            video=real_test_video["path"],
            operation="analyze"
        )
        
        assert result["success"] is True
        assert "analysis" in result
        
        video_info = result["analysis"]["video_info"]
        
        # Check real video properties
        assert abs(video_info["duration"] - real_test_video["duration"]) < 0.1
        assert video_info["fps"] == real_test_video["fps"]
        assert video_info["resolution"] == "640x480"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_frame_extraction(self, real_test_video):
        """Test real frame extraction from video."""
        tool = VideoProcessingTool()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await tool.execute(
                video=real_test_video["path"],
                operation="extract_frames",
                frame_interval=0.5,  # Extract every 0.5 seconds
                output_path=tmpdir
            )
            
            assert result["success"] is True
            assert "frames" in result
            
            frames = result["frames"]
            assert len(frames) == 4  # Should extract 4 frames (at 0s, 0.5s, 1s, 1.5s)
            
            # Verify frames are real images
            for frame_path in frames:
                assert os.path.exists(frame_path)
                assert os.path.getsize(frame_path) > 1000  # Should be actual image, not placeholder
                
                # Load and check image
                img = Image.open(frame_path)
                assert img.size == (640, 480)
                assert img.mode in ["RGB", "L"]


class TestRealImageGeneration:
    """Test real image generation without placeholders."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_dalle_generation(self):
        """Test real DALL-E image generation if API key available."""
        tool = ImageGenerationTool()
        
        # Check if we have OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await tool.execute(
                prompt="A simple red circle on white background",
                size="1024x1024",
                num_images=1,
                output_format="file",
                output_path=tmpdir
            )
            
            if result["success"]:
                assert len(result["images"]) == 1
                
                # Check if real image was generated
                img_path = result["images"][0]["path"]
                assert os.path.exists(img_path)
                
                # Verify it's a real image
                img = Image.open(img_path)
                assert img.size == (1024, 1024)
                
                # Should not be a placeholder (gray image)
                # Real DALL-E images have variety in colors
                img_array = np.array(img)
                color_variance = np.var(img_array)
                assert color_variance > 100  # Real images have color variation


class TestEndToEndPipeline:
    """Test complete multimodal pipeline with real data."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_full_multimodal_pipeline(self, real_test_image, real_test_audio, real_test_video):
        """Test processing all media types in sequence."""
        results = {}
        
        # Process image
        img_tool = ImageAnalysisTool()
        img_result = await img_tool.execute(
            image=real_test_image["path"],
            analysis_type="describe"
        )
        assert img_result["success"] is True
        results["image"] = img_result
        
        # Process audio
        audio_tool = AudioProcessingTool()
        audio_result = await audio_tool.execute(
            audio=real_test_audio["path"],
            operation="transcribe"
        )
        assert audio_result["success"] is True
        results["audio"] = audio_result
        
        # Process video
        video_tool = VideoProcessingTool()
        with tempfile.TemporaryDirectory() as tmpdir:
            video_result = await video_tool.execute(
                video=real_test_video["path"],
                operation="extract_frames",
                frame_interval=1.0,
                output_path=tmpdir
            )
            assert video_result["success"] is True
            results["video"] = video_result
            
            # Analyze extracted frames
            if video_result["frames"]:
                frame_analysis = await img_tool.execute(
                    image=video_result["frames"][0],
                    analysis_type="describe"
                )
                assert frame_analysis["success"] is True
                results["frame_analysis"] = frame_analysis
        
        # Verify all results are real, not placeholders
        assert "placeholder" not in str(results).lower()
        assert all(r["success"] for r in results.values())


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_processing():
    """Test handling multiple media files concurrently."""
    # Create multiple test files
    files = []
    
    for i in range(3):
        # Create simple images
        img = Image.new("RGB", (100, 100), color=(i*50, i*50, i*50))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f, "JPEG")
            files.append(f.name)
    
    try:
        tool = ImageAnalysisTool()
        
        # Process concurrently
        tasks = [
            tool.execute(image=f, analysis_type="describe")
            for f in files
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r["success"] for r in results)
        
        # All should have unique descriptions (not cached/duplicated)
        descriptions = [r["analysis"]["result"] for r in results]
        # Descriptions might be similar but should have some variation
        assert len(set(descriptions)) >= 2  # At least 2 unique descriptions
        
    finally:
        # Cleanup
        for f in files:
            if os.path.exists(f):
                os.unlink(f)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_error_handling():
    """Test handling of corrupted or invalid media files."""
    
    # Test with non-existent file
    tool = AudioProcessingTool()
    result = await tool.execute(
        audio="non_existent_file.wav",
        operation="analyze"
    )
    # Should handle gracefully
    assert "success" in result
    
    # Test with corrupted audio data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"This is not valid audio data")
        corrupted_path = f.name
    
    try:
        result = await tool.execute(
            audio=corrupted_path,
            operation="analyze"
        )
        # Should handle gracefully without crashing
        assert "success" in result
        
    finally:
        if os.path.exists(corrupted_path):
            os.unlink(corrupted_path)