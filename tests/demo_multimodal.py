"""Simple demonstration of multimodal tools."""

import asyncio
import base64
from PIL import Image
import io

from src.orchestrator.tools.multimodal_tools import (
    ImageAnalysisTool,
    ImageGenerationTool,
    AudioProcessingTool,
    VideoProcessingTool
)


async def demo_image_generation():
    """Demonstrate image generation."""
    print("\n=== Image Generation Demo ===")
    
    tool = ImageGenerationTool()
    
    # Generate some images
    result = await tool.execute(
        prompt="A futuristic city with flying cars",
        size="512x512",
        style="cyberpunk",
        num_images=2,
        output_format="base64"
    )
    
    if result["success"]:
        print(f"✓ Generated {len(result['images'])} images")
        print(f"  Prompt: {result['metadata']['prompt']}")
        print(f"  Size: {result['metadata']['size']}")
        
        # Decode first image to verify
        img_data = result['images'][0]['data']
        if img_data.startswith('data:'):
            img_data = img_data.split(',')[1]
        
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        print(f"  Verified image dimensions: {img.size}")
    else:
        print(f"✗ Generation failed: {result['error']}")


async def demo_image_analysis():
    """Demonstrate image analysis."""
    print("\n=== Image Analysis Demo ===")
    
    # Create a test image
    img = Image.new('RGB', (200, 200), color='blue')
    
    # Add some shapes
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill='red')
    draw.ellipse([75, 75, 125, 125], fill='yellow')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    tool = ImageAnalysisTool()
    
    # Analyze (will use placeholder without real model)
    result = await tool.execute(
        image=f"data:image/png;base64,{img_b64}",
        analysis_type="describe",
        detail_level="low"
    )
    
    if result["success"]:
        print("✓ Image analysis completed")
        print(f"  Image size: {result['analysis']['image_info']['width']}x{result['analysis']['image_info']['height']}")
        print(f"  Analysis type: {result['metadata']['analysis_type']}")
    else:
        print(f"✗ Analysis failed: {result['error']}")


async def demo_audio_processing():
    """Demonstrate audio processing."""
    print("\n=== Audio Processing Demo ===")
    
    # Create minimal WAV data
    wav_header = b'RIFF' + b'\x2c\x00\x00\x00' + b'WAVE'
    wav_header += b'fmt ' + b'\x10\x00\x00\x00'
    wav_header += b'\x01\x00' + b'\x02\x00'
    wav_header += b'\x44\xac\x00\x00' + b'\x10\xb1\x02\x00'
    wav_header += b'\x04\x00' + b'\x10\x00'
    wav_header += b'data' + b'\x00\x00\x00\x00'
    audio_data = wav_header + b'\x00' * 1000
    
    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
    
    tool = AudioProcessingTool()
    
    # Transcribe
    result = await tool.execute(
        audio=f"data:audio/wav;base64,{audio_b64}",
        operation="transcribe",
        language="en"
    )
    
    if result["success"]:
        print("✓ Audio transcription completed")
        print(f"  Language: {result['metadata']['language']}")
    
    # Analyze
    result = await tool.execute(
        audio=f"data:audio/wav;base64,{audio_b64}",
        operation="analyze"
    )
    
    if result["success"]:
        print("✓ Audio analysis completed")
        print(f"  Format: {result['analysis']['format']}")
        print(f"  Channels: {result['analysis']['channels']}")


async def demo_video_processing():
    """Demonstrate video processing."""
    print("\n=== Video Processing Demo ===")
    
    tool = VideoProcessingTool()
    
    # Analyze video
    result = await tool.execute(
        video="sample_video.mp4",
        operation="analyze"
    )
    
    if result["success"]:
        print("✓ Video analysis completed")
        print(f"  Duration: {result['analysis']['video_info']['duration']}s")
        print(f"  Resolution: {result['analysis']['video_info']['resolution']}")
        print(f"  FPS: {result['analysis']['video_info']['fps']}")
    
    # Extract frames
    result = await tool.execute(
        video="sample_video.mp4",
        operation="extract_frames",
        frame_interval=10.0,
        output_path="demo_frames"
    )
    
    if result["success"]:
        print("✓ Frame extraction completed")
        print(f"  Extracted frames: {result['metadata']['num_frames']}")
        print(f"  Output directory: {result['metadata']['output_directory']}")


async def main():
    """Run all demos."""
    print("Multimodal Tools Demonstration")
    print("=" * 50)
    
    await demo_image_generation()
    await demo_image_analysis()
    await demo_audio_processing()
    await demo_video_processing()
    
    print("\n" + "=" * 50)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())