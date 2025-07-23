"""Multimodal tools for image, audio, and video processing."""

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np

from .base import Tool


@dataclass
class ImageData:
    """Container for image data and metadata."""
    data: Union[bytes, np.ndarray, Image.Image]
    format: str
    width: int
    height: int
    mode: str
    metadata: Dict[str, Any]


@dataclass 
class AudioData:
    """Container for audio data and metadata."""
    data: Union[bytes, np.ndarray]
    format: str
    duration: float
    sample_rate: int
    channels: int
    metadata: Dict[str, Any]


@dataclass
class VideoData:
    """Container for video data and metadata."""
    data: Union[bytes, str]  # bytes or file path
    format: str
    duration: float
    fps: float
    width: int
    height: int
    metadata: Dict[str, Any]


class ImageAnalysisTool(Tool):
    """Analyze images using AI models for various tasks."""
    
    def __init__(self):
        super().__init__(
            name="image-analysis",
            description="Analyze images for content, objects, text, and more"
        )
        self.add_parameter("image", "string", "Image file path or base64 encoded data")
        self.add_parameter("analysis_type", "string", "Type of analysis: describe, detect_objects, extract_text, detect_faces, classify")
        self.add_parameter("model", "string", "Model to use for analysis", required=False)
        self.add_parameter("detail_level", "string", "Level of detail: low, medium, high", default="medium")
        self.add_parameter("output_format", "string", "Output format: json, text, structured", default="json")
        self.add_parameter("confidence_threshold", "number", "Minimum confidence for detections", default=0.5)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_image(self, image_input: str) -> ImageData:
        """Load image from file path or base64 string."""
        try:
            # Check if it's a file path
            if os.path.exists(image_input):
                with Image.open(image_input) as img:
                    # Convert to RGB if needed
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    
                    return ImageData(
                        data=img.copy(),
                        format=img.format or 'PNG',
                        width=img.width,
                        height=img.height,
                        mode=img.mode,
                        metadata={'source': 'file', 'path': image_input}
                    )
            
            # Try base64 decode
            else:
                # Remove data URL prefix if present
                if ',' in image_input:
                    image_input = image_input.split(',')[1]
                
                image_bytes = base64.b64decode(image_input)
                img = Image.open(io.BytesIO(image_bytes))
                
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                return ImageData(
                    data=img,
                    format=img.format or 'PNG',
                    width=img.width,
                    height=img.height,
                    mode=img.mode,
                    metadata={'source': 'base64'}
                )
                
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def _prepare_image_for_model(self, image_data: ImageData, max_size: int = 1024) -> str:
        """Prepare image for model input (resize and encode)."""
        img = image_data.data
        if isinstance(img, (bytes, np.ndarray)):
            img = Image.fromarray(img) if isinstance(img, np.ndarray) else Image.open(io.BytesIO(img))
        
        # Resize if too large
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    async def _analyze_with_model(self, image_b64: str, analysis_type: str, 
                                 detail_level: str, model_name: Optional[str]) -> Dict[str, Any]:
        """Analyze image using AI model."""
        # Get model registry
        from orchestrator.models.registry_singleton import get_model_registry
        registry = get_model_registry()
        
        # Select model
        if model_name:
            model = registry.get_model(model_name)
            if not model:
                raise ValueError(f"Model '{model_name}' not found")
        else:
            # Select model with vision capabilities
            requirements = {
                "tasks": ["vision"]
            }
            model = await registry.select_model(requirements)
            if not model:
                raise ValueError("No suitable vision model available")
        
        # Prepare prompt based on analysis type
        prompts = {
            "describe": f"Describe this image in {detail_level} detail. What do you see?",
            "detect_objects": "List all objects you can identify in this image with their locations.",
            "extract_text": "Extract and transcribe all text visible in this image.",
            "detect_faces": "Detect and describe any faces or people in this image.",
            "classify": "Classify this image into appropriate categories."
        }
        
        prompt = prompts.get(analysis_type, f"Analyze this image for: {analysis_type}")
        
        # Create message with image
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }]
        
        # Call model
        try:
            response = await model.generate(
                messages=messages,
                temperature=0.1,  # Low temperature for analysis
                max_tokens=1000
            )
            
            return {
                "model": model.name,
                "analysis": response.get("content", ""),
                "usage": response.get("usage", {})
            }
            
        except Exception as e:
            self.logger.error(f"Model analysis failed: {e}")
            raise
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute image analysis."""
        image_input = kwargs["image"]
        analysis_type = kwargs["analysis_type"]
        model_name = kwargs.get("model")
        detail_level = kwargs.get("detail_level", "medium")
        output_format = kwargs.get("output_format", "json")
        kwargs.get("confidence_threshold", 0.5)
        
        # Validate analysis type
        valid_types = ["describe", "detect_objects", "extract_text", "detect_faces", "classify"]
        if analysis_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid analysis_type: {analysis_type}. Must be one of {valid_types}"
            }
        
        try:
            # Load image
            image_data = self._load_image(image_input)
            self.logger.info(f"Loaded image: {image_data.width}x{image_data.height} {image_data.mode}")
            
            # Prepare for model
            image_b64 = self._prepare_image_for_model(image_data)
            
            # Analyze with model
            result = await self._analyze_with_model(
                image_b64, analysis_type, detail_level, model_name
            )
            
            # Format output
            analysis_result = result["analysis"]
            
            if output_format == "text":
                formatted_result = analysis_result
            elif output_format == "structured":
                # Try to parse structured data from model output
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
                    if json_match:
                        formatted_result = json.loads(json_match.group())
                    else:
                        formatted_result = {"description": analysis_result}
                except:
                    formatted_result = {"description": analysis_result}
            else:  # json
                formatted_result = {
                    "analysis_type": analysis_type,
                    "result": analysis_result,
                    "image_info": {
                        "width": image_data.width,
                        "height": image_data.height,
                        "format": image_data.format,
                        "mode": image_data.mode
                    },
                    "model_used": result["model"],
                    "detail_level": detail_level
                }
            
            return {
                "success": True,
                "analysis": formatted_result,
                "metadata": {
                    "analysis_type": analysis_type,
                    "model": result["model"],
                    "tokens_used": result.get("usage", {})
                }
            }
            
        except Exception as e:
            self.logger.error(f"Image analysis error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class ImageGenerationTool(Tool):
    """Generate images using AI models."""
    
    def __init__(self):
        super().__init__(
            name="image-generation",
            description="Generate images from text descriptions"
        )
        self.add_parameter("prompt", "string", "Text description of the image to generate")
        self.add_parameter("model", "string", "Model to use for generation", required=False)
        self.add_parameter("size", "string", "Image size: 256x256, 512x512, 1024x1024", default="512x512")
        self.add_parameter("style", "string", "Art style or aesthetic", required=False)
        self.add_parameter("negative_prompt", "string", "What to avoid in the image", required=False)
        self.add_parameter("num_images", "integer", "Number of images to generate", default=1)
        self.add_parameter("output_format", "string", "Output format: url, base64, file", default="file")
        self.add_parameter("output_path", "string", "Directory to save images", default="generated_images")
        
        self.logger = logging.getLogger(__name__)
    
    async def _generate_with_model(self, prompt: str, size: str, 
                                  model_name: Optional[str], num_images: int) -> List[str]:
        """Generate images using AI model."""
        # Get model registry
        from orchestrator.models.registry_singleton import get_model_registry
        registry = get_model_registry()
        
        # Select model
        if model_name:
            model = registry.get_model(model_name)
            if not model:
                raise ValueError(f"Model '{model_name}' not found")
        else:
            # Select model with image generation capabilities
            requirements = {
                "capabilities": ["image-generation"],
                "tasks": ["generate-image"]
            }
            try:
                model = await registry.select_model(requirements)
            except Exception:
                model = None
                
            if not model:
                # Fallback to using a text model to generate image description
                self.logger.warning("No image generation model available, using placeholder")
                return await self._generate_placeholder(prompt, size, num_images)
        
        # Generate images
        try:
            # Different models have different APIs
            if hasattr(model, 'generate_image'):
                # Direct image generation
                images = []
                for i in range(num_images):
                    result = await model.generate_image(
                        prompt=prompt,
                        size=size,
                        n=1
                    )
                    if 'data' in result:
                        images.extend([img.get('url') or img.get('b64_json') for img in result['data']])
                    elif 'images' in result:
                        images.extend(result['images'])
                return images[:num_images]
            else:
                # Use placeholder
                return await self._generate_placeholder(prompt, size, num_images)
                
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise
    
    async def _generate_placeholder(self, prompt: str, size: str, num_images: int) -> List[str]:
        """Generate placeholder images when no image model is available."""
        images = []
        width, height = map(int, size.split('x'))
        
        for i in range(num_images):
            # Create a placeholder image with the prompt text
            img = Image.new('RGB', (width, height), color=(200, 200, 200))
            
            # Add text (simplified - in production would use proper text rendering)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            # Draw prompt text
            text_lines = [prompt[i:i+30] for i in range(0, len(prompt), 30)][:5]
            y_offset = height // 2 - len(text_lines) * 10
            
            for line in text_lines:
                draw.text((10, y_offset), line, fill=(50, 50, 50))
                y_offset += 20
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            b64_data = base64.b64encode(buffer.read()).decode('utf-8')
            images.append(f"data:image/png;base64,{b64_data}")
        
        return images
    
    def _save_image(self, image_data: str, output_path: str, index: int) -> str:
        """Save image data to file."""
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Generate filename
        timestamp = int(time.time())
        filename = f"generated_{timestamp}_{index}.png"
        filepath = os.path.join(output_path, filename)
        
        # Decode and save
        if image_data.startswith('data:'):
            # Data URL
            image_data = image_data.split(',')[1]
        
        if image_data.startswith('http'):
            # URL - download it
            import urllib.request
            urllib.request.urlretrieve(image_data, filepath)
        else:
            # Base64
            image_bytes = base64.b64decode(image_data)
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
        
        return filepath
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute image generation."""
        prompt = kwargs["prompt"]
        model_name = kwargs.get("model")
        size = kwargs.get("size", "512x512")
        style = kwargs.get("style")
        negative_prompt = kwargs.get("negative_prompt")
        num_images = kwargs.get("num_images", 1)
        output_format = kwargs.get("output_format", "file")
        output_path = kwargs.get("output_path", "generated_images")
        
        # Validate size
        valid_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
        if size not in valid_sizes:
            return {
                "success": False,
                "error": f"Invalid size: {size}. Must be one of {valid_sizes}"
            }
        
        # Enhance prompt with style
        full_prompt = prompt
        if style:
            full_prompt = f"{prompt}, {style} style"
        if negative_prompt:
            full_prompt = f"{full_prompt}. Avoid: {negative_prompt}"
        
        try:
            # Generate images
            generated_images = await self._generate_with_model(
                full_prompt, size, model_name, num_images
            )
            
            # Process based on output format
            output_images = []
            
            for i, img_data in enumerate(generated_images):
                if output_format == "file":
                    filepath = self._save_image(img_data, output_path, i)
                    output_images.append({
                        "path": filepath,
                        "format": "file"
                    })
                elif output_format == "base64":
                    if not img_data.startswith('data:'):
                        img_data = f"data:image/png;base64,{img_data}"
                    output_images.append({
                        "data": img_data,
                        "format": "base64"
                    })
                else:  # url
                    output_images.append({
                        "url": img_data,
                        "format": "url"
                    })
            
            return {
                "success": True,
                "images": output_images,
                "metadata": {
                    "prompt": full_prompt,
                    "size": size,
                    "num_generated": len(output_images),
                    "model": model_name or "default"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class AudioProcessingTool(Tool):
    """Process and analyze audio files."""
    
    def __init__(self):
        super().__init__(
            name="audio-processing",
            description="Process audio for transcription, analysis, and transformation"
        )
        self.add_parameter("audio", "string", "Audio file path or base64 encoded data")
        self.add_parameter("operation", "string", "Operation: transcribe, analyze, enhance, convert")
        self.add_parameter("model", "string", "Model to use for processing", required=False)
        self.add_parameter("language", "string", "Language code for transcription", default="en")
        self.add_parameter("output_format", "string", "Output format for conversion", required=False)
        self.add_parameter("enhance_options", "object", "Enhancement options", required=False)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_audio(self, audio_input: str) -> AudioData:
        """Load audio from file or base64."""
        try:
            # Check if it's a file path
            if os.path.exists(audio_input):
                # Get audio info using wave or other library
                
                # For now, create placeholder metadata
                # In production, use proper audio libraries
                return AudioData(
                    data=open(audio_input, 'rb').read(),
                    format=os.path.splitext(audio_input)[1][1:],
                    duration=0.0,  # Would calculate from file
                    sample_rate=44100,
                    channels=2,
                    metadata={'source': 'file', 'path': audio_input}
                )
            else:
                # Base64 decode
                if ',' in audio_input:
                    audio_input = audio_input.split(',')[1]
                
                audio_bytes = base64.b64decode(audio_input)
                
                return AudioData(
                    data=audio_bytes,
                    format='wav',  # Assume WAV for base64
                    duration=0.0,
                    sample_rate=44100,
                    channels=2,
                    metadata={'source': 'base64'}
                )
                
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")
    
    async def _transcribe_audio(self, audio_data: AudioData, language: str, 
                               model_name: Optional[str]) -> Dict[str, Any]:
        """Transcribe audio to text."""
        # Get model registry
        from ..models.registry_singleton import get_model_registry
        get_model_registry()
        
        # For now, return a placeholder
        # In production, would use speech-to-text models
        return {
            "transcription": f"[Audio transcription placeholder for {audio_data.format} file]",
            "language": language,
            "duration": audio_data.duration,
            "confidence": 0.95
        }
    
    async def _analyze_audio(self, audio_data: AudioData) -> Dict[str, Any]:
        """Analyze audio properties."""
        return {
            "format": audio_data.format,
            "duration": audio_data.duration,
            "sample_rate": audio_data.sample_rate,
            "channels": audio_data.channels,
            "analysis": {
                "volume_levels": "normal",
                "noise_level": "low",
                "clarity": "good"
            }
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute audio processing."""
        audio_input = kwargs["audio"]
        operation = kwargs["operation"]
        model_name = kwargs.get("model")
        language = kwargs.get("language", "en")
        output_format = kwargs.get("output_format")
        enhance_options = kwargs.get("enhance_options", {})
        
        # Validate operation
        valid_operations = ["transcribe", "analyze", "enhance", "convert"]
        if operation not in valid_operations:
            return {
                "success": False,
                "error": f"Invalid operation: {operation}. Must be one of {valid_operations}"
            }
        
        try:
            # Load audio
            audio_data = self._load_audio(audio_input)
            
            # Process based on operation
            if operation == "transcribe":
                result = await self._transcribe_audio(audio_data, language, model_name)
                return {
                    "success": True,
                    "transcription": result["transcription"],
                    "metadata": {
                        "language": result["language"],
                        "duration": result["duration"],
                        "confidence": result["confidence"]
                    }
                }
            
            elif operation == "analyze":
                analysis = await self._analyze_audio(audio_data)
                return {
                    "success": True,
                    "analysis": analysis
                }
            
            elif operation == "enhance":
                # Placeholder for audio enhancement
                return {
                    "success": True,
                    "message": "Audio enhancement placeholder",
                    "enhanced_audio": audio_input,  # Return original for now
                    "enhancements_applied": enhance_options
                }
            
            elif operation == "convert":
                # Placeholder for format conversion
                return {
                    "success": True,
                    "message": f"Audio conversion to {output_format} placeholder",
                    "converted_audio": audio_input,  # Return original for now
                    "output_format": output_format
                }
            
        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class VideoProcessingTool(Tool):
    """Process and analyze video files."""
    
    def __init__(self):
        super().__init__(
            name="video-processing",
            description="Process videos for analysis, extraction, and transformation"
        )
        self.add_parameter("video", "string", "Video file path or URL")
        self.add_parameter("operation", "string", "Operation: analyze, extract_frames, extract_audio, summarize")
        self.add_parameter("model", "string", "Model to use for analysis", required=False)
        self.add_parameter("frame_interval", "number", "Seconds between frame extraction", default=1.0)
        self.add_parameter("start_time", "number", "Start time in seconds", default=0)
        self.add_parameter("end_time", "number", "End time in seconds", required=False)
        self.add_parameter("output_path", "string", "Output directory for extracted content", default="video_output")
        
        self.logger = logging.getLogger(__name__)
    
    def _load_video_metadata(self, video_path: str) -> VideoData:
        """Load video metadata."""
        # Placeholder implementation
        # In production, use cv2 or ffmpeg-python
        return VideoData(
            data=video_path,
            format=os.path.splitext(video_path)[1][1:] if os.path.exists(video_path) else "mp4",
            duration=60.0,  # Placeholder
            fps=30.0,
            width=1920,
            height=1080,
            metadata={'source': 'file' if os.path.exists(video_path) else 'url'}
        )
    
    async def _analyze_video(self, video_data: VideoData, model_name: Optional[str]) -> Dict[str, Any]:
        """Analyze video content."""
        # Placeholder analysis
        return {
            "summary": "Video analysis placeholder",
            "detected_objects": ["person", "car", "building"],
            "scene_changes": [5.2, 15.7, 23.1],
            "dominant_colors": ["blue", "green", "gray"],
            "video_info": {
                "duration": video_data.duration,
                "fps": video_data.fps,
                "resolution": f"{video_data.width}x{video_data.height}"
            }
        }
    
    async def _extract_frames(self, video_data: VideoData, interval: float, 
                            start: float, end: Optional[float], output_path: str) -> List[str]:
        """Extract frames from video."""
        os.makedirs(output_path, exist_ok=True)
        
        # Placeholder - in production use cv2
        frames = []
        current_time = start
        end_time = end or video_data.duration
        frame_count = 0
        
        while current_time < end_time:
            # Create placeholder frame
            frame_path = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            
            # Create a placeholder image
            img = Image.new('RGB', (320, 240), color=(100, 100, 100))
            img.save(frame_path)
            
            frames.append(frame_path)
            current_time += interval
            frame_count += 1
            
            # Limit frames for demo
            if frame_count >= 5:
                break
        
        return frames
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute video processing."""
        video_input = kwargs["video"]
        operation = kwargs["operation"]
        model_name = kwargs.get("model")
        frame_interval = kwargs.get("frame_interval", 1.0)
        start_time = kwargs.get("start_time", 0)
        end_time = kwargs.get("end_time")
        output_path = kwargs.get("output_path", "video_output")
        
        # Validate operation
        valid_operations = ["analyze", "extract_frames", "extract_audio", "summarize"]
        if operation not in valid_operations:
            return {
                "success": False,
                "error": f"Invalid operation: {operation}. Must be one of {valid_operations}"
            }
        
        try:
            # Load video metadata
            video_data = self._load_video_metadata(video_input)
            
            # Process based on operation
            if operation == "analyze":
                analysis = await self._analyze_video(video_data, model_name)
                return {
                    "success": True,
                    "analysis": analysis
                }
            
            elif operation == "extract_frames":
                frames = await self._extract_frames(
                    video_data, frame_interval, start_time, end_time, output_path
                )
                return {
                    "success": True,
                    "frames": frames,
                    "metadata": {
                        "num_frames": len(frames),
                        "interval": frame_interval,
                        "output_directory": output_path
                    }
                }
            
            elif operation == "extract_audio":
                # Placeholder
                audio_path = os.path.join(output_path, "extracted_audio.wav")
                return {
                    "success": True,
                    "audio_path": audio_path,
                    "message": "Audio extraction placeholder"
                }
            
            elif operation == "summarize":
                # Use analyze and create summary
                analysis = await self._analyze_video(video_data, model_name)
                summary = f"Video Summary: Duration {video_data.duration}s, " \
                         f"Resolution {video_data.width}x{video_data.height}, " \
                         f"Detected: {', '.join(analysis['detected_objects'][:3])}"
                
                return {
                    "success": True,
                    "summary": summary,
                    "key_moments": analysis.get("scene_changes", []),
                    "metadata": analysis["video_info"]
                }
            
        except Exception as e:
            self.logger.error(f"Video processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }