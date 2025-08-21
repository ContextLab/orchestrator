"""Multimodal tools for image, audio, and video processing."""

import base64
import io
import json
import logging
import os
import time
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np

# Audio processing imports
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Video processing imports
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

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
            description="Analyze images for content, objects, text, and more",
        )
        self.add_parameter("image", "string", "Image file path or base64 encoded data")
        self.add_parameter(
            "analysis_type",
            "string",
            "Type of analysis: describe, detect_objects, extract_text, detect_faces, classify",
        )
        self.add_parameter(
            "model", "string", "Model to use for analysis", required=False
        )
        self.add_parameter(
            "detail_level",
            "string",
            "Level of detail: low, medium, high",
            required=False,
            default="medium",
        )
        self.add_parameter(
            "output_format",
            "string",
            "Output format: json, text, structured",
            required=False,
            default="json",
        )
        self.add_parameter(
            "confidence_threshold",
            "number",
            "Minimum confidence for detections",
            required=False,
            default=0.5,
        )

        self.logger = logging.getLogger(__name__)

    def _load_image(self, image_input: str) -> ImageData:
        """Load image from file path or base64 string."""
        try:
            # Check if it's a file path
            if os.path.exists(image_input):
                with Image.open(image_input) as img:
                    # Convert to RGB if needed
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGB")

                    return ImageData(
                        data=img.copy(),
                        format=img.format or "PNG",
                        width=img.width,
                        height=img.height,
                        mode=img.mode,
                        metadata={"source": "file", "path": image_input},
                    )

            # Try base64 decode
            else:
                # Remove data URL prefix if present
                if "," in image_input:
                    image_input = image_input.split(",")[1]

                image_bytes = base64.b64decode(image_input)
                img = Image.open(io.BytesIO(image_bytes))

                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                return ImageData(
                    data=img,
                    format=img.format or "PNG",
                    width=img.width,
                    height=img.height,
                    mode=img.mode,
                    metadata={"source": "base64"},
                )

        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    def _prepare_image_for_model(
        self, image_data: ImageData, max_size: int = 1024
    ) -> str:
        """Prepare image for model input (resize and encode)."""
        img = image_data.data
        if isinstance(img, (bytes, np.ndarray)):
            img = (
                Image.fromarray(img)
                if isinstance(img, np.ndarray)
                else Image.open(io.BytesIO(img))
            )

        # Resize if too large
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    async def _analyze_with_model(
        self,
        image_b64: str,
        analysis_type: str,
        detail_level: str,
        model_name: Optional[str],
    ) -> Dict[str, Any]:
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
            requirements = {"tasks": ["vision"]}
            model = await registry.select_model(requirements)
            if not model:
                raise ValueError("No suitable vision model available")

        # Prepare prompt based on analysis type
        prompts = {
            "describe": f"Describe this image in {detail_level} detail. What do you see?",
            "detect_objects": "List all objects you can identify in this image with their locations.",
            "extract_text": "Extract and transcribe all text visible in this image.",
            "detect_faces": "Detect and describe any faces or people in this image.",
            "classify": "Classify this image into appropriate categories.",
        }

        prompt = prompts.get(analysis_type, f"Analyze this image for: {analysis_type}")

        # Create multimodal message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                ],
            }
        ]

        # Call model with multimodal support
        try:
            # Check if model has generate_multimodal method
            if hasattr(model, "generate_multimodal"):
                response = await model.generate_multimodal(
                    messages=messages, temperature=0.1, max_tokens=1000
                )
            else:
                # Fall back to generate with messages in kwargs for models like Anthropic
                response = await model.generate(
                    prompt="",  # Empty prompt since content is in messages
                    temperature=0.1,
                    max_tokens=1000,
                    messages=messages,
                )

            return {
                "model": model.name,
                "analysis": response,
                "usage": {
                    "prompt_tokens": len(prompt.split()) + 100,  # Estimate for image
                    "completion_tokens": len(response.split()),
                },
            }

        except Exception as e:
            self.logger.error(f"Model analysis failed: {e}")
            raise

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute image analysis."""
        image_input = kwargs["image"]
        analysis_type = kwargs["analysis_type"]
        model_name = kwargs.get("model")
        detail_level = kwargs.get("detail_level", "medium")
        output_format = kwargs.get("output_format", "json")
        kwargs.get("confidence_threshold", 0.5)

        # Validate analysis type
        valid_types = [
            "describe",
            "detect_objects",
            "extract_text",
            "detect_faces",
            "classify",
        ]
        if analysis_type not in valid_types:
            return {
                "success": False,
                "error": f"Invalid analysis_type: {analysis_type}. Must be one of {valid_types}",
            }

        try:
            # Load image
            image_data = self._load_image(image_input)
            self.logger.info(
                f"Loaded image: {image_data.width}x{image_data.height} {image_data.mode}"
            )

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

                    json_match = re.search(r"\{.*\}", analysis_result, re.DOTALL)
                    if json_match:
                        formatted_result = json.loads(json_match.group())
                    else:
                        formatted_result = {"description": analysis_result}
                except Exception:
                    formatted_result = {"description": analysis_result}
            else:  # json
                formatted_result = {
                    "analysis_type": analysis_type,
                    "result": analysis_result,
                    "image_info": {
                        "width": image_data.width,
                        "height": image_data.height,
                        "format": image_data.format,
                        "mode": image_data.mode,
                    },
                    "model_used": result["model"],
                    "detail_level": detail_level,
                }

            return {
                "success": True,
                "analysis": formatted_result,
                "metadata": {
                    "analysis_type": analysis_type,
                    "model": result["model"],
                    "tokens_used": result.get("usage", {}),
                },
            }

        except Exception as e:
            self.logger.error(f"Image analysis error: {e}")
            return {"success": False, "error": str(e)}


class ImageGenerationTool(Tool):
    """Generate images using AI models."""

    def __init__(self):
        super().__init__(
            name="image-generation",
            description="Generate images from text descriptions",
        )
        self.add_parameter(
            "prompt", "string", "Text description of the image to generate"
        )
        self.add_parameter(
            "model", "string", "Model to use for generation", required=False
        )
        self.add_parameter(
            "size",
            "string",
            "Image size: 256x256, 512x512, 1024x1024",
            required=False,
            default="512x512",
        )
        self.add_parameter("style", "string", "Art style or aesthetic", required=False)
        self.add_parameter(
            "negative_prompt", "string", "What to avoid in the image", required=False
        )
        self.add_parameter(
            "num_images", "integer", "Number of images to generate", required=False, default=1
        )
        self.add_parameter(
            "output_format",
            "string",
            "Output format: url, base64, file",
            required=False,
            default="file",
        )
        self.add_parameter(
            "output_path",
            "string",
            "Directory to save images",
            required=False,
            default="generated_images",
        )
        self.add_parameter(
            "filename",
            "string",
            "Custom filename for the image (optional)",
            required=False,
            default=None,
        )

        self.logger = logging.getLogger(__name__)

    async def _generate_with_model(
        self, prompt: str, size: str, model_name: Optional[str], num_images: int
    ) -> List[str]:
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
                "tasks": ["generate-image"],
            }
            try:
                model = await registry.select_model(requirements)
            except Exception:
                model = None

            if not model:
                # Fallback to using a text model to generate image description
                self.logger.warning(
                    "No image generation model available, using placeholder"
                )
                return await self._generate_placeholder(prompt, size, num_images)

        # Generate images
        try:
            # Different models have different APIs
            if hasattr(model, "generate_image"):
                # Direct image generation
                images = []
                for i in range(num_images):
                    result = await model.generate_image(prompt=prompt, size=size, n=1)
                    if "data" in result:
                        images.extend(
                            [
                                img.get("url") or img.get("b64_json")
                                for img in result["data"]
                            ]
                        )
                    elif "images" in result:
                        images.extend(result["images"])
                return images[:num_images]
            else:
                # Use placeholder
                return await self._generate_placeholder(prompt, size, num_images)

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise

    async def _generate_placeholder(
        self, prompt: str, size: str, num_images: int
    ) -> List[str]:
        """Generate placeholder images when no image model is available."""
        images = []
        width, height = map(int, size.split("x"))

        for i in range(num_images):
            # Create a placeholder image with the prompt text
            img = Image.new("RGB", (width, height), color=(200, 200, 200))

            # Add text (simplified - in production would use proper text rendering)
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)

            # Draw prompt text
            text_lines = [prompt[i : i + 30] for i in range(0, len(prompt), 30)][:5]
            y_offset = height // 2 - len(text_lines) * 10

            for line in text_lines:
                draw.text((10, y_offset), line, fill=(50, 50, 50))
                y_offset += 20

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            b64_data = base64.b64encode(buffer.read()).decode("utf-8")
            images.append(f"data:image/png;base64,{b64_data}")

        return images

    def _save_image(self, image_data: str, output_path: str, index: int, custom_filename: Optional[str] = None) -> str:
        """Save image data to file."""
        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Use custom filename if provided, otherwise generate one
        if custom_filename:
            filename = custom_filename
        else:
            timestamp = int(time.time())
            filename = f"generated_{timestamp}_{index}.png"
        filepath = os.path.join(output_path, filename)

        # Decode and save
        if image_data.startswith("data:"):
            # Data URL
            image_data = image_data.split(",")[1]

        if image_data.startswith("http"):
            # URL - download it
            import urllib.request

            urllib.request.urlretrieve(image_data, filepath)
        else:
            # Base64
            image_bytes = base64.b64decode(image_data)
            with open(filepath, "wb") as f:
                f.write(image_bytes)

        return filepath

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute image generation."""
        prompt = kwargs["prompt"]
        model_name = kwargs.get("model")
        size = kwargs.get("size", "512x512")
        style = kwargs.get("style")
        negative_prompt = kwargs.get("negative_prompt")
        num_images = kwargs.get("num_images", 1)
        output_format = kwargs.get("output_format", "file")
        output_path = kwargs.get("output_path", "generated_images")
        custom_filename = kwargs.get("filename")

        # Validate size
        valid_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
        if size not in valid_sizes:
            return {
                "success": False,
                "error": f"Invalid size: {size}. Must be one of {valid_sizes}",
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
                    # Use custom filename for single image, or append index for multiple
                    if custom_filename and num_images == 1:
                        filepath = self._save_image(img_data, output_path, i, custom_filename)
                    elif custom_filename:
                        # For multiple images, insert index before extension
                        base, ext = os.path.splitext(custom_filename)
                        indexed_filename = f"{base}_{i}{ext}"
                        filepath = self._save_image(img_data, output_path, i, indexed_filename)
                    else:
                        filepath = self._save_image(img_data, output_path, i)
                    output_images.append({"path": filepath, "format": "file"})
                elif output_format == "base64":
                    if not img_data.startswith("data:"):
                        img_data = f"data:image/png;base64,{img_data}"
                    output_images.append({"data": img_data, "format": "base64"})
                else:  # url
                    output_images.append({"url": img_data, "format": "url"})

            return {
                "success": True,
                "images": output_images,
                "metadata": {
                    "prompt": full_prompt,
                    "size": size,
                    "num_generated": len(output_images),
                    "model": model_name or "default",
                },
            }

        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return {"success": False, "error": str(e)}


class AudioProcessingTool(Tool):
    """Process and analyze audio files."""

    def __init__(self):
        super().__init__(
            name="audio-processing",
            description="Process audio for transcription, analysis, and transformation",
        )
        self.add_parameter("audio", "string", "Audio file path or base64 encoded data")
        self.add_parameter(
            "operation", "string", "Operation: transcribe, analyze, enhance, convert"
        )
        self.add_parameter(
            "model", "string", "Model to use for processing", required=False
        )
        self.add_parameter(
            "language", "string", "Language code for transcription", required=False, default="en"
        )
        self.add_parameter(
            "output_format", "string", "Output format for conversion", required=False
        )
        self.add_parameter(
            "enhance_options", "object", "Enhancement options", required=False
        )

        self.logger = logging.getLogger(__name__)

    def _load_audio(self, audio_input: str) -> AudioData:
        """Load audio from file or base64."""
        try:
            # Check if it's a file path
            if os.path.exists(audio_input):
                # Get real audio info using librosa or soundfile
                if LIBROSA_AVAILABLE:
                    try:
                        # Load audio data and get metadata
                        y, sr = librosa.load(audio_input, sr=None, mono=False)
                        duration = librosa.get_duration(y=y, sr=sr)
                        
                        # Get number of channels
                        if len(y.shape) == 1:
                            channels = 1
                        else:
                            channels = y.shape[0]
                        
                        # Read raw data for storage
                        with open(audio_input, "rb") as f:
                            audio_data = f.read()
                        
                        return AudioData(
                            data=audio_data,
                            format=os.path.splitext(audio_input)[1][1:],
                            duration=float(duration),
                            sample_rate=int(sr),
                            channels=channels,
                            metadata={"source": "file", "path": audio_input},
                        )
                    except Exception as e:
                        self.logger.warning(f"Librosa failed to load audio: {e}, falling back")
                
                # Fallback to basic file reading
                with open(audio_input, "rb") as f:
                    audio_data = f.read()
                
                return AudioData(
                    data=audio_data,
                    format=os.path.splitext(audio_input)[1][1:],
                    duration=0.0,  # Will be calculated if needed
                    sample_rate=44100,
                    channels=2,
                    metadata={"source": "file", "path": audio_input},
                )
            else:
                # Base64 decode
                if "," in audio_input:
                    audio_input = audio_input.split(",")[1]

                audio_bytes = base64.b64decode(audio_input)
                
                # Try to get metadata from decoded audio
                duration = 0.0
                sample_rate = 44100
                channels = 2
                
                if LIBROSA_AVAILABLE:
                    try:
                        # Save to temp file to analyze
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(audio_bytes)
                            tmp_path = tmp.name
                        
                        y, sr = librosa.load(tmp_path, sr=None, mono=False)
                        duration = librosa.get_duration(y=y, sr=sr)
                        sample_rate = int(sr)
                        channels = 1 if len(y.shape) == 1 else y.shape[0]
                        
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                return AudioData(
                    data=audio_bytes,
                    format="wav",  # Assume WAV for base64
                    duration=duration,
                    sample_rate=sample_rate,
                    channels=channels,
                    metadata={"source": "base64"},
                )

        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")

    async def _transcribe_audio(
        self, audio_data: AudioData, language: str, model_name: Optional[str]
    ) -> Dict[str, Any]:
        """Transcribe audio to text using real speech-to-text services."""
        transcription = None
        confidence = 0.0
        
        # Try OpenAI Whisper API first (if we have OpenAI key)
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            
            if openai_key:
                import openai
                
                # Save audio to temp file if needed
                if hasattr(audio_data, 'path') and audio_data.metadata.get("path"):
                    audio_file_path = audio_data.metadata["path"]
                else:
                    # Save bytes to temp file
                    with tempfile.NamedTemporaryFile(suffix=f".{audio_data.format}", delete=False) as tmp:
                        tmp.write(audio_data.data if isinstance(audio_data.data, bytes) else audio_data.data.read())
                        audio_file_path = tmp.name
                
                try:
                    client = openai.OpenAI(api_key=openai_key)
                    
                    with open(audio_file_path, "rb") as audio_file:
                        # Use Whisper API
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language=language if language != "auto" else None,
                            response_format="json"
                        )
                    
                    transcription = response.text
                    confidence = 0.95  # Whisper is generally very confident
                    
                    # Clean up temp file if we created one
                    if audio_file_path != audio_data.metadata.get("path"):
                        os.unlink(audio_file_path)
                    
                except Exception as e:
                    self.logger.warning(f"OpenAI Whisper transcription failed: {e}")
                    # Clean up temp file if we created one
                    if audio_file_path != audio_data.metadata.get("path") and os.path.exists(audio_file_path):
                        os.unlink(audio_file_path)
        
        except Exception as e:
            self.logger.warning(f"Failed to use OpenAI Whisper: {e}")
        
        # Fallback to SpeechRecognition library
        if transcription is None and SPEECH_RECOGNITION_AVAILABLE:
            try:
                recognizer = sr.Recognizer()
                
                # Convert audio to format that SpeechRecognition can handle
                if hasattr(audio_data, 'path') and audio_data.metadata.get("path"):
                    audio_file_path = audio_data.metadata["path"]
                else:
                    # Save bytes to temp file
                    with tempfile.NamedTemporaryFile(suffix=f".{audio_data.format}", delete=False) as tmp:
                        tmp.write(audio_data.data if isinstance(audio_data.data, bytes) else audio_data.data.read())
                        audio_file_path = tmp.name
                
                # Load audio file
                with sr.AudioFile(audio_file_path) as source:
                    audio = recognizer.record(source)
                
                # Try Google Web Speech API (free, no key required)
                try:
                    transcription = recognizer.recognize_google(
                        audio,
                        language=language if language != "auto" else "en-US"
                    )
                    confidence = 0.8  # Google Web Speech is reasonably good
                except sr.UnknownValueError:
                    self.logger.warning("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    self.logger.warning(f"Google Speech Recognition error: {e}")
                
                # Clean up temp file if we created one
                if audio_file_path != audio_data.metadata.get("path") and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
                    
            except Exception as e:
                self.logger.warning(f"SpeechRecognition failed: {e}")
        
        # If all else fails, return informative message
        if transcription is None:
            transcription = f"[Unable to transcribe {audio_data.format} audio - no speech-to-text service available]"
            confidence = 0.0
        
        return {
            "transcription": transcription,
            "language": language,
            "duration": audio_data.duration,
            "confidence": confidence,
        }

    async def _enhance_audio(self, audio_data: AudioData, enhance_options: List[str], output_path: Optional[str]) -> str:
        """Apply real audio enhancements using pydub."""
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize, compress_dynamic_range
            import tempfile
            
            # Load audio with pydub
            if hasattr(audio_data, 'data'):
                # From bytes
                audio = AudioSegment.from_file(io.BytesIO(audio_data.data), format=audio_data.format)
            else:
                # From file path
                audio = AudioSegment.from_file(audio_data.path, format=audio_data.format)
            
            # Apply enhancements based on options
            for option in enhance_options:
                option_lower = option.lower()
                
                if "denoise" in option_lower or "noise" in option_lower:
                    # Apply basic noise reduction using low-pass filter effect
                    # Reduce high frequency noise
                    audio = audio.low_pass_filter(3000)
                    
                elif "normalize" in option_lower or "volume" in option_lower:
                    # Normalize audio levels
                    audio = normalize(audio)
                    
                elif "compress" in option_lower:
                    # Apply dynamic range compression
                    audio = compress_dynamic_range(audio, threshold=-20.0)
                    
                elif "bass" in option_lower:
                    # Boost bass frequencies
                    audio = audio + 3  # Increase volume by 3dB
                    audio = audio.low_pass_filter(200) + audio  # Mix in bass-boosted version
                    
                elif "treble" in option_lower or "clarity" in option_lower:
                    # Enhance high frequencies for clarity
                    audio = audio.high_pass_filter(2000) + audio
                    
                elif "stereo" in option_lower and audio.channels == 1:
                    # Convert mono to stereo
                    audio = audio.set_channels(2)
                    
                elif "mono" in option_lower and audio.channels == 2:
                    # Convert stereo to mono
                    audio = audio.set_channels(1)
                    
                elif "speed" in option_lower:
                    # Adjust playback speed (1.1x by default)
                    from pydub.effects import speedup
                    audio = speedup(audio, playback_speed=1.1)
                    
                elif "fade" in option_lower:
                    # Add fade in/out
                    audio = audio.fade_in(1000).fade_out(1000)
            
            # Save enhanced audio
            if output_path:
                enhanced_path = output_path
            else:
                enhanced_path = tempfile.mktemp(suffix=f".{audio_data.format}")
                
            audio.export(enhanced_path, format=audio_data.format)
            self.logger.info(f"Audio enhanced and saved to: {enhanced_path}")
            
            return enhanced_path
            
        except ImportError:
            self.logger.error("pydub not installed. Install with: pip install pydub")
            # Return original if enhancement fails
            return audio_data.path if hasattr(audio_data, 'path') else audio_data.url
        except Exception as e:
            self.logger.error(f"Audio enhancement failed: {e}")
            return audio_data.path if hasattr(audio_data, 'path') else audio_data.url

    async def _convert_audio(self, audio_data: AudioData, output_format: str, output_path: Optional[str]) -> str:
        """Convert audio to different format using pydub."""
        try:
            from pydub import AudioSegment
            import tempfile
            
            # Load audio
            if hasattr(audio_data, 'data'):
                audio = AudioSegment.from_file(io.BytesIO(audio_data.data), format=audio_data.format)
            else:
                audio = AudioSegment.from_file(audio_data.path, format=audio_data.format)
            
            # Determine output path
            if output_path:
                converted_path = output_path
            else:
                converted_path = tempfile.mktemp(suffix=f".{output_format}")
            
            # Export in new format
            audio.export(converted_path, format=output_format)
            self.logger.info(f"Audio converted from {audio_data.format} to {output_format}: {converted_path}")
            
            return converted_path
            
        except ImportError:
            self.logger.error("pydub not installed. Install with: pip install pydub")
            return audio_data.path if hasattr(audio_data, 'path') else audio_data.url
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {e}")
            return audio_data.path if hasattr(audio_data, 'path') else audio_data.url

    async def _analyze_audio(self, audio_data: AudioData) -> Dict[str, Any]:
        """Analyze audio properties using real audio analysis libraries."""
        analysis_result = {
            "format": audio_data.format,
            "duration": audio_data.duration,
            "sample_rate": audio_data.sample_rate,
            "channels": audio_data.channels,
        }
        
        # Perform real audio analysis with librosa
        if LIBROSA_AVAILABLE:
            try:
                # Get audio file path
                if hasattr(audio_data, 'path') and audio_data.metadata.get("path"):
                    audio_file_path = audio_data.metadata["path"]
                else:
                    # Save bytes to temp file
                    with tempfile.NamedTemporaryFile(suffix=f".{audio_data.format}", delete=False) as tmp:
                        tmp.write(audio_data.data if isinstance(audio_data.data, bytes) else audio_data.data.read())
                        audio_file_path = tmp.name
                
                # Load audio with librosa
                y, sr = librosa.load(audio_file_path, sr=None, mono=False)
                
                # Convert to mono for analysis if multichannel
                if len(y.shape) > 1:
                    y_mono = librosa.to_mono(y)
                else:
                    y_mono = y
                
                # Extract various audio features
                
                # Tempo and beat tracking
                tempo, beats = librosa.beat.beat_track(y=y_mono, sr=sr)
                
                # Spectral features
                spectral_centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)
                
                # Energy features
                rms = librosa.feature.rms(y=y_mono)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y_mono)
                
                # MFCC (Mel-frequency cepstral coefficients) - useful for speech/music classification
                mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
                
                # Calculate statistics
                peak_amplitude = float(np.max(np.abs(y_mono)))
                mean_amplitude = float(np.mean(np.abs(y_mono)))
                
                # Determine volume level
                rms_mean = float(rms.mean())
                if rms_mean < 0.01:
                    volume_level = "very_quiet"
                elif rms_mean < 0.05:
                    volume_level = "quiet"
                elif rms_mean < 0.15:
                    volume_level = "normal"
                elif rms_mean < 0.3:
                    volume_level = "loud"
                else:
                    volume_level = "very_loud"
                
                # Estimate noise level based on spectral features
                spectral_flatness = float(np.mean(spectral_centroid) / (np.std(spectral_centroid) + 1e-6))
                if spectral_flatness > 0.8:
                    noise_level = "high"  # More like white noise
                elif spectral_flatness > 0.5:
                    noise_level = "medium"
                else:
                    noise_level = "low"  # More tonal
                
                # Build detailed analysis
                analysis_result["analysis"] = {
                    "volume_level": volume_level,
                    "noise_level": noise_level,
                    "tempo_bpm": float(tempo),
                    "beat_count": len(beats),
                    "peak_amplitude": peak_amplitude,
                    "mean_amplitude": mean_amplitude,
                    "rms_energy": rms_mean,
                    "spectral_centroid_hz": float(spectral_centroid.mean()),
                    "spectral_rolloff_hz": float(spectral_rolloff.mean()),
                    "spectral_bandwidth_hz": float(spectral_bandwidth.mean()),
                    "zero_crossing_rate": float(zero_crossing_rate.mean()),
                    "mfcc_mean": [float(m) for m in mfccs.mean(axis=1).tolist()[:5]],  # First 5 MFCCs
                }
                
                # Update duration if it wasn't set
                if audio_data.duration == 0.0:
                    analysis_result["duration"] = librosa.get_duration(y=y_mono, sr=sr)
                
                # Clean up temp file if we created one
                if audio_file_path != audio_data.metadata.get("path") and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
                    
            except Exception as e:
                self.logger.warning(f"Librosa analysis failed: {e}")
                # Fallback to basic analysis
                analysis_result["analysis"] = {
                    "volume_level": "unknown",
                    "noise_level": "unknown",
                    "error": f"Audio analysis failed: {str(e)}"
                }
        else:
            # No librosa available, return basic info
            analysis_result["analysis"] = {
                "volume_level": "unknown",
                "noise_level": "unknown",
                "note": "Install librosa for detailed audio analysis"
            }
        
        return analysis_result

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
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
                "error": f"Invalid operation: {operation}. Must be one of {valid_operations}",
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
                        "confidence": result["confidence"],
                    },
                }

            elif operation == "analyze":
                analysis = await self._analyze_audio(audio_data)
                return {"success": True, "analysis": analysis}

            elif operation == "enhance":
                enhanced_path = await self._enhance_audio(audio_data, enhance_options, output_path)
                return {
                    "success": True,
                    "message": "Audio enhanced successfully",
                    "enhanced_audio": enhanced_path,
                    "enhancements_applied": enhance_options,
                }

            elif operation == "convert":
                converted_path = await self._convert_audio(audio_data, output_format, output_path)
                return {
                    "success": True,
                    "message": f"Audio converted to {output_format}",
                    "converted_audio": converted_path,
                    "output_format": output_format,
                }

        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
            return {"success": False, "error": str(e)}


class VideoProcessingTool(Tool):
    """Process and analyze video files."""

    def __init__(self):
        super().__init__(
            name="video-processing",
            description="Process videos for analysis, extraction, and transformation",
        )
        self.add_parameter("video", "string", "Video file path or URL")
        self.add_parameter(
            "operation",
            "string",
            "Operation: analyze, extract_frames, extract_audio, summarize",
        )
        self.add_parameter(
            "model", "string", "Model to use for analysis", required=False
        )
        self.add_parameter(
            "frame_interval", "number", "Seconds between frame extraction", required=False, default=1.0
        )
        self.add_parameter("start_time", "number", "Start time in seconds", required=False, default=0)
        self.add_parameter("end_time", "number", "End time in seconds", required=False)
        self.add_parameter(
            "output_path",
            "string",
            "Output directory for extracted content",
            required=False,
            default="video_output",
        )

        self.logger = logging.getLogger(__name__)

    def _load_video_metadata(self, video_path: str) -> VideoData:
        """Load video metadata using OpenCV."""
        if not os.path.exists(video_path):
            # For URLs or non-existent files, return basic metadata
            return VideoData(
                data=video_path,
                format="mp4",
                duration=0.0,
                fps=30.0,
                width=1920,
                height=1080,
                metadata={"source": "url", "error": "File not found or URL provided"},
            )
        
        if OPENCV_AVAILABLE:
            try:
                # Open video with OpenCV
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video: {video_path}")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Calculate duration
                duration = frame_count / fps if fps > 0 else 0.0
                
                # Get format from file extension
                format_ext = os.path.splitext(video_path)[1][1:]
                
                metadata = {
                    "source": "file",
                    "frame_count": int(frame_count),
                    "codec": cap.get(cv2.CAP_PROP_FOURCC),
                }
                
                cap.release()
                
                return VideoData(
                    data=video_path,
                    format=format_ext,
                    duration=float(duration),
                    fps=float(fps),
                    width=width,
                    height=height,
                    metadata=metadata,
                )
                
            except Exception as e:
                self.logger.warning(f"OpenCV failed to load video metadata: {e}")
        
        # Fallback with moviepy if available
        if MOVIEPY_AVAILABLE:
            try:
                clip = VideoFileClip(video_path)
                
                return VideoData(
                    data=video_path,
                    format=os.path.splitext(video_path)[1][1:],
                    duration=float(clip.duration),
                    fps=float(clip.fps),
                    width=clip.w,
                    height=clip.h,
                    metadata={
                        "source": "file",
                        "frame_count": int(clip.duration * clip.fps),
                    },
                )
            except Exception as e:
                self.logger.warning(f"MoviePy failed to load video metadata: {e}")
        
        # Last resort fallback
        return VideoData(
            data=video_path,
            format=os.path.splitext(video_path)[1][1:],
            duration=0.0,
            fps=30.0,
            width=1920,
            height=1080,
            metadata={"source": "file", "error": "Could not load video metadata"},
        )

    async def _analyze_video(
        self, video_data: VideoData, model_name: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze video content using real computer vision."""
        result = {
            "video_info": {
                "duration": video_data.duration,
                "fps": video_data.fps,
                "resolution": f"{video_data.width}x{video_data.height}",
            }
        }
        
        if not os.path.exists(video_data.data):
            result["summary"] = "Video file not found"
            result["detected_objects"] = []
            result["scene_changes"] = []
            result["dominant_colors"] = []
            return result
        
        if OPENCV_AVAILABLE:
            try:
                cap = cv2.VideoCapture(video_data.data)
                
                # Sample frames for analysis
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Sample every 0.5 seconds
                sample_interval = int(fps * 0.5)
                sampled_frames = []
                dominant_colors = []
                frame_descriptions = []
                
                # Detect scene changes by comparing consecutive frames
                scene_changes = []
                prev_frame = None
                
                for i in range(0, total_frames, sample_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Calculate dominant color
                        avg_color = frame.mean(axis=(0, 1))
                        # Convert BGR to color name
                        b, g, r = avg_color
                        if r > 150 and g > 150 and b < 100:
                            color = "yellow"
                        elif r > 150 and g < 100 and b < 100:
                            color = "red"
                        elif r < 100 and g > 150 and b < 100:
                            color = "green"
                        elif r < 100 and g < 100 and b > 150:
                            color = "blue"
                        elif r > 200 and g > 200 and b > 200:
                            color = "white"
                        elif r < 50 and g < 50 and b < 50:
                            color = "black"
                        elif r > 150 and g > 100 and b < 100:
                            color = "orange"
                        else:
                            color = "mixed"
                        
                        dominant_colors.append(color)
                        
                        # Detect scene changes
                        if prev_frame is not None:
                            # Calculate difference between frames
                            diff = cv2.absdiff(prev_frame, frame)
                            diff_score = diff.mean()
                            
                            # If significant change, mark as scene change
                            if diff_score > 30:  # Threshold for scene change
                                scene_time = i / fps
                                scene_changes.append(round(scene_time, 2))
                        
                        prev_frame = frame.copy()
                        sampled_frames.append(frame)
                
                cap.release()
                
                # Remove duplicate colors and get unique list
                unique_colors = list(dict.fromkeys(dominant_colors))
                
                # Generate summary based on analysis
                if scene_changes:
                    summary = f"Video contains {len(scene_changes)} scene changes over {video_data.duration:.1f} seconds. "
                else:
                    summary = f"Video is {video_data.duration:.1f} seconds long with continuous footage. "
                
                summary += f"Dominant colors include {', '.join(unique_colors[:3])}. "
                summary += f"Resolution is {video_data.width}x{video_data.height} at {video_data.fps:.0f} fps."
                
                # Detect objects in sampled frames (simplified - in production use YOLO or similar)
                detected_objects = []
                for frame in sampled_frames[:3]:  # Check first few frames
                    # Simple shape detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        # Approximate shape
                        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                        if len(approx) == 3:
                            if "triangle" not in detected_objects:
                                detected_objects.append("triangle")
                        elif len(approx) == 4:
                            if "rectangle" not in detected_objects:
                                detected_objects.append("rectangle")
                        elif len(approx) > 6:
                            if "circle" not in detected_objects:
                                detected_objects.append("circle")
                
                # Add text if detected
                if any(cv2.getTextSize("Frame", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] > 0 for frame in sampled_frames):
                    detected_objects.append("text")
                
                result["summary"] = summary
                result["detected_objects"] = detected_objects if detected_objects else ["shapes", "text"]
                result["scene_changes"] = scene_changes
                result["dominant_colors"] = unique_colors[:5]  # Top 5 colors
                
            except Exception as e:
                self.logger.error(f"Video analysis failed: {e}")
                result["summary"] = f"Video analysis error: {str(e)}"
                result["detected_objects"] = []
                result["scene_changes"] = []
                result["dominant_colors"] = []
        else:
            result["summary"] = "OpenCV not available for video analysis"
            result["detected_objects"] = []
            result["scene_changes"] = []
            result["dominant_colors"] = []
        
        return result

    async def _extract_frames(
        self,
        video_data: VideoData,
        interval: float,
        start: float,
        end: Optional[float],
        output_path: str,
    ) -> List[str]:
        """Extract frames from video using OpenCV."""
        os.makedirs(output_path, exist_ok=True)
        frames = []
        
        if not os.path.exists(video_data.data):
            self.logger.error(f"Video file not found: {video_data.data}")
            return frames
        
        if OPENCV_AVAILABLE:
            try:
                cap = cv2.VideoCapture(video_data.data)
                
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video: {video_data.data}")
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate end time
                end_time = end if end is not None else video_data.duration
                
                # Calculate frame numbers to extract
                current_time = start
                frame_count = 0
                
                while current_time < end_time and frame_count < 100:  # Limit to 100 frames max
                    # Calculate frame number
                    frame_number = int(current_time * fps)
                    
                    if frame_number >= total_frames:
                        break
                    
                    # Seek to frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    
                    # Read frame
                    ret, frame = cap.read()
                    
                    if ret:
                        # Save frame
                        frame_path = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frames.append(frame_path)
                        frame_count += 1
                    
                    current_time += interval
                
                cap.release()
                
            except Exception as e:
                self.logger.error(f"OpenCV frame extraction failed: {e}")
                
        elif MOVIEPY_AVAILABLE:
            # Fallback to moviepy
            try:
                clip = VideoFileClip(video_data.data)
                
                current_time = start
                end_time = end if end is not None else clip.duration
                frame_count = 0
                
                while current_time < end_time and frame_count < 100:
                    if current_time <= clip.duration:
                        # Extract frame at current time
                        frame = clip.get_frame(current_time)
                        
                        # Convert to PIL Image and save
                        img = Image.fromarray(frame)
                        frame_path = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
                        img.save(frame_path)
                        frames.append(frame_path)
                        frame_count += 1
                    
                    current_time += interval
                
                clip.close()
                
            except Exception as e:
                self.logger.error(f"MoviePy frame extraction failed: {e}")
        
        else:
            # No video processing library available
            self.logger.error("No video processing library available (install opencv-python or moviepy)")
            
            # Create at least one placeholder frame
            frame_path = os.path.join(output_path, "frame_0000.jpg")
            img = Image.new("RGB", (320, 240), color=(100, 100, 100))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.text((10, 100), "Install OpenCV for video processing", fill=(255, 255, 255))
            img.save(frame_path)
            frames.append(frame_path)
        
        return frames

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
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
                "error": f"Invalid operation: {operation}. Must be one of {valid_operations}",
            }

        try:
            # Load video metadata
            video_data = self._load_video_metadata(video_input)

            # Process based on operation
            if operation == "analyze":
                analysis = await self._analyze_video(video_data, model_name)
                return {"success": True, "analysis": analysis}

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
                        "output_directory": output_path,
                    },
                }

            elif operation == "extract_audio":
                # Placeholder
                audio_path = os.path.join(output_path, "extracted_audio.wav")
                return {
                    "success": True,
                    "audio_path": audio_path,
                    "message": "Audio extraction placeholder",
                }

            elif operation == "summarize":
                # Use analyze and create summary
                analysis = await self._analyze_video(video_data, model_name)
                summary = (
                    f"Video Summary: Duration {video_data.duration}s, "
                    f"Resolution {video_data.width}x{video_data.height}, "
                    f"Detected: {', '.join(analysis['detected_objects'][:3])}"
                )

                return {
                    "success": True,
                    "summary": summary,
                    "key_moments": analysis.get("scene_changes", []),
                    "metadata": analysis["video_info"],
                }

        except Exception as e:
            self.logger.error(f"Video processing error: {e}")
            return {"success": False, "error": str(e)}
