# Multimodal Content Processing Pipeline

**Pipeline**: `examples/multimodal_processing.yaml`  
**Category**: Multimedia & AI Processing  
**Complexity**: Advanced  
**Key Features**: Image analysis, Audio processing, Video analysis, AI image generation, Multimodal integration, Professional reporting

## Overview

The Multimodal Content Processing Pipeline demonstrates comprehensive multimedia processing capabilities by analyzing images, audio, and video content using AI-powered tools. It showcases advanced multimodal workflows including content analysis, object detection, audio transcription, video frame extraction, and AI-generated content variations with professional reporting.

## Key Features Demonstrated

### 1. Advanced Image Processing
```yaml
# Image description and analysis
- id: analyze_image
  tool: image-analysis
  action: execute
  parameters:
    image: "{{ parameters.input_image }}"
    analysis_type: "describe"
    detail_level: "high"
    output_format: "json"

# Object detection with confidence thresholds
- id: detect_objects
  tool: image-analysis
  action: execute
  parameters:
    analysis_type: "detect_objects"
    confidence_threshold: 0.7
    prompt_suffix: "List objects directly without conversational language."
```

### 2. AI Image Generation
```yaml
- id: generate_variations
  tool: image-generation
  action: execute
  parameters:
    prompt: "A colorful abstract geometric design with rectangles and frames"
    size: "1024x1024"
    style: "vivid"
    num_images: 3
    output_format: "file"
    output_path: "{{ output_path }}/generated_images"
```

### 3. Comprehensive Audio Processing
```yaml
# Audio transcription
- id: transcribe_audio
  tool: audio-processing
  action: execute
  parameters:
    audio: "{{ parameters.input_audio }}"
    operation: "transcribe"
    language: "en"

# Audio analysis with spectral features
- id: analyze_audio
  tool: audio-processing
  action: execute
  parameters:
    audio: "{{ parameters.input_audio }}"
    operation: "analyze"
```

### 4. Video Processing and Frame Extraction
```yaml
# Video content analysis
- id: analyze_video
  tool: video-processing
  action: execute
  parameters:
    video: "{{ parameters.input_video }}"
    operation: "analyze"

# Key frame extraction
- id: extract_key_frames
  tool: video-processing
  action: execute
  parameters:
    video: "{{ parameters.input_video }}"
    operation: "extract_frames"
    frame_interval: 0.5
    output_path: "{{ output_path }}/video_frames"
```

### 5. Professional Multimedia Reporting
```yaml
content: |
  # Multimodal Analysis Results
  
  ## 📸 Image Analysis
  ### Original Image
  ![Original Image](test_image.jpg)
  
  ### Description
  {{ analyze_image.analysis.result }}
  
  ## 🎵 Audio Analysis
  ### Transcription
  > "{{ transcribe_audio.transcription }}"
  
  ## 🎬 Video Analysis
  ### Extracted Key Frames
  ![Frame 0](video_frames/frame_0000.jpg)
```

## Pipeline Architecture

### Input Parameters
- **input_image** (optional): Path to image file for analysis (default: "samples/test_image.jpg")
- **input_audio** (optional): Path to audio file for processing (default: "samples/test_speech.wav")
- **input_video** (optional): Path to video file for analysis (default: "samples/test_video_real.mp4")
- **output_dir** (optional): Output directory for processed files (default: auto-generated)

### Processing Flow

1. **Image Analysis Section**:
   - **Image Description** - Detailed AI-powered image analysis
   - **Object Detection** - Identifies and categorizes objects in the image
   - **Variation Generation** - Creates artistic variations using AI image generation

2. **Audio Processing Section**:
   - **Audio Transcription** - Converts speech to text
   - **Audio Analysis** - Extracts audio characteristics and spectral features

3. **Video Processing Section**:
   - **Video Analysis** - Comprehensive video content analysis
   - **Frame Extraction** - Extracts key frames at specified intervals
   - **Frame Analysis** - Analyzes extracted frames for content

4. **Integration and Reporting**:
   - **File Organization** - Copies and organizes original files
   - **Report Generation** - Creates comprehensive multimodal analysis report

### Supported Media Types

#### Image Formats
- JPEG, PNG, GIF, BMP, TIFF
- High-resolution images supported
- Detailed description and object detection
- AI-powered artistic variation generation

#### Audio Formats
- WAV, MP3, FLAC, AAC, OGG
- Speech transcription with language detection
- Spectral analysis and audio characteristics
- Tempo, volume, and frequency analysis

#### Video Formats
- MP4, AVI, MOV, WMV, MKV
- Frame-by-frame analysis capability
- Scene detection and object tracking
- Key frame extraction with temporal sampling

## Usage Examples

### Basic Multimodal Analysis
```bash
python scripts/run_pipeline.py examples/multimodal_processing.yaml \
  -i input_image="my_image.jpg" \
  -i input_audio="my_audio.wav" \
  -i input_video="my_video.mp4"
```

### Image-Only Processing
```bash
python scripts/run_pipeline.py examples/multimodal_processing.yaml \
  -i input_image="product_photo.jpg" \
  -i input_audio="" \
  -i input_video=""
```

### Audio Transcription Focus
```bash
python scripts/run_pipeline.py examples/multimodal_processing.yaml \
  -i input_audio="interview_recording.wav" \
  -i output_dir="audio_analysis"
```

### Video Analysis Pipeline
```bash
python scripts/run_pipeline.py examples/multimodal_processing.yaml \
  -i input_video="presentation_video.mp4" \
  -i output_dir="video_analysis"
```

## Sample Output Structure

### Generated Analysis Report

#### Image Analysis Section
- **Original Image Display**: Embedded image with proper formatting
- **Detailed Description**: AI-generated comprehensive image description
- **Object Detection Results**: Categorized list of detected objects with confidence
- **Generated Variations**: AI-created artistic variations with image previews

#### Audio Analysis Section
```markdown
### File Information
- **File**: `samples/test_speech.wav`
- **Format**: wav
- **Duration**: 6.24 seconds
- **Sample Rate**: 44100 Hz
- **Channels**: 1

### Transcription
> "Hello, this is a test of the audio transcription system."

### Audio Characteristics
- **Volume Level**: normal
- **Noise Level**: high
- **Tempo**: 84.72 BPM
- **Peak Amplitude**: 0.8288
- **RMS Energy**: 0.1311

### Spectral Analysis
- **Spectral Centroid**: 2636.58 Hz
- **Spectral Rolloff**: 4423.63 Hz
- **Spectral Bandwidth**: 2164.15 Hz
```

#### Video Analysis Section
```markdown
### Video Information
- **Duration**: 3.0 seconds
- **Resolution**: 1920x1080
- **Frame Rate**: 30 FPS
- **Total Frames**: 90

### Content Analysis
[AI-generated video content summary]

### Scene Detection
- **Total Scene Changes**: 2
- **Scene Change Timestamps**: 1.2, 2.5 seconds
- **Detected Objects**: person, chair, table
- **Dominant Colors**: blue, white, gray

### Extracted Key Frames
![Frame 0](video_frames/frame_0000.jpg)
![Frame 1](video_frames/frame_0001.jpg)
```

### File Structure Output
```
examples/outputs/multimodal_processing/
├── analysis_report.md          # Comprehensive analysis report
├── test_image.jpg              # Original image copy
├── generated_images/           # AI-generated image variations
│   ├── generated_1755805514_0.png
│   ├── generated_1755805515_1.png
│   └── generated_1755805516_2.png
└── video_frames/              # Extracted video frames
    ├── frame_0000.jpg
    ├── frame_0001.jpg
    ├── frame_0002.jpg
    ├── frame_0003.jpg
    ├── frame_0004.jpg
    └── frame_0005.jpg
```

Check actual generated report: [analysis_report.md](../../examples/outputs/multimodal_processing/analysis_report.md)

## Technical Implementation

### Multi-Tool Integration
```yaml
# Image processing tools
tool: image-analysis
tool: image-generation

# Audio processing tools
tool: audio-processing

# Video processing tools  
tool: video-processing

# File management tools
tool: filesystem
```

### Conditional Processing
```yaml
condition: "{{ extract_key_frames.frames | length > 0 }}"
# Only process frames if extraction was successful
```

### Advanced Template Processing
```yaml
# Clean up conversational text from AI responses
{% set frame_text = analyze_key_frames.analysis.result %}
{% set frame_text = frame_text | regex_replace('This image shows ', '') %}
{% set frame_text = frame_text | regex_replace('The image shows ', '') %}
{{ frame_text }}
```

### Professional Report Formatting
```yaml
content: |
  # Multimodal Analysis Results
  
  ## 📸 Image Analysis
  ### Generated Variations
  {% for image in generate_variations.images %}
  ![Variation {{ loop.index }}](generated_images/{{ image.path | basename }})
  {% endfor %}
```

### Spectral Audio Analysis
The pipeline extracts comprehensive audio features:
- **Temporal Features**: RMS energy, zero crossing rate
- **Spectral Features**: Spectral centroid, rolloff, bandwidth
- **Perceptual Features**: Volume level, noise level, tempo
- **Amplitude Analysis**: Peak amplitude, dynamic range

## Advanced Features

### AI Image Generation Integration
```yaml
parameters:
  prompt: "A colorful abstract geometric design with rectangles and frames on a gradient background, modern digital art style"
  size: "1024x1024"
  style: "vivid"
  num_images: 3
```

### Video Frame Temporal Sampling
```yaml
frame_interval: 0.5  # Extract frame every 0.5 seconds
# Enables temporal analysis across video duration
```

### Object Detection with Confidence Filtering
```yaml
confidence_threshold: 0.7
# Only report objects detected with >70% confidence
```

### Intelligent Text Processing
```yaml
{% for line in lines %}
{% if line and not 'I can identify' in line and not 'In this image' in line %}
{{ line }}
{% endif %}
{% endfor %}
# Filters out conversational AI artifacts for clean output
```

### Dynamic Content Adaptation
```yaml
{% if generate_variations.success and generate_variations.images %}
Created {{ generate_variations.images | length }} artistic variations
{% else %}
*Image generation was not successful or no images were generated.*
{% endif %}
```

## Common Use Cases

- **Content Creation**: Generate variations and analyze multimedia content
- **Media Analysis**: Comprehensive analysis of user-generated content
- **Accessibility**: Audio transcription and image description for accessibility
- **Quality Assurance**: Automated content quality assessment
- **Educational Content**: Analyze and describe educational materials
- **Marketing Analytics**: Analyze marketing multimedia content effectiveness
- **Research Applications**: Academic research with multimodal data analysis
- **Content Moderation**: Automated content analysis for moderation systems

## Best Practices Demonstrated

1. **Multimodal Integration**: Seamless processing of different media types
2. **Professional Reporting**: Comprehensive, publication-ready analysis reports
3. **Error Handling**: Conditional processing based on successful operations
4. **File Organization**: Structured output directory management
5. **Template Optimization**: Clean, readable report generation
6. **Quality Control**: Confidence thresholds and result validation
7. **Scalable Architecture**: Modular processing pipeline design

## Troubleshooting

### Common Issues
- **File Format Support**: Verify media files are in supported formats
- **File Path Access**: Ensure input files are accessible and readable
- **Model Availability**: Confirm required AI models are available
- **Output Permissions**: Verify write permissions for output directories

### Performance Considerations
- **Large Files**: Video processing may require significant time and memory
- **AI Model Latency**: Image generation and analysis can be time-intensive
- **Concurrent Processing**: Consider parallel processing for multiple files
- **Storage Requirements**: Generated files may require substantial disk space

### Quality Optimization
- **Confidence Thresholds**: Adjust object detection confidence as needed
- **Frame Sampling**: Optimize frame extraction intervals for video length
- **Analysis Detail**: Balance detail level with processing time requirements
- **Output Format**: Choose appropriate output formats for downstream use

## Related Examples
- [creative_image_pipeline.md](creative_image_pipeline.md) - Advanced image generation workflows
- [data_processing.md](data_processing.md) - Data processing and analysis patterns
- [modular_analysis_pipeline.md](modular_analysis_pipeline.md) - Modular analysis architectures

## Technical Requirements

- **Tools**: image-analysis, image-generation, audio-processing, video-processing, filesystem
- **Models**: Support for multimodal AI analysis and generation
- **Storage**: Adequate space for original files, generated content, and reports
- **Memory**: Sufficient RAM for video processing and image generation
- **Network**: Internet access for AI model API calls

This pipeline provides comprehensive multimodal processing capabilities essential for modern content analysis workflows, educational applications, and multimedia research projects requiring professional-grade analysis and reporting.