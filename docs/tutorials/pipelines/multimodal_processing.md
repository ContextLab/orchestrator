# Pipeline Tutorial: multimodal_processing

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 50/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates conditional_execution, data_flow, image_generation and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- Visual content creation

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Conditional logic and branching
- Data flow between pipeline steps
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 11 template patterns for dynamic content
- **feature_highlights**: Demonstrates 6 key orchestrator features

### Data Flow
This pipeline processes input parameters through 6 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Multimodal Processing Pipeline
# Demonstrates image, audio, and video processing capabilities
id: multimodal_processing
name: Multimodal Content Processing Pipeline
description: Process various media types with AI-powered analysis
version: "1.0.0"

parameters:
  input_image:
    type: string
    default: "samples/test_image.jpg"
  input_audio:
    type: string
    default: "samples/test_speech.wav"
  input_video:
    type: string
    default: "samples/test_video_real.mp4"
  output_dir:
    type: string
    default: ""

steps:
  # Image Processing Section
  - id: analyze_image
    tool: image-analysis
    action: execute
    parameters:
      image: "{{ parameters.input_image }}"
      analysis_type: "describe"
      detail_level: "high"
      output_format: "json"
    
  - id: detect_objects
    tool: image-analysis
    action: execute
    parameters:
      image: "{{ parameters.input_image }}"
      analysis_type: "detect_objects"
      confidence_threshold: 0.7
      prompt_suffix: "List objects directly without conversational language. Use bullet points."
    dependencies:
      - analyze_image
    
  - id: generate_variations
    tool: image-generation
    action: execute
    parameters:
      prompt: "A colorful abstract geometric design with rectangles and frames on a gradient background, modern digital art style"
      size: "1024x1024"
      style: "vivid"
      num_images: 3
      output_format: "file"
      output_path: "{{ output_path }}/generated_images"
    dependencies:
      - analyze_image
    
  # Audio Processing Section
  - id: transcribe_audio
    tool: audio-processing
    action: execute
    parameters:
      audio: "{{ parameters.input_audio }}"
      operation: "transcribe"
      language: "en"
    
  - id: analyze_audio
    tool: audio-processing
    action: execute
    parameters:
      audio: "{{ parameters.input_audio }}"
      operation: "analyze"
    dependencies:
      - transcribe_audio
    
  # Video Processing Section
  - id: analyze_video
    tool: video-processing
    action: execute
    parameters:
      video: "{{ parameters.input_video }}"
      operation: "analyze"
    
  - id: extract_key_frames
    tool: video-processing
    action: execute
    parameters:
      video: "{{ parameters.input_video }}"
      operation: "extract_frames"
      frame_interval: 0.5
      output_path: "{{ output_path }}/video_frames"
    dependencies:
      - analyze_video
    
  - id: analyze_key_frames
    tool: image-analysis
    action: execute
    parameters:
      image: "{{ extract_key_frames.frames[0] }}"
      analysis_type: "describe"
      detail_level: "medium"
    dependencies:
      - extract_key_frames
    condition: "{{ extract_key_frames.frames | length > 0 }}"
    
  # Copy original image for report
  - id: copy_original_image
    tool: filesystem
    action: copy
    parameters:
      path: "{{ parameters.input_image }}"
      destination: "{{ output_path }}/test_image.jpg"
    dependencies:
      - analyze_key_frames
    
  # Combined Analysis
  - id: generate_summary_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/analysis_report.md"
      content: |
        # Multimodal Analysis Results
        
        ## 📸 Image Analysis
        
        ### Original Image
        ![Original Image](test_image.jpg)
        
        ### Description
        {{ analyze_image.analysis.result }}
        
        ### Detected Objects
        {% set objects_text = detect_objects.analysis.result %}
        {% set lines = objects_text.split('\n') %}
        {% for line in lines %}
        {% if line and not 'I can identify' in line and not 'In this image' in line and not 'appears to be' in line %}
        {{ line }}
        {% endif %}
        {% endfor %}
        
        ### Generated Variations
        {% if generate_variations.success and generate_variations.images %}
        Created {{ generate_variations.images | length }} artistic variations using DALL-E 3:
        
        {% for image in generate_variations.images %}
        ![Variation {{ loop.index }}](generated_images/{{ image.path | basename }})
        {% endfor %}
        {% else %}
        *Image generation was not successful or no images were generated.*
        {% endif %}
        
        ## 🎵 Audio Analysis
        
        ### File Information
        - **File**: `{{ parameters.input_audio }}`
        - **Format**: {{ analyze_audio.analysis.format }}
        - **Duration**: {{ analyze_audio.analysis.duration }} seconds
        - **Sample Rate**: {{ analyze_audio.analysis.sample_rate }} Hz
        - **Channels**: {{ analyze_audio.analysis.channels }}
        
        ### Transcription
        > "{{ transcribe_audio.transcription }}"
        
        ### Audio Characteristics
        - **Volume Level**: {{ analyze_audio.analysis.analysis.volume_level }}
        - **Noise Level**: {{ analyze_audio.analysis.analysis.noise_level }}
        - **Tempo**: {{ analyze_audio.analysis.analysis.tempo_bpm }} BPM
        - **Peak Amplitude**: {{ analyze_audio.analysis.analysis.peak_amplitude | round(4) }}
        - **RMS Energy**: {{ analyze_audio.analysis.analysis.rms_energy | round(4) }}
        
        ### Spectral Analysis
        - **Spectral Centroid**: {{ analyze_audio.analysis.analysis.spectral_centroid_hz | round(2) }} Hz
        - **Spectral Rolloff**: {{ analyze_audio.analysis.analysis.spectral_rolloff_hz | round(2) }} Hz  
        - **Spectral Bandwidth**: {{ analyze_audio.analysis.analysis.spectral_bandwidth_hz | round(2) }} Hz
        - **Zero Crossing Rate**: {{ analyze_audio.analysis.analysis.zero_crossing_rate | round(6) }}
        
        ## 🎬 Video Analysis
        
        ### Video Information
        - **File**: `{{ parameters.input_video }}`
        - **Duration**: {{ analyze_video.analysis.video_info.duration }} seconds
        - **Resolution**: {{ analyze_video.analysis.video_info.resolution }}
        - **Frame Rate**: {{ analyze_video.analysis.video_info.fps }} FPS
        - **Total Frames**: {{ (analyze_video.analysis.video_info.duration * analyze_video.analysis.video_info.fps) | int }}
        
        ### Content Analysis
        {{ analyze_video.analysis.summary }}
        
        ### Scene Detection
        - **Total Scene Changes**: {{ analyze_video.analysis.scene_changes | length }}
        - **Scene Change Timestamps**: {{ analyze_video.analysis.scene_changes | join(', ') }} seconds
        - **Detected Objects**: {{ analyze_video.analysis.detected_objects | join(', ') }}
        - **Dominant Colors**: {{ analyze_video.analysis.dominant_colors | join(', ') }}
        
        ### Extracted Key Frames
        
        #### Frame at 0.0s
        ![Frame 0](video_frames/frame_0000.jpg)
        
        #### Frame at 0.5s  
        ![Frame 1](video_frames/frame_0001.jpg)
        
        #### Frame at 1.0s
        ![Frame 2](video_frames/frame_0002.jpg)
        
        #### Frame at 1.5s
        ![Frame 3](video_frames/frame_0003.jpg)
        
        #### Frame at 2.0s
        ![Frame 4](video_frames/frame_0004.jpg)
        
        #### Frame at 2.5s
        ![Frame 5](video_frames/frame_0005.jpg)
        
        ### Frame Analysis
        {% set frame_text = analyze_key_frames.analysis.result %}
        {% set frame_text = frame_text | regex_replace('This image shows ', '') %}
        {% set frame_text = frame_text | regex_replace('The image shows ', '') %}
        {% set frame_text = frame_text | regex_replace('The overall composition is ', 'Overall composition: ') %}
        {{ frame_text }}
        
        ## 📊 Processing Summary
        - **Total Media Files Processed**: 3 (1 image, 1 audio, 1 video)
        - **Generated Images**: {{ generate_variations.images | length }}
        - **Extracted Video Frames**: {{ extract_key_frames.frames | length }}
        - **Processing Time**: Completed successfully
        
        ---
        *Report generated on {{ timestamp }}*
    dependencies:
      - generate_variations
      - analyze_audio
      - analyze_key_frames
      - copy_original_image
    

outputs:
  image_analysis: "{{ analyze_image.analysis }}"
  audio_transcription: "{{ transcribe_audio.transcription }}"
  video_summary: "{{ analyze_video.analysis.summary }}"
  generated_images: "{{ generate_variations.images }}"
  report_location: "{{ parameters.output_dir }}/analysis_report.md"
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust step parameters to customize behavior for your needs

### Step Modifications
- Add new steps by following the same pattern as existing ones
- Remove steps that aren't needed for your specific use case
- Reorder steps if your workflow requires different sequencing
- Replace tool actions with alternatives that provide similar functionality

### Output Customization
- Change output file paths and formats to match your requirements
- Modify output templates to customize the structure and content
- This pipeline produces Analysis results, Images, JSON data, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- Most basic pipelines can be combined with this pattern

### Extension Ideas
- Add iterative processing for continuous improvement
- Implement parallel processing for better performance
- Include advanced error recovery mechanisms

### Combination Examples
- Can be combined with most other pipeline patterns

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 2. Run: python scripts/run_pipeline.py examples/multimodal_processing.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated Images in the specified output directory
- Generated JSON data in the specified output directory
- Generated Markdown documents in the specified output directory
- Generated Reports in the specified output directory
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **Template Resolution Errors**: Check that all input parameters are provided and template syntax is correct
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy

---

*Tutorial generated on 2025-08-27T23:40:24.396469*
