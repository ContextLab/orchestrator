# Creative Image Generation Pipeline

**Pipeline**: `examples/creative_image_pipeline.yaml`  
**Category**: Creative & Multimodal  
**Complexity**: Advanced  
**Key Features**: AI image generation, Style variations, Image analysis, Template rendering

## Overview

The Creative Image Generation Pipeline demonstrates advanced AI-powered image creation workflows. It generates multiple style variations of images based on creative prompts, analyzes the generated content, and creates organized galleries with comprehensive documentation.

## Key Features Demonstrated

### 1. Dynamic Folder Naming
```yaml
- id: generate_folder_name
  action: generate
  parameters:
    prompt: |
      Convert this image prompt into a short, descriptive folder name (no spaces, use underscores):
      "{{ base_prompt }}"
```

### 2. Image Generation with Parameters
```yaml
- id: generate_base_image
  action: execute
  tool: image-generation
  parameters:
    prompt: "{{ base_prompt }}"
    size: "1024x1024"
    style: "high quality, detailed"
    output_format: "file"
    output_path: "{{ output_path }}/{{ generate_folder_name }}"
    filename: "base.png"
```

### 3. Style Variations with Array Iteration
```yaml
parameters:
  prompt: "{{ base_prompt }}, {{ art_styles[0] }} style"  # Array indexing
  style: "{{ art_styles[0] }}"
  filename: "{{ art_styles[0] }}.png"
```

### 4. Image Analysis Integration
```yaml
- id: analyze_base
  action: execute
  tool: image-analysis
  parameters:
    image: "{{ generate_base_image.images[0].path }}"
    analysis_type: "describe"
    detail_level: "high"
    output_format: "json"
```

### 5. Conditional Processing
```yaml
condition: "{{ generate_enhanced.images | length > 0 }}"
# Only analyze if images were successfully generated
```

## Pipeline Structure

### Input Parameters
- **base_prompt** (string): Creative prompt for image generation (default: "A serene landscape with mountains and a lake")
- **num_variations** (integer): Number of variations to generate (default: 3)
- **art_styles** (array): Artistic styles to apply (default: ["photorealistic", "impressionist", "cyberpunk"])
- **output_path** (string): Base output directory (default: "examples/outputs/creative_image_pipeline")

### Processing Flow

1. **Generate Folder Name** - Creates descriptive folder name from prompt
2. **Generate Base Image** - Creates initial image from prompt
3. **Analyze Base Image** - AI analyzes generated image for quality/content
4. **Generate Style Variations** (parallel):
   - Photorealistic version
   - Impressionist version  
   - Cyberpunk version
5. **Enhance Prompt** - AI optimizes prompt for better results
6. **Generate Enhanced Image** - Creates image with optimized prompt
7. **Analyze Enhanced Image** - Classifies and analyzes final result
8. **Count Generated Images** - Tallies total outputs
9. **Create Gallery Report** - Generates markdown documentation
10. **Save Gallery Report** - Writes README.md with image gallery

### Output Structure
```
examples/outputs/creative_image_pipeline/[prompt_folder]/
├── base.png                    # Original prompt image
├── photorealistic.png         # Photorealistic style
├── impressionist.png          # Impressionist style
├── cyberpunk.png              # Cyberpunk style
├── enhanced.png               # AI-enhanced version
└── README.md                  # Gallery documentation
```

## Usage Examples

### Basic Usage
```bash
python scripts/run_pipeline.py examples/creative_image_pipeline.yaml
```

### Custom Creative Prompt
```bash
python scripts/run_pipeline.py examples/creative_image_pipeline.yaml \
  -i base_prompt="a dragon made of crystal soaring through aurora borealis"
```

### Different Art Styles
```bash
python scripts/run_pipeline.py examples/creative_image_pipeline.yaml \
  -i base_prompt="a futuristic cityscape at sunset" \
  -i art_styles='["art_nouveau", "minimalist", "baroque"]'
```

### Custom Output Location
```bash
python scripts/run_pipeline.py examples/creative_image_pipeline.yaml \
  -i base_prompt="magical forest with glowing mushrooms" \
  -i output_path="my_creative_gallery"
```

## Sample Outputs

### Example Gallery: Astronaut Kitten
**Generated from**: "an astronaut kitten playing the guitar while floating in space"

**Output Directory**: [astronaut_kitten_guitar_space/](../../examples/outputs/creative_image_pipeline/astronaut_kitten_guitar_space/)

**Generated Files**:
- `base.png` - Original prompt rendering
- `photorealistic.png` - Photorealistic style variation
- `impressionist.png` - Impressionist style variation  
- `cyberpunk.png` - Cyberpunk style variation
- `enhanced.png` - AI-enhanced prompt rendering
- `README.md` - Complete gallery documentation

### Image Analysis Example
From the astronaut kitten example:
> "This is a whimsical and highly detailed digital artwork depicting an adorable kitten astronaut floating in space. The central figure is a small tabby kitten with distinctive brown and cream striped markings, bright blue eyes, and a pink tongue playfully sticking out..."

### Gallery Report Structure
- **Original Prompt**: Shows input prompt
- **Output Folder**: Directory location
- **Base Image Analysis**: Detailed AI description
- **Style Variations**: List of generated styles
- **Enhanced Prompt**: Shows prompt optimization
- **Generated Images**: Table with image previews and descriptions
- **Total Count**: Number of images created

## Advanced Features

### Array Parameter Processing
```yaml
# Access array elements by index
art_styles[0]  # "photorealistic"
art_styles[1]  # "impressionist" 
art_styles[2]  # "cyberpunk"

# Template loops over arrays
{% for style in art_styles %}
- {{ style }} style
{% endfor %}
```

### Image Path References
```yaml
# Reference generated image paths
image: "{{ generate_base_image.images[0].path }}"
# Access file system paths from image generation results
```

### Prompt Engineering Integration
```yaml
- id: enhance_prompt
  tool: prompt-optimization
  parameters:
    optimization_goal: "artistic_quality"
    include_style_suggestions: true
```

### Markdown Table Generation
```yaml
| Image | Description |
|-------|-------------|
| ![Base](base.png) | Original prompt rendering |
{% for style in art_styles %}
| ![{{ style|capitalize }}]({{ style }}.png) | {{ style|capitalize }} style variation |
{% endfor %}
```

## Best Practices Demonstrated

1. **Organized Output**: Creates structured directories for each generation session
2. **Comprehensive Documentation**: Auto-generates gallery documentation
3. **Style Consistency**: Maintains consistent file naming and organization  
4. **Error Handling**: Conditional processing prevents failures from breaking workflow
5. **Resource Management**: Proper dependency ordering prevents resource conflicts
6. **Template Flexibility**: Uses advanced template features for dynamic content

## Common Use Cases

- **Art Generation**: Create artistic variations of concepts
- **Design Exploration**: Generate multiple design approaches
- **Style Transfer**: Apply different artistic styles to concepts
- **Creative Workflows**: Automate creative image production pipelines
- **Portfolio Generation**: Create organized collections of generated art
- **Prompt Engineering**: Test and optimize image generation prompts

## Troubleshooting

### Common Issues

#### Image Generation Failures
```
Error: Failed to generate image
```
**Solutions**:
- Check image generation service availability
- Verify prompt doesn't contain prohibited content
- Ensure output directory is writable

#### Style Application Issues
```
Error: Style not recognized
```
**Solutions**:
- Use standard art style names
- Verify art_styles array format
- Check style compatibility with generation model

#### Large File Handling
- Generated images can be large (1024x1024 PNG files)
- Ensure sufficient disk space
- Monitor output directory size

### Performance Optimization

1. **Parallel Style Generation**: Multiple styles generated simultaneously
2. **Efficient File Organization**: Organized folder structure
3. **Image Format Optimization**: PNG format balances quality and size
4. **Dependency Management**: Prevents unnecessary regeneration

## Extension Examples

### Video Generation Pipeline
```yaml
- id: generate_animation
  tool: video-generation
  parameters:
    frames: ["{{ base_image }}", "{{ style_variation_1 }}", "{{ style_variation_2 }}"]
    fps: 2
    output_path: "{{ output_folder }}/animation.mp4"
```

### Batch Prompt Processing
```yaml
parameters:
  prompt_list:
    type: array
    default: ["prompt1", "prompt2", "prompt3"]
    
steps:
  - id: process_all_prompts
    for_each: "{{ prompt_list }}"
    # Generate galleries for multiple prompts
```

### Quality Assessment
```yaml
- id: assess_quality
  tool: image-analysis
  parameters:
    image: "{{ generated_image.path }}"
    analysis_type: "quality_score"
    criteria: ["composition", "color_harmony", "detail_level"]
```

## Related Examples
- [multimodal_processing.md](multimodal_processing.md) - Multi-format media processing
- [control_flow_for_loop.md](control_flow_for_loop.md) - Batch processing patterns
- [auto_tags_demo.md](auto_tags_demo.md) - Dynamic parameter resolution

## Technical Requirements

- **Tools**: image-generation, image-analysis, prompt-optimization, filesystem
- **Models**: Depends on image generation service (DALL-E, Midjourney, etc.)
- **Storage**: ~50-100MB per prompt session (5+ high-res images)
- **Processing**: GPU acceleration recommended for image generation
- **Network**: Stable connection for AI service APIs

This pipeline demonstrates enterprise-grade creative workflows with professional output organization, making it suitable for design teams, content creators, and automated creative production systems.