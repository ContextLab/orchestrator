"""
Enhanced visual quality assessment for pipeline outputs.

This module provides specialized visual content analysis capabilities
using vision-enabled LLM models to assess images, charts, and visual
content for production quality standards.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.quality_assessment import (
    IssueCategory, IssueSeverity, QualityIssue, VisualQuality
)

logger = logging.getLogger(__name__)


class VisualContentAnalyzer:
    """Analyzes visual content for production quality standards."""
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'}
    
    # Expected chart/visualization types
    CHART_TYPES = {
        'bar_chart', 'line_chart', 'pie_chart', 'scatter_chart', 'histogram_chart',
        'heatmap', 'boxplot', 'violin_plot', 'area_chart', 'bubble_chart'
    }
    
    def __init__(self):
        """Initialize visual content analyzer."""
        pass
    
    def analyze_image_quality(self, image_path: str) -> List[QualityIssue]:
        """Analyze image for basic quality issues without vision model."""
        issues = []
        
        try:
            path = Path(image_path)
            
            # Check if file exists
            if not path.exists():
                issues.append(QualityIssue(
                    category=IssueCategory.VISUAL_QUALITY,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Image file does not exist: {image_path}",
                    file_path=image_path,
                    suggestion="Ensure image generation completed successfully"
                ))
                return issues
            
            # Check file size (empty files)
            if path.stat().st_size == 0:
                issues.append(QualityIssue(
                    category=IssueCategory.VISUAL_QUALITY,
                    severity=IssueSeverity.CRITICAL,
                    description="Image file is empty (0 bytes)",
                    file_path=image_path,
                    suggestion="Regenerate image as file appears corrupted"
                ))
            
            # Check file extension
            if path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
                issues.append(QualityIssue(
                    category=IssueCategory.VISUAL_QUALITY,
                    severity=IssueSeverity.MINOR,
                    description=f"Unsupported image format: {path.suffix}",
                    file_path=image_path,
                    suggestion="Consider using standard formats like PNG or JPEG"
                ))
            
            # Check file size (very small files might be corrupt)
            if path.stat().st_size < 1024:  # Less than 1KB
                issues.append(QualityIssue(
                    category=IssueCategory.VISUAL_QUALITY,
                    severity=IssueSeverity.MAJOR,
                    description="Image file is very small - may be corrupted or incomplete",
                    file_path=image_path,
                    suggestion="Verify image generation completed successfully"
                ))
            
        except Exception as e:
            issues.append(QualityIssue(
                category=IssueCategory.VISUAL_QUALITY,
                severity=IssueSeverity.MAJOR,
                description=f"Error analyzing image file: {str(e)}",
                file_path=image_path,
                suggestion="Check file permissions and integrity"
            ))
        
        return issues
    
    def analyze_chart_quality(self, image_path: str) -> List[QualityIssue]:
        """Analyze chart-specific quality without vision model."""
        issues = []
        
        # Check naming patterns for chart types
        filename = Path(image_path).stem.lower()
        
        # Detect chart type from filename
        detected_chart_type = None
        for chart_type in self.CHART_TYPES:
            if chart_type in filename or chart_type.replace('_', '') in filename:
                detected_chart_type = chart_type
                break
        
        if detected_chart_type:
            # Chart-specific quality checks
            issues.extend(self._assess_chart_naming_quality(image_path, detected_chart_type))
        else:
            # Generic visual content - check for descriptive naming
            if any(generic in filename for generic in ['image', 'picture', 'photo', 'output']):
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MINOR,
                    description="Generic image filename - not descriptive of content",
                    file_path=image_path,
                    suggestion="Use descriptive filenames that indicate the image content or purpose"
                ))
        
        return issues
    
    def _assess_chart_naming_quality(self, image_path: str, chart_type: str) -> List[QualityIssue]:
        """Assess chart naming conventions."""
        issues = []
        
        filename = Path(image_path).stem
        
        # Check for timestamp or unique identifiers in chart names
        if any(pattern in filename for pattern in ['chart', 'graph', 'plot']) and not any(
            pattern in filename for pattern in ['202', '_', '-']
        ):
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Chart filename lacks timestamp or unique identifier",
                file_path=image_path,
                suggestion="Include timestamps or unique identifiers in chart filenames"
            ))
        
        return issues
    
    def assess_visual_directory_structure(self, directory_path: str) -> List[QualityIssue]:
        """Assess the organization of visual content within a directory."""
        issues = []
        
        try:
            path = Path(directory_path)
            if not path.exists():
                return issues
            
            # Find all image files
            image_files = []
            for ext in self.SUPPORTED_IMAGE_FORMATS:
                image_files.extend(path.glob(f"*{ext}"))
                image_files.extend(path.glob(f"**/*{ext}"))
            
            if not image_files:
                return issues
            
            # Check for image organization patterns
            issues.extend(self._assess_image_organization(path, image_files))
            
            # Check for missing README or documentation
            readme_files = list(path.glob("README*")) + list(path.glob("readme*"))
            if not readme_files and len(image_files) > 3:
                issues.append(QualityIssue(
                    category=IssueCategory.FILE_ORGANIZATION,
                    severity=IssueSeverity.MINOR,
                    description="Directory with multiple images lacks README documentation",
                    file_path=str(path),
                    suggestion="Add README file explaining the visual content and organization"
                ))
            
        except Exception as e:
            logger.error(f"Error assessing visual directory structure for {directory_path}: {e}")
        
        return issues
    
    def _assess_image_organization(self, base_path: Path, image_files: List[Path]) -> List[QualityIssue]:
        """Assess image organization within directory."""
        issues = []
        
        # Check for images in subdirectories vs root directory
        root_images = [f for f in image_files if f.parent == base_path]
        subdir_images = [f for f in image_files if f.parent != base_path]
        
        # If many images, suggest subdirectory organization
        if len(root_images) > 10:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description=f"Many images ({len(root_images)}) in root directory - consider organizing in subdirectories",
                file_path=str(base_path),
                suggestion="Organize images into logical subdirectories (e.g., by type, date, or content)"
            ))
        
        # Check for consistent naming patterns
        if len(image_files) > 1:
            issues.extend(self._check_naming_consistency(image_files))
        
        return issues
    
    def _check_naming_consistency(self, image_files: List[Path]) -> List[QualityIssue]:
        """Check for consistent naming patterns across images."""
        issues = []
        
        # Extract naming patterns
        stems = [f.stem for f in image_files]
        
        # Check for very different naming styles
        has_underscores = any('_' in stem for stem in stems)
        has_hyphens = any('-' in stem for stem in stems)
        has_spaces = any(' ' in stem for stem in stems)
        
        inconsistent_separators = sum([has_underscores, has_hyphens, has_spaces])
        
        if inconsistent_separators > 1:
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description="Inconsistent filename separators across images (mix of spaces, hyphens, underscores)",
                file_path="multiple_files",
                suggestion="Use consistent filename separator convention (recommend underscores or hyphens)"
            ))
        
        # Check for generic names
        generic_names = ['image', 'picture', 'photo', 'output', 'result', 'generated']
        generic_count = sum(1 for stem in stems if any(generic in stem.lower() for generic in generic_names))
        
        if generic_count > len(stems) * 0.3:  # More than 30% generic names
            issues.append(QualityIssue(
                category=IssueCategory.FILE_ORGANIZATION,
                severity=IssueSeverity.MINOR,
                description="Many files have generic names that don't describe content",
                file_path="multiple_files",
                suggestion="Use descriptive filenames that indicate image content or purpose"
            ))
        
        return issues


class EnhancedVisualAssessor:
    """Enhanced visual quality assessor with LLM vision integration."""
    
    def __init__(self):
        """Initialize enhanced visual assessor."""
        self.analyzer = VisualContentAnalyzer()
    
    def create_enhanced_visual_assessment_prompt(self, image_path: str, context: Dict[str, Any] = None) -> str:
        """Create detailed visual assessment prompt with context."""
        
        base_prompt = f"""You are reviewing this image for production quality in a pipeline output system.

IMAGE FILE: {image_path}

Analyze this image comprehensively and provide a JSON response with:
{{
  "overall_rating": "EXCELLENT|GOOD|NEEDS_IMPROVEMENT|POOR",
  "image_quality_score": 0-100,
  "issues": [
    {{
      "category": "visual_quality|chart_quality|professional_standards",
      "severity": "critical|major|minor",
      "description": "specific visual issue description",
      "suggestion": "actionable improvement recommendation"
    }}
  ],
  "assessment_details": {{
    "image_renders_correctly": boolean,
    "resolution_appropriate": boolean,
    "colors_professional": boolean,
    "text_readable": boolean,
    "charts_properly_labeled": boolean,
    "legends_clear": boolean,
    "visual_hierarchy_clear": boolean,
    "branding_consistent": boolean
  }},
  "feedback": "detailed visual assessment explanation",
  "recommendations": ["list of specific improvements"]
}}

COMPREHENSIVE QUALITY ANALYSIS:

1. **Technical Quality:**
   - Image renders without corruption or artifacts
   - Resolution is appropriate for intended use
   - File format is suitable for content type
   - Colors are balanced and professional
   - No pixelation or compression artifacts

2. **Chart/Visualization Quality (if applicable):**
   - Data is clearly visualized and understandable
   - Axes are properly labeled with units
   - Legends are present and clear
   - Color choices are accessible and meaningful
   - Scale and proportions are appropriate
   - Text size is readable at normal viewing distance

3. **Professional Standards:**
   - Visual design follows professional standards
   - Color scheme is appropriate for business/academic use
   - Typography is consistent and readable
   - Layout is well-organized and clean
   - Visual hierarchy guides attention appropriately

4. **Content Appropriateness:**
   - Visual content matches expected output type
   - Styling is consistent with professional documentation
   - No inappropriate or distracting elements
   - Suitable for showcasing platform capabilities

5. **Accessibility Considerations:**
   - Color contrast is sufficient for readability
   - Text is large enough to read
   - Color is not the only way information is conveyed
   - Visual elements support understanding

CRITICAL ISSUES to flag:
- Image corruption or rendering failures
- Unreadable text or labels
- Missing essential chart elements (axes, legends)
- Inappropriate color schemes or unprofessional appearance
- Accessibility barriers

Respond ONLY with valid JSON."""

        # Add context-specific guidance if provided
        if context:
            pipeline_type = context.get('pipeline_type', 'unknown')
            if 'chart' in pipeline_type.lower() or 'analysis' in pipeline_type.lower():
                base_prompt += f"""

PIPELINE-SPECIFIC CONTEXT: This image is from a {pipeline_type} pipeline.
Expected characteristics:
- Professional data visualization standards
- Clear labeling and legends
- Appropriate chart type for data
- Business/academic presentation quality
"""
        
        return base_prompt
    
    def assess_visual_content_with_context(
        self, 
        image_path: str, 
        pipeline_context: Optional[Dict[str, Any]] = None
    ) -> VisualQuality:
        """Assess visual content with pipeline context."""
        
        # Start with rule-based analysis
        issues = self.analyzer.analyze_image_quality(image_path)
        issues.extend(self.analyzer.analyze_chart_quality(image_path))
        
        # Determine overall rating based on issues
        if any(issue.severity == IssueSeverity.CRITICAL for issue in issues):
            rating = IssueSeverity.CRITICAL
        elif any(issue.severity == IssueSeverity.MAJOR for issue in issues):
            rating = IssueSeverity.MAJOR
        elif any(issue.severity == IssueSeverity.MINOR for issue in issues):
            rating = IssueSeverity.MINOR
        else:
            rating = IssueSeverity.ACCEPTABLE
        
        # Create assessment
        visual_quality = VisualQuality(
            rating=rating,
            issues=issues,
            feedback="Rule-based visual quality assessment completed",
            image_renders_correctly=not any(
                "corrupt" in issue.description.lower() or "empty" in issue.description.lower()
                for issue in issues
            ),
            charts_have_labels=True,  # Default assumption, would be assessed by vision model
            professional_appearance=rating in [IssueSeverity.ACCEPTABLE, IssueSeverity.MINOR],
            appropriate_styling=rating != IssueSeverity.CRITICAL
        )
        
        return visual_quality
    
    def parse_vision_model_response(self, response: Dict[str, Any], image_path: str) -> VisualQuality:
        """Parse vision model response into VisualQuality object."""
        
        # Extract basic information
        overall_rating = response.get('overall_rating', 'NEEDS_IMPROVEMENT')
        feedback = response.get('feedback', '')
        assessment_details = response.get('assessment_details', {})
        
        # Convert rating to IssueSeverity
        rating_map = {
            'EXCELLENT': IssueSeverity.ACCEPTABLE,
            'GOOD': IssueSeverity.MINOR,
            'NEEDS_IMPROVEMENT': IssueSeverity.MAJOR,
            'POOR': IssueSeverity.CRITICAL
        }
        rating = rating_map.get(overall_rating, IssueSeverity.MAJOR)
        
        # Parse issues
        issues = []
        for issue_data in response.get('issues', []):
            try:
                severity_map = {
                    'critical': IssueSeverity.CRITICAL,
                    'major': IssueSeverity.MAJOR,
                    'minor': IssueSeverity.MINOR
                }
                
                severity = severity_map.get(issue_data.get('severity'), IssueSeverity.MAJOR)
                
                issues.append(QualityIssue(
                    category=IssueCategory.VISUAL_QUALITY,
                    severity=severity,
                    description=issue_data.get('description', ''),
                    file_path=image_path,
                    suggestion=issue_data.get('suggestion', '')
                ))
                
            except Exception as e:
                logger.warning(f"Failed to parse visual issue: {issue_data} - {e}")
        
        return VisualQuality(
            rating=rating,
            issues=issues,
            feedback=feedback,
            image_renders_correctly=assessment_details.get('image_renders_correctly', True),
            charts_have_labels=assessment_details.get('charts_properly_labeled', True),
            professional_appearance=assessment_details.get('colors_professional', True) and 
                                  assessment_details.get('visual_hierarchy_clear', True),
            appropriate_styling=assessment_details.get('branding_consistent', True) and
                              assessment_details.get('resolution_appropriate', True)
        )


class ChartQualitySpecialist:
    """Specialized assessor for chart and data visualization quality."""
    
    # Chart-specific quality criteria
    CHART_QUALITY_CRITERIA = {
        'bar_chart': {
            'required_elements': ['axes_labels', 'data_labels', 'title'],
            'optional_elements': ['legend', 'grid_lines'],
            'quality_checks': ['readable_text', 'appropriate_colors', 'clear_categories']
        },
        'line_chart': {
            'required_elements': ['x_axis_label', 'y_axis_label', 'data_points'],
            'optional_elements': ['legend', 'trend_lines'],
            'quality_checks': ['line_clarity', 'point_visibility', 'axis_scaling']
        },
        'pie_chart': {
            'required_elements': ['data_labels', 'percentages'],
            'optional_elements': ['legend', 'title'],
            'quality_checks': ['slice_clarity', 'color_contrast', 'label_readability']
        },
        'scatter_chart': {
            'required_elements': ['x_axis_label', 'y_axis_label', 'data_points'],
            'optional_elements': ['trend_line', 'legend', 'correlation_info'],
            'quality_checks': ['point_visibility', 'axis_scaling', 'pattern_clarity']
        },
        'histogram': {
            'required_elements': ['x_axis_label', 'y_axis_label', 'bins'],
            'optional_elements': ['title', 'distribution_curve'],
            'quality_checks': ['bin_appropriate', 'axis_scaling', 'distribution_clear']
        }
    }
    
    def create_chart_specific_prompt(self, image_path: str, chart_type: str = None) -> str:
        """Create chart-specific assessment prompt."""
        
        if not chart_type:
            # Try to detect chart type from filename
            filename = Path(image_path).stem.lower()
            for ctype in self.CHART_QUALITY_CRITERIA.keys():
                if ctype in filename or ctype.replace('_', '') in filename:
                    chart_type = ctype
                    break
        
        base_prompt = f"""You are reviewing this data visualization for production quality.

IMAGE FILE: {image_path}
CHART TYPE: {chart_type or 'Auto-detect from image'}

Provide detailed chart quality assessment in JSON format:
{{
  "chart_type_detected": "detected chart type",
  "overall_rating": "EXCELLENT|GOOD|NEEDS_IMPROVEMENT|POOR",
  "chart_quality_score": 0-100,
  "required_elements_present": {{
    "axes_labels": boolean,
    "data_labels": boolean,
    "title": boolean,
    "legend": boolean
  }},
  "quality_assessment": {{
    "text_readability": 0-100,
    "color_appropriateness": 0-100,
    "data_clarity": 0-100,
    "professional_appearance": 0-100,
    "accessibility_compliance": 0-100
  }},
  "specific_issues": [
    {{
      "element": "specific chart element",
      "issue": "description of problem",
      "severity": "critical|major|minor",
      "recommendation": "how to fix"
    }}
  ],
  "recommendations": ["list of improvements"]
}}

CHART QUALITY CRITERIA:

1. **Data Representation:**
   - Data is accurately and clearly represented
   - Chart type is appropriate for the data
   - Scale and proportions are correct
   - No misleading visual elements

2. **Labels and Text:**
   - All axes have clear, descriptive labels
   - Units are specified where appropriate
   - Text is large enough to read (minimum 10pt)
   - No overlapping or cut-off text

3. **Legend and Keys:**
   - Legend is present when needed
   - Legend items are clearly distinguishable
   - Color coding is consistent and meaningful

4. **Visual Design:**
   - Colors are professional and accessible
   - Sufficient contrast for readability
   - Grid lines aid without cluttering
   - Clean, uncluttered appearance

5. **Professional Standards:**
   - Suitable for business/academic presentation
   - Consistent with data visualization best practices
   - No distracting decorative elements
   - Appropriate aspect ratio"""
        
        # Add chart-specific criteria if we know the type
        if chart_type and chart_type in self.CHART_QUALITY_CRITERIA:
            criteria = self.CHART_QUALITY_CRITERIA[chart_type]
            base_prompt += f"""

CHART-SPECIFIC REQUIREMENTS for {chart_type}:
Required elements: {', '.join(criteria['required_elements'])}
Optional elements: {', '.join(criteria['optional_elements'])}
Quality checks: {', '.join(criteria['quality_checks'])}
"""
        
        base_prompt += "\n\nRespond ONLY with valid JSON."
        return base_prompt