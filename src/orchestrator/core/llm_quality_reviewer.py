"""
LLM-powered quality assessment system for pipeline outputs.

This module provides comprehensive quality review capabilities using
Claude Sonnet 4 and ChatGPT-5 with vision capabilities to assess
pipeline outputs for production quality.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .credential_manager import CredentialManager, CredentialConfig, create_credential_manager
from .quality_assessment import (
    ContentQuality, ContentQualityAssessor, IssueSeverity, 
    PipelineQualityReview, QualityIssue, QualityScorer,
    TemplateArtifactDetector, VisualQuality, OrganizationReview
)

# Import optional LLM client dependencies with fallbacks
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class LLMQualityError(Exception):
    """Raised when LLM quality assessment operations fail."""
    pass


class ModelConfig:
    """Configuration for LLM models used in quality assessment."""
    
    def __init__(
        self,
        name: str,
        supports_vision: bool = False,
        max_tokens: int = 4000,
        temperature: float = 0.1,
        rate_limit_per_minute: int = 60
    ):
        self.name = name
        self.supports_vision = supports_vision
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limit_per_minute = rate_limit_per_minute


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make an API call."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        # Check if we're under the limit
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)


class LLMClient:
    """Base class for LLM client implementations."""
    
    def __init__(self, model_config: ModelConfig, rate_limiter: RateLimiter):
        self.config = model_config
        self.rate_limiter = rate_limiter
    
    async def assess_content_quality(self, content: str, file_path: str = "") -> Dict[str, Any]:
        """Assess content quality using LLM."""
        raise NotImplementedError
    
    async def assess_visual_quality(self, image_path: str) -> Dict[str, Any]:
        """Assess visual content quality using vision model."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Anthropic Claude client for quality assessment."""
    
    def __init__(self, api_key: str, model_config: ModelConfig, rate_limiter: RateLimiter):
        super().__init__(model_config, rate_limiter)
        if not HAS_ANTHROPIC:
            raise LLMQualityError("Anthropic package not available")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    async def assess_content_quality(self, content: str, file_path: str = "") -> Dict[str, Any]:
        """Assess content quality using Claude."""
        await self.rate_limiter.acquire()
        
        prompt = self._create_content_assessment_prompt(content, file_path)
        
        try:
            response = self.client.messages.create(
                model=self.config.name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_assessment_response(response.content[0].text)
            
        except Exception as e:
            logger.error(f"Claude assessment failed: {e}")
            raise LLMQualityError(f"Claude assessment failed: {e}")
    
    async def assess_visual_quality(self, image_path: str) -> Dict[str, Any]:
        """Assess visual content using Claude's vision capabilities."""
        if not self.config.supports_vision:
            raise LLMQualityError("Model does not support vision capabilities")
        
        await self.rate_limiter.acquire()
        
        try:
            # Read image file
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Determine image media type
            image_type = self._get_image_media_type(image_path)
            
            prompt = self._create_visual_assessment_prompt(image_path)
            
            response = self.client.messages.create(
                model=self.config.name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_type,
                                    "data": image_data.hex()
                                }
                            }
                        ]
                    }
                ]
            )
            
            return self._parse_visual_assessment_response(response.content[0].text)
            
        except Exception as e:
            logger.error(f"Claude visual assessment failed: {e}")
            raise LLMQualityError(f"Claude visual assessment failed: {e}")
    
    def _get_image_media_type(self, image_path: str) -> str:
        """Determine image media type from file extension."""
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg', 
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(ext, 'image/png')
    
    def _create_content_assessment_prompt(self, content: str, file_path: str) -> str:
        """Create content assessment prompt for Claude."""
        return f"""You are reviewing content for production quality in a pipeline output system.

FILE: {file_path}
CONTENT: {content[:4000]}  # Truncate for token limits

Assess this content for production quality and provide a JSON response with:
{{
  "overall_rating": "CRITICAL|MAJOR|MINOR|ACCEPTABLE",
  "issues": [
    {{
      "category": "template_artifact|content_quality|completeness|professional_standards",
      "severity": "critical|major|minor",
      "description": "specific issue description",
      "suggestion": "how to fix this issue"
    }}
  ],
  "feedback": "detailed assessment explanation",
  "template_artifacts": boolean,
  "debug_artifacts": boolean,
  "conversational_tone": boolean,
  "incomplete_content": boolean
}}

CRITICAL ISSUES to detect:
- Unrendered templates: {{{{variable}}}} or similar artifacts
- Debug/conversational text: "Certainly!", "Here's the...", "I'll help you..."
- Incomplete content: Cut-off text, partial responses, truncated output
- Placeholder content: Lorem ipsum, TODO, TBD, [brackets]

PROFESSIONAL STANDARDS:
- Clear, professional formatting and presentation
- Accurate and complete content
- Appropriate for showcasing platform capabilities
- No debugging artifacts or temporary content

Respond ONLY with valid JSON."""
    
    def _create_visual_assessment_prompt(self, image_path: str) -> str:
        """Create visual assessment prompt for Claude."""
        return f"""You are reviewing this image for production quality in a pipeline output system.

IMAGE FILE: {image_path}

Assess this image for production quality and provide a JSON response with:
{{
  "overall_rating": "EXCELLENT|GOOD|NEEDS_IMPROVEMENT|POOR",
  "issues": [
    {{
      "category": "visual_quality",
      "severity": "critical|major|minor", 
      "description": "specific visual issue",
      "suggestion": "how to improve"
    }}
  ],
  "feedback": "detailed visual assessment",
  "image_renders_correctly": boolean,
  "charts_have_labels": boolean,
  "professional_appearance": boolean,
  "appropriate_styling": boolean
}}

QUALITY CHECKS:
1. Image renders correctly (no corruption, clear visibility)
2. Charts have proper labels, legends, readable text
3. Visual quality is professional-grade
4. Content matches expected visualization type
5. Colors and styling are appropriate
6. No artifacts or rendering errors

Respond ONLY with valid JSON."""
    
    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured assessment."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing if no JSON found
                return {
                    "overall_rating": "MAJOR",
                    "issues": [],
                    "feedback": response,
                    "template_artifacts": False,
                    "debug_artifacts": False,
                    "conversational_tone": False,
                    "incomplete_content": False
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:200]}...")
            return {
                "overall_rating": "MAJOR",
                "issues": [],
                "feedback": response,
                "template_artifacts": False,
                "debug_artifacts": False,
                "conversational_tone": False,
                "incomplete_content": False
            }
    
    def _parse_visual_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse visual assessment response."""
        return self._parse_assessment_response(response)


class OpenAIClient(LLMClient):
    """OpenAI GPT client for quality assessment."""
    
    def __init__(self, api_key: str, model_config: ModelConfig, rate_limiter: RateLimiter):
        super().__init__(model_config, rate_limiter)
        if not HAS_OPENAI:
            raise LLMQualityError("OpenAI package not available")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    async def assess_content_quality(self, content: str, file_path: str = "") -> Dict[str, Any]:
        """Assess content quality using GPT."""
        await self.rate_limiter.acquire()
        
        prompt = self._create_content_assessment_prompt(content, file_path)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_assessment_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"GPT assessment failed: {e}")
            raise LLMQualityError(f"GPT assessment failed: {e}")
    
    async def assess_visual_quality(self, image_path: str) -> Dict[str, Any]:
        """Assess visual content using GPT vision capabilities."""
        if not self.config.supports_vision:
            raise LLMQualityError("Model does not support vision capabilities")
        
        await self.rate_limiter.acquire()
        
        try:
            import base64
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            image_type = self._get_image_media_type(image_path)
            prompt = self._create_visual_assessment_prompt(image_path)
            
            response = self.client.chat.completions.create(
                model=self.config.name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            return self._parse_visual_assessment_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"GPT visual assessment failed: {e}")
            raise LLMQualityError(f"GPT visual assessment failed: {e}")
    
    def _get_image_media_type(self, image_path: str) -> str:
        """Determine image media type from file extension."""
        ext = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(ext, 'image/png')
    
    def _create_content_assessment_prompt(self, content: str, file_path: str) -> str:
        """Create content assessment prompt for GPT."""
        return f"""You are reviewing content for production quality in a pipeline output system.

FILE: {file_path}
CONTENT: {content[:4000]}  # Truncate for token limits

Assess this content for production quality and provide a JSON response with:
{{
  "overall_rating": "CRITICAL|MAJOR|MINOR|ACCEPTABLE",
  "issues": [
    {{
      "category": "template_artifact|content_quality|completeness|professional_standards",
      "severity": "critical|major|minor",
      "description": "specific issue description", 
      "suggestion": "how to fix this issue"
    }}
  ],
  "feedback": "detailed assessment explanation",
  "template_artifacts": boolean,
  "debug_artifacts": boolean,
  "conversational_tone": boolean,
  "incomplete_content": boolean
}}

CRITICAL ISSUES to detect:
- Unrendered templates: {{{{variable}}}} or similar artifacts
- Debug/conversational text: "Certainly!", "Here's the...", "I'll help you..."
- Incomplete content: Cut-off text, partial responses, truncated output
- Placeholder content: Lorem ipsum, TODO, TBD, [brackets]

PROFESSIONAL STANDARDS:
- Clear, professional formatting and presentation
- Accurate and complete content
- Appropriate for showcasing platform capabilities
- No debugging artifacts or temporary content

Respond ONLY with valid JSON."""
    
    def _create_visual_assessment_prompt(self, image_path: str) -> str:
        """Create visual assessment prompt for GPT.""" 
        return f"""You are reviewing this image for production quality in a pipeline output system.

IMAGE FILE: {image_path}

Assess this image for production quality and provide a JSON response with:
{{
  "overall_rating": "EXCELLENT|GOOD|NEEDS_IMPROVEMENT|POOR",
  "issues": [
    {{
      "category": "visual_quality",
      "severity": "critical|major|minor",
      "description": "specific visual issue",
      "suggestion": "how to improve"
    }}
  ],
  "feedback": "detailed visual assessment",
  "image_renders_correctly": boolean,
  "charts_have_labels": boolean, 
  "professional_appearance": boolean,
  "appropriate_styling": boolean
}}

QUALITY CHECKS:
1. Image renders correctly (no corruption, clear visibility)
2. Charts have proper labels, legends, readable text
3. Visual quality is professional-grade
4. Content matches expected visualization type
5. Colors and styling are appropriate
6. No artifacts or rendering errors

Respond ONLY with valid JSON."""
    
    def _parse_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured assessment."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback parsing if no JSON found
                return {
                    "overall_rating": "MAJOR", 
                    "issues": [],
                    "feedback": response,
                    "template_artifacts": False,
                    "debug_artifacts": False,
                    "conversational_tone": False,
                    "incomplete_content": False
                }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:200]}...")
            return {
                "overall_rating": "MAJOR",
                "issues": [],
                "feedback": response,
                "template_artifacts": False,
                "debug_artifacts": False,
                "conversational_tone": False,
                "incomplete_content": False
            }
    
    def _parse_visual_assessment_response(self, response: str) -> Dict[str, Any]:
        """Parse visual assessment response."""
        return self._parse_assessment_response(response)


class LLMQualityReviewer:
    """Comprehensive LLM-powered quality assessment system."""
    
    # Default model configurations
    DEFAULT_MODELS = {
        "claude-sonnet-4": ModelConfig(
            name="claude-3-5-sonnet-20241022",
            supports_vision=True,
            max_tokens=4000,
            rate_limit_per_minute=50
        ),
        "chatgpt-5": ModelConfig(
            name="gpt-4o",  # Using GPT-4o as GPT-5 placeholder
            supports_vision=True,
            max_tokens=4000,
            rate_limit_per_minute=60
        ),
        "claude-3-sonnet": ModelConfig(
            name="claude-3-sonnet-20240229",
            supports_vision=True,
            max_tokens=4000,
            rate_limit_per_minute=50
        ),
        "gpt-4-vision": ModelConfig(
            name="gpt-4-vision-preview",
            supports_vision=True,
            max_tokens=4000,
            rate_limit_per_minute=30
        )
    }
    
    def __init__(
        self,
        credential_manager: Optional[CredentialManager] = None,
        primary_models: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None
    ):
        self.credential_manager = credential_manager or create_credential_manager()
        
        # Model prioritization
        self.primary_models = primary_models or ["claude-sonnet-4", "chatgpt-5"]
        self.fallback_models = fallback_models or ["claude-3-sonnet", "gpt-4-vision"]
        
        # Initialize clients
        self.clients = {}
        self.rate_limiters = {}
        self._initialize_clients()
        
        # Quality assessment components
        self.content_assessor = ContentQualityAssessor()
        self.template_detector = TemplateArtifactDetector()
        self.quality_scorer = QualityScorer()
        
        logger.info(f"Initialized LLM Quality Reviewer with {len(self.clients)} model clients")
    
    def _initialize_clients(self):
        """Initialize LLM clients with credentials from existing system."""
        all_models = self.primary_models + self.fallback_models
        
        for model_name in all_models:
            if model_name not in self.DEFAULT_MODELS:
                logger.warning(f"Unknown model configuration: {model_name}")
                continue
            
            model_config = self.DEFAULT_MODELS[model_name]
            rate_limiter = RateLimiter(model_config.rate_limit_per_minute)
            self.rate_limiters[model_name] = rate_limiter
            
            try:
                if "claude" in model_name.lower():
                    # Get Anthropic API key
                    api_key = self._get_api_key("anthropic", "api_key")
                    if api_key and HAS_ANTHROPIC:
                        self.clients[model_name] = AnthropicClient(api_key, model_config, rate_limiter)
                        logger.info(f"Initialized Claude client: {model_name}")
                    else:
                        logger.warning(f"Could not initialize Claude client: {model_name}")
                
                elif "gpt" in model_name.lower() or "chatgpt" in model_name.lower():
                    # Get OpenAI API key
                    api_key = self._get_api_key("openai", "api_key")
                    if api_key and HAS_OPENAI:
                        self.clients[model_name] = OpenAIClient(api_key, model_config, rate_limiter)
                        logger.info(f"Initialized GPT client: {model_name}")
                    else:
                        logger.warning(f"Could not initialize GPT client: {model_name}")
                        
            except Exception as e:
                logger.error(f"Failed to initialize client for {model_name}: {e}")
    
    def _get_api_key(self, service: str, key: str) -> Optional[str]:
        """Get API key from credential manager or environment variables."""
        # Try credential manager first
        try:
            credential = self.credential_manager.retrieve_credential(service, key)
            if credential:
                return credential
        except Exception as e:
            logger.debug(f"Could not retrieve {service}/{key} from credential manager: {e}")
        
        # Fall back to environment variables
        env_var_patterns = [
            f"{service.upper()}_{key.upper()}",
            f"{service.upper()}_API_KEY",
            f"ANTHROPIC_API_KEY" if service == "anthropic" else f"OPENAI_API_KEY"
        ]
        
        for env_var in env_var_patterns:
            value = os.environ.get(env_var)
            if value:
                logger.debug(f"Found {service} API key in environment variable {env_var}")
                return value
        
        logger.warning(f"Could not find API key for {service}/{key}")
        return None
    
    async def review_pipeline_outputs(self, pipeline_name: str) -> PipelineQualityReview:
        """Comprehensive quality review of a pipeline's outputs."""
        start_time = time.time()
        
        # Get pipeline output directory
        outputs_path = Path("examples/outputs") / pipeline_name
        if not outputs_path.exists():
            raise LLMQualityError(f"Pipeline output directory not found: {outputs_path}")
        
        logger.info(f"Starting quality review for pipeline: {pipeline_name}")
        
        # Scan all output files
        files = self._scan_output_directory(outputs_path)
        logger.info(f"Found {len(files)} files to review in {pipeline_name}")
        
        # Categorize files by type
        text_files = [f for f in files if f.suffix.lower() in ['.md', '.txt', '.csv', '.json', '.html']]
        image_files = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']]
        
        # Collect all issues
        all_issues = []
        reviewed_files = []
        
        # Review text files
        for file_path in text_files:
            try:
                issues = await self._review_text_file(file_path)
                all_issues.extend(issues)
                reviewed_files.append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to review text file {file_path}: {e}")
        
        # Review image files (if vision models available)
        for file_path in image_files:
            try:
                issues = await self._review_image_file(file_path)
                all_issues.extend(issues)
                reviewed_files.append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to review image file {file_path}: {e}")
        
        # Categorize issues by severity
        critical_issues = [issue for issue in all_issues if issue.severity == IssueSeverity.CRITICAL]
        major_issues = [issue for issue in all_issues if issue.severity == IssueSeverity.MAJOR]
        minor_issues = [issue for issue in all_issues if issue.severity == IssueSeverity.MINOR]
        
        # Calculate overall quality score
        overall_score = self.quality_scorer.calculate_score(all_issues)
        production_ready = self.quality_scorer.determine_production_readiness(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        review_duration = time.time() - start_time
        
        review = PipelineQualityReview(
            pipeline_name=pipeline_name,
            overall_score=overall_score,
            files_reviewed=reviewed_files,
            critical_issues=critical_issues,
            major_issues=major_issues,
            minor_issues=minor_issues,
            recommendations=recommendations,
            production_ready=production_ready,
            reviewer_model=self._get_primary_model_name(),
            review_duration_seconds=review_duration
        )
        
        logger.info(f"Completed quality review for {pipeline_name}: Score {overall_score}/100, "
                   f"{len(all_issues)} issues found, Production ready: {production_ready}")
        
        return review
    
    def _scan_output_directory(self, outputs_path: Path) -> List[Path]:
        """Scan directory for all reviewable files."""
        files = []
        
        # Supported file extensions
        supported_extensions = {'.md', '.txt', '.csv', '.json', '.html', '.png', '.jpg', '.jpeg', '.gif', '.webp'}
        
        for file_path in outputs_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        return sorted(files)
    
    async def _review_text_file(self, file_path: Path) -> List[QualityIssue]:
        """Review a single text file for quality issues."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return []
        
        # First do rule-based assessment
        rule_based_quality = self.content_assessor.assess_content_quality(content, str(file_path))
        issues = rule_based_quality.issues.copy()
        
        # Then try LLM-based assessment if available
        llm_client = self._get_available_client()
        if llm_client:
            try:
                llm_response = await llm_client.assess_content_quality(content, str(file_path))
                llm_issues = self._parse_llm_content_issues(llm_response, str(file_path))
                issues.extend(llm_issues)
            except Exception as e:
                logger.warning(f"LLM assessment failed for {file_path}: {e}")
        
        return issues
    
    async def _review_image_file(self, file_path: Path) -> List[QualityIssue]:
        """Review a single image file for quality issues."""
        # Get vision-capable client
        vision_client = self._get_vision_client()
        if not vision_client:
            logger.warning(f"No vision-capable client available for {file_path}")
            return []
        
        try:
            llm_response = await vision_client.assess_visual_quality(str(file_path))
            return self._parse_llm_visual_issues(llm_response, str(file_path))
        except Exception as e:
            logger.warning(f"Visual assessment failed for {file_path}: {e}")
            return []
    
    def _get_available_client(self) -> Optional[LLMClient]:
        """Get first available LLM client."""
        # Try primary models first
        for model_name in self.primary_models:
            if model_name in self.clients:
                return self.clients[model_name]
        
        # Then fallback models
        for model_name in self.fallback_models:
            if model_name in self.clients:
                return self.clients[model_name]
        
        return None
    
    def _get_vision_client(self) -> Optional[LLMClient]:
        """Get first available vision-capable client."""
        all_models = self.primary_models + self.fallback_models
        
        for model_name in all_models:
            if (model_name in self.clients and 
                model_name in self.DEFAULT_MODELS and
                self.DEFAULT_MODELS[model_name].supports_vision):
                return self.clients[model_name]
        
        return None
    
    def _get_primary_model_name(self) -> str:
        """Get name of primary model being used."""
        for model_name in self.primary_models:
            if model_name in self.clients:
                return model_name
        
        for model_name in self.fallback_models:
            if model_name in self.clients:
                return model_name
        
        return "none"
    
    def _parse_llm_content_issues(self, response: Dict[str, Any], file_path: str) -> List[QualityIssue]:
        """Parse LLM content assessment response into QualityIssue objects."""
        issues = []
        
        for issue_data in response.get('issues', []):
            try:
                from .quality_assessment import IssueCategory
                
                # Map category strings to enums
                category_map = {
                    'template_artifact': IssueCategory.TEMPLATE_ARTIFACT,
                    'content_quality': IssueCategory.CONTENT_QUALITY,
                    'completeness': IssueCategory.COMPLETENESS,
                    'professional_standards': IssueCategory.PROFESSIONAL_STANDARDS
                }
                
                severity_map = {
                    'critical': IssueSeverity.CRITICAL,
                    'major': IssueSeverity.MAJOR,
                    'minor': IssueSeverity.MINOR
                }
                
                category = category_map.get(issue_data.get('category'), IssueCategory.CONTENT_QUALITY)
                severity = severity_map.get(issue_data.get('severity'), IssueSeverity.MAJOR)
                
                issues.append(QualityIssue(
                    category=category,
                    severity=severity,
                    description=issue_data.get('description', ''),
                    file_path=file_path,
                    suggestion=issue_data.get('suggestion', '')
                ))
                
            except Exception as e:
                logger.warning(f"Failed to parse LLM issue: {issue_data} - {e}")
        
        return issues
    
    def _parse_llm_visual_issues(self, response: Dict[str, Any], file_path: str) -> List[QualityIssue]:
        """Parse LLM visual assessment response into QualityIssue objects."""
        issues = []
        
        for issue_data in response.get('issues', []):
            try:
                from .quality_assessment import IssueCategory
                
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
                    file_path=file_path,
                    suggestion=issue_data.get('suggestion', '')
                ))
                
            except Exception as e:
                logger.warning(f"Failed to parse visual issue: {issue_data} - {e}")
        
        return issues
    
    def _generate_recommendations(self, issues: List[QualityIssue]) -> List[str]:
        """Generate actionable recommendations based on issues found."""
        recommendations = []
        
        # Group issues by category
        issues_by_category = {}
        for issue in issues:
            if issue.category not in issues_by_category:
                issues_by_category[issue.category] = []
            issues_by_category[issue.category].append(issue)
        
        # Generate category-specific recommendations
        from .quality_assessment import IssueCategory
        
        if IssueCategory.TEMPLATE_ARTIFACT in issues_by_category:
            recommendations.append("Fix all unrendered template variables before production deployment")
        
        if IssueCategory.CONTENT_QUALITY in issues_by_category:
            recommendations.append("Review and improve content quality, removing conversational tone and debug artifacts")
        
        if IssueCategory.COMPLETENESS in issues_by_category:
            recommendations.append("Ensure all content is complete and replace any placeholder text")
        
        if IssueCategory.VISUAL_QUALITY in issues_by_category:
            recommendations.append("Improve visual content quality, ensuring charts have proper labels and professional appearance")
        
        if IssueCategory.FILE_ORGANIZATION in issues_by_category:
            recommendations.append("Fix file organization issues, ensuring proper naming conventions and locations")
        
        # Add general recommendations based on issue count
        critical_count = len([i for i in issues if i.severity == IssueSeverity.CRITICAL])
        if critical_count > 0:
            recommendations.insert(0, f"Address {critical_count} critical issues immediately - pipeline not ready for production")
        
        return recommendations