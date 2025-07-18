"""Enhanced AUTO tag resolution system using AI models."""

import re
import logging
from typing import Dict, Any, List, Optional
from ..core.model import Model

logger = logging.getLogger(__name__)


class EnhancedAutoResolver:
    """AI-powered resolution of abstract task descriptions into executable prompts."""
    
    def __init__(self, model: Optional[Model] = None) -> None:
        self.model = model
        self.action_patterns = self._build_action_patterns()
        self.tool_mappings = self._build_tool_mappings()
    
    def _build_action_patterns(self) -> Dict[str, Dict[str, str]]:
        """Build patterns for common action types."""
        return {
            "search": {
                "prompt_template": "Search for information about: {query}. Return results with title, URL, and snippet for each result. Format as JSON array.",
                "output_format": "json",
                "tools_needed": ["web-search"]
            },
            "analyze": {
                "prompt_template": "Analyze the provided data: {data}. Extract key insights, patterns, and important findings. Return structured analysis with insights, themes, and summary.",
                "output_format": "structured",
                "tools_needed": ["data-processing"]
            },
            "generate": {
                "prompt_template": "Generate content based on: {instructions}. Follow the specified format and requirements. Ensure high quality and relevance.",
                "output_format": "text",
                "tools_needed": ["report-generator"]
            },
            "summarize": {
                "prompt_template": "Summarize the following content: {content}. Create a concise summary highlighting key points and main ideas.",
                "output_format": "text", 
                "tools_needed": []
            },
            "extract": {
                "prompt_template": "Extract information from: {source}. Focus on relevant content and structure the output appropriately.",
                "output_format": "structured",
                "tools_needed": ["headless-browser", "data-processing"]
            },
            "validate": {
                "prompt_template": "Validate the provided data: {data}. Check for completeness, accuracy, and compliance with requirements. Return validation results.",
                "output_format": "structured",
                "tools_needed": ["validation"]
            },
            "transform": {
                "prompt_template": "Transform the data: {data} according to: {requirements}. Apply necessary conversions and formatting.",
                "output_format": "structured",
                "tools_needed": ["data-processing"]
            }
        }
    
    def _build_tool_mappings(self) -> Dict[str, List[str]]:
        """Build mappings from action keywords to required tools."""
        return {
            # Web operations
            "search": ["web-search"],
            "web": ["web-search"],
            "scrape": ["headless-browser"],
            "browse": ["headless-browser"],
            "fetch": ["web-search", "headless-browser"],
            
            # Data operations  
            "analyze": ["data-processing"],
            "process": ["data-processing"],
            "transform": ["data-processing"],
            "filter": ["data-processing"],
            "clean": ["data-processing"],
            "validate": ["validation"],
            "verify": ["validation"],
            
            # Content generation
            "generate": ["report-generator"],
            "create": ["report-generator"],
            "write": ["report-generator"],
            "compose": ["report-generator"],
            "report": ["report-generator"],
            "document": ["report-generator"],
            
            # File operations
            "read": ["filesystem"],
            "write": ["filesystem"],  
            "save": ["filesystem"],
            "load": ["filesystem"],
            "file": ["filesystem"],
            
            # System operations
            "run": ["terminal"],
            "execute": ["terminal"],
            "command": ["terminal"],
            "shell": ["terminal"]
        }
    
    async def resolve_auto_tag(self, auto_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve an AUTO tag into executable task specification."""
        logger.debug(f"Resolving AUTO tag: {auto_content}")
        
        # Extract template variables
        variables = self._extract_template_variables(auto_content)
        
        # Determine action type and generate prompt
        action_type = self._classify_action(auto_content)
        
        if action_type in self.action_patterns:
            # Use predefined pattern
            pattern = self.action_patterns[action_type]
            prompt = self._apply_pattern(pattern, auto_content, context)
            tools = pattern["tools_needed"]
        else:
            # Use AI model to generate custom prompt
            prompt = await self._generate_custom_prompt(auto_content, context)
            tools = self._infer_tools_needed(auto_content)
        
        return {
            "prompt": prompt,
            "tools": tools,
            "variables": variables,
            "action_type": action_type,
            "output_format": self._infer_output_format(auto_content)
        }
    
    def _extract_template_variables(self, content: str) -> List[str]:
        """Extract template variables like {{topic}} from content."""
        return re.findall(r'\{\{([^}]+)\}\}', content)
    
    def _classify_action(self, content: str) -> str:
        """Classify the type of action based on content."""
        content_lower = content.lower()
        
        # Check for explicit action keywords
        for action_type in self.action_patterns.keys():
            if action_type in content_lower:
                return action_type
        
        # Pattern matching for common actions
        if any(word in content_lower for word in ["search", "find", "look", "query"]):
            return "search"
        elif any(word in content_lower for word in ["analyze", "examine", "study", "review"]):
            return "analyze"
        elif any(word in content_lower for word in ["generate", "create", "write", "produce"]):
            return "generate"
        elif any(word in content_lower for word in ["summarize", "summary", "brief"]):
            return "summarize"
        elif any(word in content_lower for word in ["extract", "scrape", "get", "fetch"]):
            return "extract"
        elif any(word in content_lower for word in ["validate", "verify", "check", "confirm"]):
            return "validate"
        elif any(word in content_lower for word in ["transform", "convert", "change", "modify"]):
            return "transform"
        else:
            return "custom"
    
    def _apply_pattern(self, pattern: Dict[str, str], content: str, context: Dict[str, Any]) -> str:
        """Apply a predefined pattern to generate a prompt."""
        template = pattern["prompt_template"]
        
        # Replace template variables with context values
        for var_match in re.finditer(r'\{\{([^}]+)\}\}', content):
            var_name = var_match.group(1).strip()
            if var_name in context:
                content = content.replace(var_match.group(0), str(context[var_name]))
        
        # Replace pattern placeholders
        if "{query}" in template:
            template = template.replace("{query}", content)
        elif "{data}" in template:
            template = template.replace("{data}", str(context.get("data", content)))
        elif "{content}" in template:
            template = template.replace("{content}", str(context.get("content", content)))
        elif "{instructions}" in template:
            template = template.replace("{instructions}", content)
        elif "{source}" in template:
            template = template.replace("{source}", str(context.get("source", content)))
        elif "{requirements}" in template:
            template = template.replace("{requirements}", content)
        
        return template
    
    async def _generate_custom_prompt(self, content: str, context: Dict[str, Any]) -> str:
        """Use AI model to generate a custom prompt for complex actions."""
        if not self.model:
            # Fallback to simple prompt if no model available
            return f"Execute the following task: {content}. Provide structured output with clear results."
        
        try:
            system_prompt = """You are an AI task prompt generator. Convert abstract task descriptions into clear, executable prompts for AI models.

Guidelines:
- Be specific about desired output format
- Include context variables where appropriate  
- Ensure the prompt is actionable and clear
- Return only the generated prompt, no explanation

Examples:
Input: "search web for information about quantum computing"
Output: "Search for recent information about quantum computing. Return results as JSON array with title, URL, snippet, and relevance score for each result."

Input: "analyze sales data and find trends" 
Output: "Analyze the provided sales data to identify trends, patterns, and insights. Return structured analysis with key findings, trend descriptions, and recommendations."
"""
            
            user_prompt = f"Convert this abstract task into an executable AI prompt: {content}"
            
            response = await self.model.generate(
                f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            )
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate custom prompt with AI model: {e}")
            return f"Execute the following task: {content}. Provide structured output with clear results."
    
    def _infer_tools_needed(self, content: str) -> List[str]:
        """Infer which tools are needed based on action content."""
        content_lower = content.lower()
        tools = set()
        
        for keyword, tool_list in self.tool_mappings.items():
            if keyword in content_lower:
                tools.update(tool_list)
        
        return list(tools)
    
    def _infer_output_format(self, content: str) -> str:
        """Infer the expected output format."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["json", "structured", "array", "list"]):
            return "json"
        elif any(word in content_lower for word in ["report", "document", "markdown"]):
            return "markdown"
        elif any(word in content_lower for word in ["summary", "brief", "text"]):
            return "text"
        else:
            return "structured"
    
    def get_suggested_tools_for_action(self, action: str) -> List[str]:
        """Get suggested tools for a given action description."""
        return self._infer_tools_needed(action)
    
    def validate_auto_tag_syntax(self, content: str) -> bool:
        """Validate that AUTO tag syntax is correct."""
        auto_pattern = r'<AUTO>.*?</AUTO>'
        matches = re.findall(auto_pattern, content, re.DOTALL)
        
        # Check for unclosed tags
        open_tags = content.count('<AUTO>')
        close_tags = content.count('</AUTO>')
        
        return open_tags == close_tags and len(matches) > 0