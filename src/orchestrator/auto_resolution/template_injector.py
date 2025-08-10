"""Template Variable Injection for intelligent AUTO tag resolution.

This module automatically injects discovered context into AUTO tag prompts,
eliminating the need for users to explicitly reference variables.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .context_discovery import DiscoveredContext


@dataclass
class InjectionResult:
    """Result of template injection process."""
    
    enriched_prompt: str
    injected_variables: Dict[str, Any]
    template_references: List[str]
    original_intent: str


class TemplateInjector:
    """Automatically injects discovered context into AUTO tag prompts."""
    
    def __init__(self):
        """Initialize the template injector."""
        self.max_context_length = 2000  # Limit context size to avoid token issues
    
    def inject_context(
        self,
        auto_tag_content: str,
        discovered_context: DiscoveredContext,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> InjectionResult:
        """
        Build enriched prompt with automatically discovered context.
        
        Args:
            auto_tag_content: Original AUTO tag content (natural language)
            discovered_context: Context discovered by ContextDiscoveryEngine
            additional_context: Any additional context to include
            
        Returns:
            InjectionResult with enriched prompt and metadata
        """
        # Start with the original intent
        original_intent = auto_tag_content.strip()
        
        # Prepare the injected variables
        injected_vars = {}
        
        # Add discovered data
        if discovered_context.relevant_data:
            injected_vars.update(discovered_context.relevant_data)
        
        # Add additional context if provided
        if additional_context:
            injected_vars.update(additional_context)
        
        # Build the enriched prompt
        enriched_prompt = self._build_enriched_prompt(
            original_intent,
            discovered_context,
            injected_vars
        )
        
        # Get template references
        template_refs = discovered_context.suggested_references
        
        return InjectionResult(
            enriched_prompt=enriched_prompt,
            injected_variables=injected_vars,
            template_references=template_refs,
            original_intent=original_intent
        )
    
    def _build_enriched_prompt(
        self,
        intent: str,
        discovered: DiscoveredContext,
        variables: Dict[str, Any]
    ) -> str:
        """
        Build an enriched prompt with context for the LLM.
        
        Args:
            intent: Original user intent
            discovered: Discovered context
            variables: All available variables
            
        Returns:
            Enriched prompt string
        """
        # Handle case with no discovered context
        if not discovered.relevant_data:
            return self._build_fallback_prompt(intent, variables)
        
        # Build prompt with discovered context
        prompt_parts = []
        
        # Add the task description
        prompt_parts.append(f"Task: {intent}")
        prompt_parts.append("")
        
        # Add discovered context with clear labeling
        if discovered.discovered_paths:
            prompt_parts.append("Relevant context data discovered:")
            
            for path in discovered.discovered_paths[:5]:  # Limit to top 5
                data = discovered.relevant_data.get(path)
                if data is not None:
                    data_repr = self._format_data_for_prompt(data, path)
                    prompt_parts.append(data_repr)
            
            prompt_parts.append("")
        
        # Add instruction for resolution
        prompt_parts.append(f"Based on the context above, {intent}.")
        
        # Special handling for list extraction
        if self._is_list_extraction_intent(intent):
            prompt_parts.append("")
            prompt_parts.append("Return the appropriate list or array from the context.")
            prompt_parts.append("If multiple relevant lists exist, choose the most appropriate one.")
        
        # Special handling for iteration
        if self._is_iteration_intent(intent):
            prompt_parts.append("")
            prompt_parts.append("Return the collection of items to iterate over.")
            prompt_parts.append("Ensure the result is a valid list or array.")
        
        return "\n".join(prompt_parts)
    
    def _build_fallback_prompt(
        self,
        intent: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Build a fallback prompt when no specific context is discovered.
        
        Args:
            intent: Original user intent
            variables: All available variables
            
        Returns:
            Fallback prompt string
        """
        prompt_parts = [
            f"Task: {intent}",
            "",
            "Available context:"
        ]
        
        # Add a summary of available data
        if variables:
            for key, value in list(variables.items())[:10]:  # Limit to 10 items
                value_summary = self._summarize_value(value)
                prompt_parts.append(f"- {key}: {value_summary}")
        else:
            prompt_parts.append("No context data available.")
        
        prompt_parts.extend([
            "",
            f"Based on the available context, {intent}."
        ])
        
        return "\n".join(prompt_parts)
    
    def _format_data_for_prompt(
        self,
        data: Any,
        path: str,
        max_length: int = 500
    ) -> str:
        """
        Format data for inclusion in prompt.
        
        Args:
            data: Data to format
            path: Path to the data
            max_length: Maximum string length
            
        Returns:
            Formatted data representation
        """
        prefix = f"- {path}: "
        
        if isinstance(data, list):
            if not data:
                return f"{prefix}Empty list"
            
            # For lists, show count and sample
            sample_size = min(3, len(data))
            if all(isinstance(item, (str, int, float, bool)) for item in data[:sample_size]):
                # Simple list - show first few items
                sample = data[:sample_size]
                sample_str = json.dumps(sample, indent=2)
                if len(data) > sample_size:
                    return f"{prefix}List with {len(data)} items (showing first {sample_size}):\n  {sample_str}"
                return f"{prefix}List:\n  {sample_str}"
            else:
                # Complex list - describe structure
                return f"{prefix}List with {len(data)} complex items"
        
        elif isinstance(data, dict):
            # For dicts, show keys and structure
            if len(data) <= 5:
                data_str = json.dumps(data, indent=2, default=str)
                if len(data_str) <= max_length:
                    return f"{prefix}Dict:\n  {data_str}"
            
            keys = list(data.keys())[:5]
            return f"{prefix}Dict with keys: {', '.join(str(k) for k in keys)}"
        
        elif isinstance(data, str):
            if len(data) <= max_length:
                return f"{prefix}'{data}'"
            return f"{prefix}String ({len(data)} chars): '{data[:100]}...'"
        
        else:
            value_str = str(data)
            if len(value_str) <= max_length:
                return f"{prefix}{value_str}"
            return f"{prefix}{value_str[:100]}..."
    
    def _summarize_value(self, value: Any, max_length: int = 50) -> str:
        """
        Create a brief summary of a value.
        
        Args:
            value: Value to summarize
            max_length: Maximum summary length
            
        Returns:
            Brief summary string
        """
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            if len(value) <= max_length:
                return f"'{value}'"
            return f"String ({len(value)} chars)"
        elif isinstance(value, list):
            return f"List ({len(value)} items)"
        elif isinstance(value, dict):
            return f"Dict ({len(value)} keys)"
        else:
            return type(value).__name__
    
    def _is_list_extraction_intent(self, intent: str) -> bool:
        """
        Check if the intent is about extracting a list.
        
        Args:
            intent: User intent string
            
        Returns:
            True if intent is about list extraction
        """
        list_keywords = [
            "list of", "array of", "collection of",
            "all", "extract", "get", "return"
        ]
        intent_lower = intent.lower()
        return any(kw in intent_lower for kw in list_keywords)
    
    def _is_iteration_intent(self, intent: str) -> bool:
        """
        Check if the intent is about iteration.
        
        Args:
            intent: User intent string
            
        Returns:
            True if intent is about iteration
        """
        iteration_keywords = [
            "iterate", "loop", "for each", "process each",
            "verify each", "check each", "validate each",
            "every", "all items"
        ]
        intent_lower = intent.lower()
        return any(kw in intent_lower for kw in iteration_keywords)
    
    def create_structured_output_prompt(
        self,
        injection_result: InjectionResult,
        expected_type: str = "any"
    ) -> str:
        """
        Create a prompt specifically for structured output generation.
        
        Args:
            injection_result: Result from inject_context
            expected_type: Expected output type (list, dict, string, etc.)
            
        Returns:
            Prompt optimized for structured output
        """
        base_prompt = injection_result.enriched_prompt
        
        # Add type-specific instructions
        type_instructions = {
            "list": "\n\nReturn a JSON array containing the requested items.",
            "dict": "\n\nReturn a JSON object with the requested structure.",
            "string": "\n\nReturn a single string value.",
            "number": "\n\nReturn a numeric value.",
            "boolean": "\n\nReturn true or false.",
            "any": "\n\nReturn the appropriate data structure based on the request."
        }
        
        instruction = type_instructions.get(expected_type, type_instructions["any"])
        
        return base_prompt + instruction