"""Handles nested AUTO tags."""

import re
from typing import Any, Dict, List, Optional, Tuple

from .models import AutoTagContext, AutoTagNestingError
from .resolver import LazyAutoTagResolver


class NestedAutoTagHandler:
    """Handles nested AUTO tags with proper depth tracking."""
    
    def __init__(self, resolver: Optional[LazyAutoTagResolver] = None):
        self.resolver = resolver
        # Non-greedy pattern to match AUTO tags
        self.auto_tag_pattern = re.compile(r'<AUTO>(.*?)</AUTO>', re.DOTALL)
    
    async def resolve_nested(
        self,
        content: str,
        context: AutoTagContext,
        max_depth: int = 5
    ) -> str:
        """Recursively resolve nested AUTO tags.
        
        Args:
            content: Content that may contain AUTO tags
            context: Current resolution context
            max_depth: Maximum nesting depth allowed
            
        Returns:
            Content with all AUTO tags resolved
            
        Raises:
            AutoTagNestingError: If maximum depth exceeded
        """
        if context.resolution_depth >= max_depth:
            raise AutoTagNestingError(
                f"Maximum nesting depth {max_depth} exceeded at depth {context.resolution_depth}"
            )
        
        # Find all AUTO tags in content
        auto_tags = self._find_auto_tags(content)
        
        if not auto_tags:
            return content
        
        # Log nested resolution
        if hasattr(self.resolver, 'logger'):
            for tag in auto_tags:
                self.resolver.logger.log_nested_resolution(
                    context.resolution_depth,
                    content[:50] + "...",
                    tag["content"][:50] + "..."
                )
        
        # Resolve inner-most tags first (depth-first)
        resolved_content = content
        for tag in reversed(auto_tags):  # Process inner-most first
            # Create nested context
            nested_context = self._create_nested_context(context, tag)
            
            # Resolve the tag
            resolution = await self.resolver.resolve(
                tag["content"],
                nested_context
            )
            
            # Apply the action plan
            replacement_value = await self._apply_action_plan(
                resolution.resolved_value,
                resolution.action_plan,
                nested_context
            )
            
            # Replace the tag with resolved value
            resolved_content = resolved_content.replace(
                tag["full_tag"],
                str(replacement_value)
            )
        
        # Check if new AUTO tags were introduced
        if self._has_auto_tags(resolved_content):
            # Recursively resolve new tags
            deeper_context = AutoTagContext(
                pipeline=context.pipeline,
                current_task_id=context.current_task_id,
                tag_location=context.tag_location,
                variables=context.variables,
                step_results=context.step_results,
                loop_context=context.loop_context,
                resolution_depth=context.resolution_depth + 1,
                parent_resolutions=context.parent_resolutions
            )
            
            return await self.resolve_nested(resolved_content, deeper_context, max_depth)
        
        return resolved_content
    
    def _find_auto_tags(self, content: str) -> List[Dict[str, Any]]:
        """Find all AUTO tags in content with their positions."""
        tags = []
        stack = []
        i = 0
        
        while i < len(content):
            # Look for opening tag
            if content[i:].startswith("<AUTO>"):
                start = i
                i += 6  # len("<AUTO>")
                stack.append(start)
            # Look for closing tag
            elif content[i:].startswith("</AUTO>"):
                if stack:
                    start = stack.pop()
                    end = i + 7  # len("</AUTO>")
                    tag_content = content[start + 6:i]  # Content between tags
                    
                    # Calculate depth based on remaining stack
                    depth = len(stack)
                    
                    tags.append({
                        "full_tag": content[start:end],
                        "content": tag_content.strip(),
                        "start": start,
                        "end": end,
                        "depth": depth
                    })
                i += 7
            else:
                i += 1
        
        # Sort by depth (deepest first) then by position
        tags.sort(key=lambda t: (-t["depth"], t["start"]))
        
        return tags
    
    def _calculate_depth(self, content: str, start_pos: int, end_pos: int) -> int:
        """Calculate nesting depth for a tag."""
        depth = 0
        
        # Count how many other AUTO tags contain this one
        for match in self.auto_tag_pattern.finditer(content):
            if match.start() < start_pos and match.end() > end_pos:
                depth += 1
        
        return depth
    
    def _has_auto_tags(self, content: str) -> bool:
        """Check if content contains AUTO tags."""
        return bool(self.auto_tag_pattern.search(content))
    
    def _create_nested_context(
        self, 
        parent_context: AutoTagContext,
        tag_info: Dict[str, Any]
    ) -> AutoTagContext:
        """Create context for nested AUTO tag resolution."""
        # Update tag location to indicate nesting
        nested_location = f"{parent_context.tag_location}[nested@{tag_info['start']}]"
        
        return AutoTagContext(
            pipeline=parent_context.pipeline,
            current_task_id=parent_context.current_task_id,
            tag_location=nested_location,
            variables=parent_context.variables,
            step_results=parent_context.step_results,
            loop_context=parent_context.loop_context,
            resolution_depth=parent_context.resolution_depth + 1,
            parent_resolutions=parent_context.parent_resolutions
        )
    
    async def _apply_action_plan(
        self,
        resolved_value: Any,
        action_plan: Any,
        context: AutoTagContext
    ) -> Any:
        """Apply action plan to get final value.
        
        Args:
            resolved_value: The resolved value
            action_plan: Action plan from determination
            context: Resolution context
            
        Returns:
            Final value after applying action
        """
        if action_plan.action_type == "return_value":
            # Simple return
            return resolved_value
            
        elif action_plan.action_type == "call_tool":
            # Tool execution would happen here
            # For now, return the value with tool metadata
            return {
                "_tool_call": {
                    "tool": action_plan.tool_name,
                    "parameters": action_plan.tool_parameters,
                    "input": resolved_value
                }
            }
            
        elif action_plan.action_type == "save_file":
            # File saving would happen here
            # For now, return file path
            return action_plan.file_path
            
        elif action_plan.action_type == "update_context":
            # Context update would happen here
            # For now, return the value
            return resolved_value
            
        elif action_plan.action_type == "chain_resolution":
            # Chain resolution would trigger here
            # For now, return the next AUTO tag
            return f"<AUTO>{action_plan.next_auto_tag}</AUTO>"
            
        else:
            # Unknown action type
            return resolved_value
    
    def extract_auto_tags(self, content: str) -> List[str]:
        """Extract all AUTO tag contents from a string.
        
        Args:
            content: String that may contain AUTO tags
            
        Returns:
            List of AUTO tag contents (without markers)
        """
        tags = self._find_auto_tags(content)
        return [tag["content"] for tag in tags]
    
    def replace_auto_tag(
        self,
        content: str,
        tag_content: str,
        replacement: str
    ) -> str:
        """Replace a specific AUTO tag with a value.
        
        Args:
            content: Content containing AUTO tags
            tag_content: The content of the AUTO tag to replace
            replacement: Replacement value
            
        Returns:
            Content with tag replaced
        """
        full_tag = f"<AUTO>{tag_content}</AUTO>"
        return content.replace(full_tag, replacement)
    
    def count_auto_tags(self, content: str) -> int:
        """Count number of AUTO tags in content.
        
        Args:
            content: String to check
            
        Returns:
            Number of AUTO tags found
        """
        return len(self._find_auto_tags(content))