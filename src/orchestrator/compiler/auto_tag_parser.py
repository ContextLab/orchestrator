"""YAML parser that handles <AUTO> tags with arbitrary content including special characters."""

import re
import yaml
from typing import Dict, Any, List, Tuple, Optional
import uuid


class AutoTagParser:
    """Parser that handles <AUTO> tags in YAML content."""
    
    def __init__(self):
        self.auto_tag_pattern = re.compile(r'<AUTO>(.*?)</AUTO>', re.DOTALL)
        self.placeholder_prefix = "AUTO_PLACEHOLDER_"
        self.tag_map: Dict[str, str] = {}
    
    def parse_yaml_with_auto_tags(self, yaml_content: str) -> Dict[str, Any]:
        """
        Parse YAML content that contains <AUTO> tags.
        
        This method:
        1. Extracts all <AUTO> tags and replaces them with safe placeholders
        2. Parses the YAML with standard parser
        3. Restores the <AUTO> tags in the parsed structure
        
        Args:
            yaml_content: YAML string potentially containing <AUTO> tags
            
        Returns:
            Parsed YAML structure with <AUTO> tags preserved
        """
        # Step 1: Extract and replace AUTO tags with placeholders
        processed_yaml = self._extract_auto_tags(yaml_content)
        
        # Step 2: Parse the processed YAML
        try:
            parsed_data = yaml.safe_load(processed_yaml)
        except yaml.YAMLError as e:
            # If parsing fails, provide helpful error message
            raise ValueError(f"Failed to parse YAML after AUTO tag extraction: {e}")
        
        # Step 3: Restore AUTO tags in the parsed structure
        restored_data = self._restore_auto_tags(parsed_data)
        
        return restored_data
    
    def _extract_auto_tags(self, content: str) -> str:
        """
        Extract AUTO tags and replace with placeholders.
        
        Handles nested tags by processing from innermost to outermost.
        """
        self.tag_map.clear()
        processed_content = content
        
        # Keep extracting until no more AUTO tags are found
        iteration = 0
        while True:
            # Find all AUTO tags
            matches = list(self.auto_tag_pattern.finditer(processed_content))
            
            if not matches:
                break
            
            # Find the innermost tag (the one with no AUTO tags inside it)
            innermost_match = None
            for match in matches:
                inner_content = match.group(1)
                # Check if this match contains any AUTO tags
                if not self.auto_tag_pattern.search(inner_content):
                    innermost_match = match
                    break
            
            if not innermost_match:
                # All remaining matches have nested tags, process the first one
                innermost_match = matches[0]
            
            # Generate unique placeholder
            placeholder_id = str(uuid.uuid4()).replace('-', '')
            # Don't add quotes if we're inside a value that will be quoted
            # Check if this AUTO tag is the entire value on a line
            start, end = innermost_match.span()
            
            # Look for context around the match
            line_start = processed_content.rfind('\n', 0, start) + 1
            line_end = processed_content.find('\n', end)
            if line_end == -1:
                line_end = len(processed_content)
            
            line = processed_content[line_start:line_end]
            before_match = processed_content[line_start:start]
            after_match = processed_content[end:line_end]
            
            # Check if the AUTO tag is a complete YAML value
            if ':' in before_match and before_match.strip().endswith(':'):
                # This is a complete value, use quotes
                placeholder = f'"{self.placeholder_prefix}{placeholder_id}"'
            else:
                # This might be inside another AUTO tag, don't add quotes
                placeholder = f'{self.placeholder_prefix}{placeholder_id}'
            
            # Store the full AUTO tag
            full_tag = innermost_match.group(0)
            self.tag_map[placeholder] = full_tag
            
            # Replace the AUTO tag with placeholder
            processed_content = (
                processed_content[:start] + 
                placeholder + 
                processed_content[end:]
            )
            
            iteration += 1
            if iteration > 100:  # Safety check
                raise ValueError("Too many iterations processing AUTO tags")
        
        return processed_content
    
    def _restore_auto_tags(self, data: Any) -> Any:
        """
        Recursively restore AUTO tags in the parsed data structure.
        """
        if isinstance(data, dict):
            return {
                key: self._restore_auto_tags(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._restore_auto_tags(item) for item in data]
        elif isinstance(data, str):
            # Check if this string is a placeholder
            for placeholder, auto_tag in self.tag_map.items():
                # Remove quotes from placeholder for comparison
                clean_placeholder = placeholder.strip('"')
                if data == clean_placeholder:
                    return auto_tag
            return data
        else:
            return data
    
    def extract_auto_tag_content(self, value: str) -> Optional[str]:
        """
        Extract the content from an AUTO tag.
        
        Args:
            value: String that may contain an AUTO tag
            
        Returns:
            The content inside the AUTO tag, or None if not an AUTO tag
        """
        if isinstance(value, str):
            match = self.auto_tag_pattern.match(value.strip())
            if match:
                return match.group(1).strip()
        return None
    
    def find_all_auto_tags(self, data: Any) -> List[Tuple[str, str]]:
        """
        Find all AUTO tags in a data structure.
        
        Returns:
            List of tuples (path, content) where path is like "steps.0.parameters.output_format"
        """
        auto_tags = []
        
        def _traverse(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, str):
                        content = self.extract_auto_tag_content(value)
                        if content:
                            auto_tags.append((new_path, content))
                    else:
                        _traverse(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}.{i}"
                    if isinstance(item, str):
                        content = self.extract_auto_tag_content(item)
                        if content:
                            auto_tags.append((new_path, content))
                    else:
                        _traverse(item, new_path)
        
        _traverse(data)
        return auto_tags


def parse_yaml_with_auto_tags(yaml_content: str) -> Dict[str, Any]:
    """
    Convenience function to parse YAML with AUTO tags.
    
    Args:
        yaml_content: YAML string potentially containing <AUTO> tags
        
    Returns:
        Parsed YAML structure with <AUTO> tags preserved
    """
    parser = AutoTagParser()
    return parser.parse_yaml_with_auto_tags(yaml_content)