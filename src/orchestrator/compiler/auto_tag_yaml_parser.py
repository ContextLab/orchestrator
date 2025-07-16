"""YAML parser that properly handles <AUTO> tags with arbitrary content."""

import re
import yaml
from typing import Dict, Any, List, Tuple, Optional
import json


class AutoTagYAMLParser:
    """
    A YAML parser that handles <AUTO> tags by temporarily replacing them
    with JSON-encoded strings that are YAML-safe.
    """
    
    def __init__(self):
        # Pattern to match AUTO tags (non-greedy to handle nested tags)
        self.auto_tag_pattern = re.compile(r'<AUTO>(.*?)</AUTO>', re.DOTALL)
        self.placeholder_prefix = "__AUTO_TAG_PLACEHOLDER_"
        self.placeholder_suffix = "__"
        self.tag_registry: Dict[str, str] = {}
        
    def parse(self, yaml_content: str) -> Dict[str, Any]:
        """
        Parse YAML content containing <AUTO> tags.
        
        Args:
            yaml_content: YAML string with potential <AUTO> tags
            
        Returns:
            Parsed YAML structure with <AUTO> tags preserved as strings
        """
        # Clear registry for new parse
        self.tag_registry.clear()
        
        # Step 1: Replace AUTO tags with placeholders
        processed_yaml = self._replace_auto_tags(yaml_content)
        
        # Step 2: Parse the YAML
        try:
            parsed_data = yaml.safe_load(processed_yaml)
        except yaml.YAMLError as e:
            # Provide helpful error context
            lines = processed_yaml.split('\n')
            for i, line in enumerate(lines, 1):
                print(f"{i:3d}: {line}")
            raise ValueError(f"Failed to parse YAML after AUTO tag processing: {e}")
        
        # Step 3: Restore AUTO tags
        restored_data = self._restore_auto_tags(parsed_data)
        
        return restored_data
    
    def _replace_auto_tags(self, content: str) -> str:
        """
        Replace all AUTO tags with YAML-safe placeholders.
        
        This handles nested tags by processing innermost first.
        """
        result = content
        counter = 0
        
        # Keep replacing until no more AUTO tags
        while True:
            # Find all AUTO tag matches
            matches = list(self.auto_tag_pattern.finditer(result))
            
            if not matches:
                break
                
            # Find innermost tags (those without AUTO tags inside)
            innermost_matches = []
            for match in matches:
                inner_content = match.group(1)
                if '<AUTO>' not in inner_content:
                    innermost_matches.append(match)
            
            # If no innermost found, we have only nested tags left
            if not innermost_matches:
                # Process from end to beginning to maintain positions
                for match in reversed(matches):
                    counter += 1
                    placeholder_key = f"{self.placeholder_prefix}{counter}{self.placeholder_suffix}"
                    
                    # Store the complete AUTO tag
                    self.tag_registry[placeholder_key] = match.group(0)
                    
                    # Replace with a JSON-encoded string that's YAML-safe
                    json_safe = json.dumps(placeholder_key)
                    
                    start, end = match.span()
                    result = result[:start] + json_safe + result[end:]
            else:
                # Process innermost tags from end to beginning
                for match in reversed(innermost_matches):
                    counter += 1
                    placeholder_key = f"{self.placeholder_prefix}{counter}{self.placeholder_suffix}"
                    
                    # Store the complete AUTO tag
                    self.tag_registry[placeholder_key] = match.group(0)
                    
                    # Replace with a JSON-encoded string that's YAML-safe
                    json_safe = json.dumps(placeholder_key)
                    
                    start, end = match.span()
                    result = result[:start] + json_safe + result[end:]
        
        return result
    
    def _restore_auto_tags(self, data: Any) -> Any:
        """
        Recursively restore AUTO tags from placeholders.
        """
        if isinstance(data, dict):
            return {
                key: self._restore_auto_tags(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._restore_auto_tags(item) for item in data]
        elif isinstance(data, str):
            # Check if this is a placeholder
            if data.startswith(self.placeholder_prefix) and data.endswith(self.placeholder_suffix):
                if data in self.tag_registry:
                    # Restore the original AUTO tag
                    original = self.tag_registry[data]
                    
                    # Check if the original contains nested placeholders
                    # and recursively restore them
                    for placeholder, auto_tag in self.tag_registry.items():
                        if placeholder in original and placeholder != data:
                            original = original.replace(f'"{placeholder}"', auto_tag)
                            original = original.replace(placeholder, auto_tag)
                    
                    return original
            return data
        else:
            return data
    
    def extract_auto_content(self, value: str) -> Optional[str]:
        """
        Extract content from an AUTO tag if the value is one.
        
        Args:
            value: String that might be an AUTO tag
            
        Returns:
            Content inside the AUTO tag, or None if not an AUTO tag
        """
        if isinstance(value, str) and value.strip().startswith('<AUTO>') and value.strip().endswith('</AUTO>'):
            match = self.auto_tag_pattern.match(value.strip())
            if match:
                return match.group(1).strip()
        return None
    
    def find_auto_tags(self, data: Any, path: str = "") -> List[Tuple[str, str]]:
        """
        Find all AUTO tags in the data structure.
        
        Returns:
            List of (path, content) tuples
        """
        results = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, str):
                    content = self.extract_auto_content(value)
                    if content is not None:
                        results.append((new_path, content))
                else:
                    results.extend(self.find_auto_tags(value, new_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                if isinstance(item, str):
                    content = self.extract_auto_content(item)
                    if content is not None:
                        results.append((new_path, content))
                else:
                    results.extend(self.find_auto_tags(item, new_path))
        
        return results


# Convenience function
def parse_yaml_with_auto_tags(yaml_content: str) -> Dict[str, Any]:
    """Parse YAML content that contains <AUTO> tags."""
    parser = AutoTagYAMLParser()
    return parser.parse(yaml_content)