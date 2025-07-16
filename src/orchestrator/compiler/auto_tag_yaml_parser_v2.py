"""YAML parser that properly handles <AUTO> tags with arbitrary content including nested tags."""

import re
import yaml
from typing import Dict, Any, List, Tuple, Optional, NamedTuple
import json


class AutoTag(NamedTuple):
    """Represents an AUTO tag with its position and content."""
    start: int
    end: int
    content: str
    full_text: str


class AutoTagYAMLParser:
    """
    A YAML parser that handles <AUTO> tags by parsing them with proper nesting support.
    """
    
    def __init__(self):
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
    
    def _find_auto_tags(self, content: str) -> List[AutoTag]:
        """
        Find all AUTO tags in content, handling nesting properly.
        
        Returns:
            List of AutoTag objects sorted by start position
        """
        tags = []
        
        # Find all opening and closing tags
        open_pattern = re.compile(r'<AUTO>')
        close_pattern = re.compile(r'</AUTO>')
        
        # Find all positions
        open_positions = [(m.start(), m.end()) for m in open_pattern.finditer(content)]
        close_positions = [(m.start(), m.end()) for m in close_pattern.finditer(content)]
        
        # Match opening and closing tags
        for open_start, open_end in open_positions:
            # Find the matching close tag
            depth = 1
            close_start = None
            close_end = None
            
            pos = open_end
            while depth > 0 and pos < len(content):
                # Look for next open or close tag
                next_open = content.find('<AUTO>', pos)
                next_close = content.find('</AUTO>', pos)
                
                if next_close == -1:
                    # No closing tag found
                    break
                
                if next_open != -1 and next_open < next_close:
                    # Found another opening tag first
                    depth += 1
                    pos = next_open + 6  # len('<AUTO>')
                else:
                    # Found a closing tag
                    depth -= 1
                    if depth == 0:
                        close_start = next_close
                        close_end = next_close + 7  # len('</AUTO>')
                    pos = next_close + 7
            
            if close_start is not None:
                # Extract the content between tags
                inner_content = content[open_end:close_start]
                full_text = content[open_start:close_end]
                
                tags.append(AutoTag(
                    start=open_start,
                    end=close_end,
                    content=inner_content,
                    full_text=full_text
                ))
        
        return sorted(tags, key=lambda t: t.start)
    
    def _replace_auto_tags(self, content: str) -> str:
        """
        Replace all AUTO tags with YAML-safe placeholders.
        
        This handles nested tags by processing innermost first.
        """
        result = content
        counter = 0
        
        # Keep replacing until no more AUTO tags
        while '<AUTO>' in result:
            # Find all AUTO tags
            tags = self._find_auto_tags(result)
            
            if not tags:
                break
            
            # Find innermost tags (those that don't contain other AUTO tags)
            innermost_tags = []
            for tag in tags:
                # Check if this tag contains other tags
                is_innermost = True
                for other in tags:
                    if other != tag and other.start > tag.start and other.end < tag.end:
                        is_innermost = False
                        break
                if is_innermost:
                    innermost_tags.append(tag)
            
            # Replace innermost tags from end to beginning (to preserve positions)
            for tag in reversed(sorted(innermost_tags, key=lambda t: t.start)):
                counter += 1
                placeholder_key = f"{self.placeholder_prefix}{counter}{self.placeholder_suffix}"
                
                # Store the complete AUTO tag
                self.tag_registry[placeholder_key] = tag.full_text
                
                # Replace with a JSON-encoded string that's YAML-safe
                json_safe = json.dumps(placeholder_key)
                
                # Replace in result
                result = result[:tag.start] + json_safe + result[tag.end:]
        
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
                    # Get the original AUTO tag
                    original = self.tag_registry[data]
                    
                    # Recursively restore any nested placeholders
                    restored = original
                    for placeholder, auto_tag in self.tag_registry.items():
                        if placeholder != data:  # Don't replace self
                            # Try both quoted and unquoted versions
                            restored = restored.replace(f'"{placeholder}"', auto_tag)
                            restored = restored.replace(placeholder, auto_tag)
                    
                    return restored
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
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('<AUTO>') and value.endswith('</AUTO>'):
                # Extract content properly
                return value[6:-7]  # Remove <AUTO> and </AUTO>
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