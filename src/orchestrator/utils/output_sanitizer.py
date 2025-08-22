"""Output sanitizer to remove conversational markers and fluff from AI model outputs."""

import re
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class OutputSanitizer:
    """
    A utility class to clean AI model outputs by removing conversational markers,
    unnecessary prefixes/suffixes, and other "fluff" content that doesn't add value
    to the actual output.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the OutputSanitizer.
        
        Args:
            enabled: Whether the sanitizer is active. If False, returns input unchanged.
        """
        self.enabled = enabled
        
        # Common conversational starters that should be removed
        self.conversational_starters = [
            r"^Certainly!?\s*",
            r"^Sure!?\s*",
            r"^Of course!?\s*",
            r"^Absolutely!?\s*",
            r"^I'd be happy to\s+[^.]*\.\s*",
            r"^I can help\s+[^.]*\.\s*",
            r"^I'll help\s+[^.]*\.\s*",
            r"^Let me help\s+[^.]*\.\s*",
            r"^I'm happy to\s+[^.]*\.\s*",
            r"^I'd be glad to\s+[^.]*\.\s*",
            r"^No problem!?\s*",
            r"^Here is\s+[^:]*:\s*",
            r"^Here are\s+[^:]*:\s*",
            r"^Here's\s+[^:]*:\s*",
            r"^Below is\s+[^:]*:\s*",
            r"^Below are\s+[^:]*:\s*",
            r"^The following is\s+[^:]*:\s*",
            r"^The following are\s+[^:]*:\s*",
            r"^As requested,?\s*",
            r"^As you requested,?\s*",
            r"^Based on your request,?\s*",
            # Specific fixes for edge cases
            r"^Certainly!\s*(?=[A-Z])",  # Only match if followed by actual content
            r"^Sure!\s*(?=[A-Z])",  # Only match if followed by actual content
        ]
        
        # Patterns for removing conversational closings
        self.conversational_endings = [
            r"\s*Let me know if you need.*?\.?\s*$",
            r"\s*Feel free to ask.*?\.?\s*$",
            r"\s*If you have any.*?questions.*?\.?\s*$",
            r"\s*Please let me know.*?\.?\s*$",
            r"\s*I hope this helps.*?\.?\s*$",
            r"\s*Hope this helps.*?\.?\s*$",
            r"\s*Is there anything else.*?\?\s*$",
            r"\s*Would you like.*?more.*?\?\s*$",
            r"\s*Do you need.*?help.*?\?\s*$",
            r"\s*Let me know if.*?\.?\s*$",
        ]
        
        # Patterns for conversational transitions/fillers
        self.conversational_fillers = [
            r"Now,?\s+let's\s+.*?\.\s*",
            r"First,?\s+let's\s+.*?\.\s*",
            r"To begin,?\s+.*?\.\s*",
            r"Let's start by\s+.*?\.\s*",
            r"I'll start by\s+.*?\.\s*",
            r"I'll begin by\s+.*?\.\s*",
            r"Let me start by\s+.*?\.\s*",
            r"Let me begin by\s+.*?\.\s*",
            r"So,?\s+",
            r"Well,?\s+",
            r"Actually,?\s+",
            r"In fact,?\s+",
        ]
        
        # Meta-commentary patterns that add no value
        self.meta_commentary = [
            r"I'll\s+(create|provide|write|generate|build)\s+[^.]*\.?\s*",
            r"I'm going to\s+(create|provide|write|generate|build)\s+[^.]*\.?\s*",
            r"I will\s+(create|provide|write|generate|build)\s+[^.]*\.?\s*",
            r"Let me\s+(create|provide|write|generate|build)\s+[^.]*\.?\s*",
            r"I'll now\s+(create|provide|write|generate|build)\s+[^.]*\.?\s*",
            r"Here's what I'll do:\s*",
            r"This is what I'll\s+[^:]*:\s*",
            r"What I'll do is\s+[^.]*\.?\s*",
        ]
        
        # Compile all patterns for better performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_starters = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                 for pattern in self.conversational_starters]
        self.compiled_endings = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                for pattern in self.conversational_endings]
        self.compiled_fillers = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                for pattern in self.conversational_fillers]
        self.compiled_meta = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                             for pattern in self.meta_commentary]
    
    def sanitize(self, text: Union[str, Dict, List]) -> Union[str, Dict, List]:
        """
        Sanitize text by removing conversational markers and fluff.
        
        Args:
            text: Input text, dict, or list to sanitize. If not a string, returns unchanged.
            
        Returns:
            Sanitized output with conversational markers removed.
        """
        if not self.enabled:
            return text
        
        # Only process strings
        if not isinstance(text, str):
            return text
        
        if not text or not text.strip():
            return text
        
        original_text = text
        sanitized = text.strip()
        
        try:
            # Remove conversational starters
            for pattern in self.compiled_starters:
                sanitized = pattern.sub("", sanitized).strip()
            
            # Remove conversational endings
            for pattern in self.compiled_endings:
                sanitized = pattern.sub("", sanitized).strip()
            
            # Remove conversational fillers
            for pattern in self.compiled_fillers:
                sanitized = pattern.sub("", sanitized).strip()
            
            # Remove meta-commentary
            for pattern in self.compiled_meta:
                sanitized = pattern.sub("", sanitized).strip()
            
            # Clean up multiple newlines and spaces
            sanitized = re.sub(r'\n\s*\n\s*\n', '\n\n', sanitized)  # Max 2 consecutive newlines
            sanitized = re.sub(r'[ \t]+', ' ', sanitized)  # Multiple spaces to single space
            sanitized = sanitized.strip()
            
            # Handle whitespace-only strings
            if not sanitized or sanitized.isspace():
                if not original_text.strip():
                    return ""
                else:
                    logger.debug("OutputSanitizer: Result is whitespace-only, keeping original")
                    return original_text
            
            # If we've removed everything or just left punctuation, return original
            if not sanitized or len(sanitized.strip('.,!?;: \n\t')) < 3:
                logger.debug("OutputSanitizer: Sanitization removed too much content, keeping original")
                return original_text
            
            # Log if significant changes were made
            if len(original_text) - len(sanitized) > 50:
                logger.debug(f"OutputSanitizer: Removed {len(original_text) - len(sanitized)} characters")
                logger.debug(f"OutputSanitizer: Original start: '{original_text[:100]}...'")
                logger.debug(f"OutputSanitizer: Sanitized start: '{sanitized[:100]}...'")
            
            return sanitized
            
        except Exception as e:
            logger.warning(f"OutputSanitizer: Error during sanitization: {e}")
            return original_text
    
    def sanitize_batch(self, texts: List[str]) -> List[str]:
        """
        Sanitize a batch of texts.
        
        Args:
            texts: List of texts to sanitize
            
        Returns:
            List of sanitized texts
        """
        if not self.enabled:
            return texts
        
        return [self.sanitize(text) for text in texts]
    
    def add_custom_pattern(self, pattern: str, pattern_type: str = "starter"):
        """
        Add a custom pattern to remove from outputs.
        
        Args:
            pattern: Regex pattern to remove
            pattern_type: Type of pattern - "starter", "ending", "filler", or "meta"
        """
        if pattern_type == "starter":
            self.conversational_starters.append(pattern)
            self.compiled_starters.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
        elif pattern_type == "ending":
            self.conversational_endings.append(pattern)
            self.compiled_endings.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
        elif pattern_type == "filler":
            self.conversational_fillers.append(pattern)
            self.compiled_fillers.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
        elif pattern_type == "meta":
            self.meta_commentary.append(pattern)
            self.compiled_meta.append(re.compile(pattern, re.IGNORECASE | re.MULTILINE))
        else:
            raise ValueError(f"Invalid pattern_type: {pattern_type}")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable the sanitizer."""
        self.enabled = enabled
    
    def is_enabled(self) -> bool:
        """Check if the sanitizer is enabled."""
        return self.enabled


# Global instance for easy access
_default_sanitizer = OutputSanitizer()

def sanitize_output(text: Union[str, Dict, List], enabled: bool = True) -> Union[str, Dict, List]:
    """
    Convenience function to sanitize text using the default sanitizer.
    
    Args:
        text: Text to sanitize
        enabled: Whether sanitization should be applied
        
    Returns:
        Sanitized text
    """
    if enabled:
        return _default_sanitizer.sanitize(text)
    return text


def configure_sanitizer(enabled: bool = True, custom_patterns: Optional[Dict[str, List[str]]] = None):
    """
    Configure the default sanitizer.
    
    Args:
        enabled: Whether to enable sanitization
        custom_patterns: Dict of pattern types to lists of custom patterns
    """
    global _default_sanitizer
    _default_sanitizer.set_enabled(enabled)
    
    if custom_patterns:
        for pattern_type, patterns in custom_patterns.items():
            for pattern in patterns:
                _default_sanitizer.add_custom_pattern(pattern, pattern_type)