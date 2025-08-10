"""Context Discovery Engine for intelligent AUTO tag resolution.

This module automatically discovers relevant context data based on
natural language intent in AUTO tags, eliminating the need for
explicit variable references.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class DiscoveredContext:
    """Represents discovered context data for AUTO tag resolution."""
    
    # Discovered data mapped by relevance score
    relevant_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata about discovery
    discovered_paths: List[str] = field(default_factory=list)  # e.g., ["extract_sources.result.sources"]
    keywords_matched: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Suggested variable references for template injection
    suggested_references: List[str] = field(default_factory=list)
    
    def get_highest_confidence_data(self) -> Tuple[str, Any]:
        """Get the data with highest confidence score."""
        if not self.confidence_scores:
            return None, None
            
        best_path = max(self.confidence_scores, key=self.confidence_scores.get)
        return best_path, self.relevant_data.get(best_path)
    
    def to_context_dict(self) -> Dict[str, Any]:
        """Convert discovered context to dictionary for template injection."""
        return {
            "discovered_data": self.relevant_data,
            "data_paths": self.discovered_paths,
            "matched_keywords": self.keywords_matched
        }


class ContextDiscoveryEngine:
    """Automatically discovers relevant context for AUTO tags based on intent."""
    
    def __init__(self):
        """Initialize the context discovery engine."""
        # Common keywords that indicate data types
        self.data_type_keywords = {
            "list": ["list", "array", "items", "collection", "set"],
            "sources": ["source", "sources", "reference", "references", "citation", "citations"],
            "claims": ["claim", "claims", "assertion", "assertions", "statement", "statements"],
            "files": ["file", "files", "document", "documents", "path", "paths"],
            "results": ["result", "results", "output", "outputs", "response", "responses"],
            "data": ["data", "information", "content", "value", "values"],
            "analysis": ["analysis", "analyzed", "processed", "extracted", "parsed"],
            "verification": ["verify", "verification", "check", "validate", "validation", "fact-check"]
        }
        
        # Keywords that suggest iteration/looping
        self.iteration_keywords = [
            "all", "each", "every", "iterate", "loop", "for each",
            "process", "handle", "verify", "check", "validate"
        ]
    
    def discover_relevant_data(
        self,
        intent: str,
        step_results: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None
    ) -> DiscoveredContext:
        """
        Discover relevant data based on natural language intent.
        
        Args:
            intent: Natural language intent from AUTO tag (e.g., "list of sources to verify")
            step_results: Results from previous pipeline steps
            variables: Additional variables from pipeline context
            
        Returns:
            DiscoveredContext with relevant data and metadata
        """
        discovered = DiscoveredContext()
        
        # Extract keywords from intent
        keywords = self._extract_keywords(intent.lower())
        discovered.keywords_matched = keywords
        
        # Search step results for relevant data
        for step_id, result in step_results.items():
            relevance_score = self._calculate_relevance(
                step_id, result, keywords, intent
            )
            
            if relevance_score > 0.3:  # Threshold for relevance
                # Find the most relevant data within the result
                relevant_paths = self._find_relevant_paths(
                    result, keywords, prefix=f"{step_id}.result"
                )
                
                for path, data in relevant_paths:
                    discovered.relevant_data[path] = data
                    discovered.discovered_paths.append(path)
                    discovered.confidence_scores[path] = relevance_score
                    discovered.suggested_references.append(f"{{{{ {path} }}}}")
        
        # Also search in variables if provided
        if variables:
            for var_name, var_value in variables.items():
                relevance_score = self._calculate_relevance(
                    var_name, var_value, keywords, intent
                )
                
                if relevance_score > 0.3:
                    discovered.relevant_data[var_name] = var_value
                    discovered.discovered_paths.append(var_name)
                    discovered.confidence_scores[var_name] = relevance_score
                    discovered.suggested_references.append(f"{{{{ {var_name} }}}}")
        
        return discovered
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from natural language text.
        
        Args:
            text: Natural language text
            
        Returns:
            List of extracted keywords
        """
        # Remove common words
        stop_words = {"the", "a", "an", "to", "of", "in", "for", "that", "need", "needs"}
        
        # Split into words and filter
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Add data type keywords if they match
        for category, category_keywords in self.data_type_keywords.items():
            if any(kw in text for kw in category_keywords):
                keywords.append(category)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_relevance(
        self,
        data_id: str,
        data: Any,
        keywords: List[str],
        intent: str
    ) -> float:
        """
        Calculate relevance score between data and intent.
        
        Args:
            data_id: Identifier of the data (step_id or variable name)
            data: The actual data value
            keywords: Extracted keywords from intent
            intent: Original intent string
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0
        
        # Check if data_id contains keywords
        data_id_lower = data_id.lower()
        for keyword in keywords:
            if keyword in data_id_lower:
                score += 0.3
        
        # Check data structure
        if isinstance(data, dict):
            # Check keys for keyword matches
            for key in data.keys():
                key_lower = str(key).lower()
                for keyword in keywords:
                    if keyword in key_lower:
                        score += 0.2
            
            # Special handling for result objects
            if "result" in data:
                nested_score = self._calculate_relevance(
                    f"{data_id}.result",
                    data["result"],
                    keywords,
                    intent
                )
                score += nested_score * 0.5
        
        elif isinstance(data, list):
            # Lists are highly relevant for iteration
            if any(kw in keywords for kw in ["list", "all", "each", "items"]):
                score += 0.4
            
            # Check if it's a list of sources/claims/etc
            if data and isinstance(data[0], dict):
                sample = data[0]
                for keyword in keywords:
                    if any(keyword in str(k).lower() for k in sample.keys()):
                        score += 0.3
                        break
        
        # Check for iteration keywords in intent
        if any(iter_kw in intent.lower() for iter_kw in self.iteration_keywords):
            if isinstance(data, (list, tuple)):
                score += 0.2
        
        # Cap score at 1.0
        return min(score, 1.0)
    
    def _find_relevant_paths(
        self,
        data: Any,
        keywords: List[str],
        prefix: str = "",
        max_depth: int = 5
    ) -> List[Tuple[str, Any]]:
        """
        Recursively find relevant data paths within a nested structure.
        
        Args:
            data: Data structure to search
            keywords: Keywords to match
            prefix: Path prefix for nested data
            max_depth: Maximum recursion depth
            
        Returns:
            List of (path, data) tuples for relevant data
        """
        relevant_paths = []
        
        if max_depth <= 0:
            return relevant_paths
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{prefix}.{key}" if prefix else key
                key_lower = str(key).lower()
                
                # Check if this key is relevant
                is_relevant = any(kw in key_lower for kw in keywords)
                
                if is_relevant:
                    relevant_paths.append((current_path, value))
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    nested_paths = self._find_relevant_paths(
                        value, keywords, current_path, max_depth - 1
                    )
                    relevant_paths.extend(nested_paths)
        
        elif isinstance(data, list) and prefix:
            # For lists, check if the list itself is relevant
            if any(kw in ["list", "items", "all", "each"] for kw in keywords):
                relevant_paths.append((prefix, data))
        
        return relevant_paths
    
    def suggest_template_references(
        self,
        discovered: DiscoveredContext
    ) -> List[str]:
        """
        Suggest template variable references based on discovered context.
        
        Args:
            discovered: DiscoveredContext from discovery
            
        Returns:
            List of suggested template references
        """
        suggestions = []
        
        # Sort paths by confidence score
        sorted_paths = sorted(
            discovered.confidence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for path, score in sorted_paths[:3]:  # Top 3 suggestions
            suggestions.append(f"{{{{ {path} }}}}")
        
        return suggestions
    
    def build_enriched_context(
        self,
        intent: str,
        discovered: DiscoveredContext
    ) -> str:
        """
        Build an enriched context description for AUTO tag resolution.
        
        Args:
            intent: Original intent from AUTO tag
            discovered: Discovered context data
            
        Returns:
            Enriched context description
        """
        if not discovered.relevant_data:
            return f"Task: {intent}\n\nNo relevant data found in context."
        
        # Get the most relevant data
        best_path, best_data = discovered.get_highest_confidence_data()
        
        context_description = f"""Task: {intent}

Available relevant data:
"""
        
        # Add discovered data descriptions
        for path in discovered.discovered_paths[:3]:  # Limit to top 3
            data = discovered.relevant_data.get(path)
            if data is not None:
                data_description = self._describe_data(data)
                context_description += f"\n- {path}: {data_description}"
        
        context_description += f"""

Based on the available data above, {intent}"""
        
        return context_description
    
    def _describe_data(self, data: Any, max_length: int = 100) -> str:
        """
        Create a brief description of data for context.
        
        Args:
            data: Data to describe
            max_length: Maximum description length
            
        Returns:
            Brief description of the data
        """
        if isinstance(data, list):
            return f"List with {len(data)} items"
        elif isinstance(data, dict):
            keys = list(data.keys())[:3]
            return f"Dict with keys: {', '.join(str(k) for k in keys)}"
        elif isinstance(data, str):
            if len(data) > max_length:
                return f"String: {data[:max_length]}..."
            return f"String: {data}"
        else:
            return f"{type(data).__name__}: {str(data)[:max_length]}"