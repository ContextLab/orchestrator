"""Smart tool discovery engine for automatic tool mapping."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .base import default_registry

logger = logging.getLogger(__name__)


@dataclass
class ToolMatch:
    """Represents a tool matched to an action."""

    tool_name: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]


class ToolDiscoveryEngine:
    """Automatically maps actions to appropriate tools with smart discovery."""

    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry or default_registry
        self.action_patterns = self._build_action_patterns()
        self.semantic_mappings = self._build_semantic_mappings()
        self.tool_capabilities = self._analyze_tool_capabilities()

    def _build_action_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build regex patterns for common action types."""
        return {
            # Web operations
            r"search.*web|web.*search|find.*online|look.*up": {
                "tools": ["web-search"],
                "confidence": 0.95,
                "parameters": {"action": "search", "max_results": 10},
            },
            r"scrape.*web|extract.*web|crawl": {
                "tools": ["headless-browser"],
                "confidence": 0.90,
                "parameters": {"action": "scrape"},
            },
            r"browse.*website|visit.*site|navigate": {
                "tools": ["headless-browser"],
                "confidence": 0.85,
                "parameters": {"action": "browse"},
            },
            # Data operations
            r"analyze.*data|data.*analysis|process.*data": {
                "tools": ["data-processing"],
                "confidence": 0.95,
                "parameters": {"action": "analyze"},
            },
            r"clean.*data|sanitize.*data|normalize": {
                "tools": ["data-processing"],
                "confidence": 0.90,
                "parameters": {"action": "clean"},
            },
            r"transform.*data|convert.*data|format": {
                "tools": ["data-processing"],
                "confidence": 0.88,
                "parameters": {"action": "transform"},
            },
            r"validate.*data|verify.*data|check.*data": {
                "tools": ["validation"],
                "confidence": 0.92,
                "parameters": {"action": "validate"},
            },
            # Content generation
            r"generate.*report|create.*report|write.*report": {
                "tools": ["report-generator"],
                "confidence": 0.95,
                "parameters": {"format": "report"},
            },
            r"generate.*content|create.*content|write.*content": {
                "tools": ["report-generator"],
                "confidence": 0.85,
                "parameters": {"format": "content"},
            },
            r"summarize|create.*summary|brief": {
                "tools": ["report-generator"],
                "confidence": 0.80,
                "parameters": {"format": "summary"},
            },
            # File operations
            r"read.*file|load.*file|open.*file": {
                "tools": ["filesystem"],
                "confidence": 0.95,
                "parameters": {"action": "read"},
            },
            r"write.*file|save.*file|create.*file": {
                "tools": ["filesystem"],
                "confidence": 0.95,
                "parameters": {"action": "write"},
            },
            r"list.*files|directory.*contents|folder": {
                "tools": ["filesystem"],
                "confidence": 0.90,
                "parameters": {"action": "list"},
            },
            # System operations
            r"run.*command|execute.*command|shell": {
                "tools": ["terminal"],
                "confidence": 0.90,
                "parameters": {"action": "execute"},
            },
            r"install.*package|download.*tool|setup": {
                "tools": ["terminal"],
                "confidence": 0.85,
                "parameters": {"action": "install"},
            },
        }

    def _build_semantic_mappings(self) -> Dict[str, List[str]]:
        """Build semantic word mappings to tools."""
        return {
            # Search related
            "search": ["web-search"],
            "find": ["web-search", "filesystem"],
            "lookup": ["web-search"],
            "query": ["web-search"],
            "discover": ["web-search"],
            # Analysis related
            "analyze": ["data-processing"],
            "examine": ["data-processing"],
            "study": ["data-processing"],
            "investigate": ["data-processing"],
            "review": ["data-processing"],
            "inspect": ["data-processing"],
            # Generation related
            "generate": ["report-generator"],
            "create": ["report-generator", "filesystem"],
            "produce": ["report-generator"],
            "write": ["report-generator", "filesystem"],
            "compose": ["report-generator"],
            "draft": ["report-generator"],
            # Processing related
            "process": ["data-processing"],
            "transform": ["data-processing"],
            "convert": ["data-processing"],
            "clean": ["data-processing"],
            "filter": ["data-processing"],
            "sort": ["data-processing"],
            # Validation related
            "validate": ["validation"],
            "verify": ["validation"],
            "check": ["validation"],
            "confirm": ["validation"],
            "test": ["validation"],
            # Web operations
            "scrape": ["headless-browser"],
            "crawl": ["headless-browser"],
            "browse": ["headless-browser"],
            "navigate": ["headless-browser"],
            "click": ["headless-browser"],
            # File operations
            "read": ["filesystem"],
            "load": ["filesystem"],
            "save": ["filesystem"],
            "store": ["filesystem"],
            "file": ["filesystem"],
            # System operations
            "run": ["terminal"],
            "execute": ["terminal"],
            "command": ["terminal"],
            "install": ["terminal"],
            "setup": ["terminal"],
        }

    def _analyze_tool_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Analyze available tools and their capabilities."""
        capabilities = {}

        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                capabilities[tool_name] = {
                    "description": getattr(tool, "description", ""),
                    "input_types": getattr(tool, "input_types", []),
                    "output_types": getattr(tool, "output_types", []),
                    "capabilities": getattr(tool, "capabilities", []),
                    "tags": getattr(tool, "tags", []),
                }

        return capabilities

    def discover_tools_for_action(
        self, action_description: str, context: Dict[str, Any] = None
    ) -> List[ToolMatch]:
        """Discover appropriate tools for a given action description."""
        action_lower = action_description.lower()
        matches = []

        # 1. Pattern-based matching
        pattern_matches = self._match_by_patterns(action_lower)
        matches.extend(pattern_matches)

        # 2. Semantic word matching
        semantic_matches = self._match_by_semantics(action_lower)
        matches.extend(semantic_matches)

        # 3. Context-based enhancement
        if context:
            context_matches = self._enhance_with_context(matches, context)
            matches.extend(context_matches)

        # 4. Remove duplicates and sort by confidence
        unique_matches = self._deduplicate_and_rank(matches)

        # 5. Filter for available tools
        available_matches = [
            match for match in unique_matches if match.tool_name in self.tool_registry.list_tools()
        ]

        logger.debug(f"Discovered {len(available_matches)} tools for action: {action_description}")
        return available_matches

    def _match_by_patterns(self, action_lower: str) -> List[ToolMatch]:
        """Match actions using regex patterns."""
        matches = []

        for pattern, config in self.action_patterns.items():
            if re.search(pattern, action_lower):
                for tool_name in config["tools"]:
                    matches.append(
                        ToolMatch(
                            tool_name=tool_name,
                            confidence=config["confidence"],
                            reasoning=f"Pattern match: {pattern}",
                            parameters=config["parameters"].copy(),
                        )
                    )

        return matches

    def _match_by_semantics(self, action_lower: str) -> List[ToolMatch]:
        """Match actions using semantic word analysis."""
        matches = []
        words = re.findall(r"\b\w+\b", action_lower)

        for word in words:
            if word in self.semantic_mappings:
                for tool_name in self.semantic_mappings[word]:
                    matches.append(
                        ToolMatch(
                            tool_name=tool_name,
                            confidence=0.7,  # Lower confidence for semantic matches
                            reasoning=f"Semantic match: '{word}' → {tool_name}",
                            parameters={},
                        )
                    )

        return matches

    def _enhance_with_context(
        self, existing_matches: List[ToolMatch], context: Dict[str, Any]
    ) -> List[ToolMatch]:
        """Enhance tool selection based on context."""
        enhanced_matches = []

        # Check for data types in context
        if "data" in context:
            data = context["data"]
            if isinstance(data, (list, dict)):
                enhanced_matches.append(
                    ToolMatch(
                        tool_name="data-processing",
                        confidence=0.75,
                        reasoning="Context contains structured data",
                        parameters={"data_type": type(data).__name__},
                    )
                )

        # Check for URLs in context
        if any("url" in str(v).lower() or "http" in str(v) for v in context.values()):
            enhanced_matches.append(
                ToolMatch(
                    tool_name="headless-browser",
                    confidence=0.80,
                    reasoning="Context contains URLs",
                    parameters={"action": "fetch"},
                )
            )

        # Check for file paths in context
        if any("path" in k.lower() or "file" in k.lower() for k in context.keys()):
            enhanced_matches.append(
                ToolMatch(
                    tool_name="filesystem",
                    confidence=0.75,
                    reasoning="Context contains file references",
                    parameters={"action": "read"},
                )
            )

        return enhanced_matches

    def _deduplicate_and_rank(self, matches: List[ToolMatch]) -> List[ToolMatch]:
        """Remove duplicate tools and rank by confidence."""
        # Group by tool name and keep highest confidence
        best_matches = {}

        for match in matches:
            tool_name = match.tool_name
            if (
                tool_name not in best_matches
                or match.confidence > best_matches[tool_name].confidence
            ):
                best_matches[tool_name] = match

        # Sort by confidence descending
        return sorted(best_matches.values(), key=lambda x: x.confidence, reverse=True)

    def get_tool_chain_for_action(
        self, action_description: str, context: Dict[str, Any] = None
    ) -> List[ToolMatch]:
        """Get an ordered chain of tools to execute an action."""
        matches = self.discover_tools_for_action(action_description, context)

        # For now, return single best match or multiple complementary tools
        if not matches:
            return []

        # If we have a high-confidence match, use it
        best_match = matches[0]
        if best_match.confidence > 0.9:
            return [best_match]

        # Otherwise, consider tool combinations
        return self._build_tool_chain(matches, action_description)

    def _build_tool_chain(
        self, matches: List[ToolMatch], action_description: str
    ) -> List[ToolMatch]:
        """Build a chain of tools that work together."""
        action_lower = action_description.lower()

        # Common tool chains
        if "search" in action_lower and "analyze" in action_lower:
            # Search + analysis pipeline
            chain = []
            for match in matches:
                if match.tool_name == "web-search":
                    chain.append(match)
                elif match.tool_name == "data-processing":
                    chain.append(match)
            return chain

        elif "scrape" in action_lower and "process" in action_lower:
            # Scraping + processing pipeline
            chain = []
            for match in matches:
                if match.tool_name == "headless-browser":
                    chain.append(match)
                elif match.tool_name == "data-processing":
                    chain.append(match)
            return chain

        else:
            # Default: return best match
            return [matches[0]] if matches else []

    def validate_tool_for_action(self, tool_name: str, action_description: str) -> bool:
        """Validate that a tool is appropriate for an action."""
        matches = self.discover_tools_for_action(action_description)
        return any(match.tool_name == tool_name for match in matches)

    def suggest_missing_tools(self, required_tools: List[str]) -> Dict[str, List[str]]:
        """Suggest alternatives for missing tools."""
        available_tools = set(self.tool_registry.list_tools())
        missing_tools = [tool for tool in required_tools if tool not in available_tools]

        suggestions = {}
        for missing_tool in missing_tools:
            alternatives = []

            # Look for similar tools based on name patterns
            for available_tool in available_tools:
                if any(word in available_tool for word in missing_tool.split("-")):
                    alternatives.append(available_tool)

            suggestions[missing_tool] = alternatives

        return suggestions

    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get statistics on tool usage patterns."""
        # This would track tool usage in a real implementation
        return {tool: 0 for tool in self.tool_registry.list_tools()}
