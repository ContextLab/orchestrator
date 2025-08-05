"""Structured models for loop condition evaluation and tracking."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoopCondition:
    """Represents a comprehensive loop termination condition."""
    
    # Core properties
    expression: str
    condition_type: Literal["until", "while"] = "until"
    
    # Analysis results
    has_auto_tags: bool = False
    has_templates: bool = False
    dependencies: Set[str] = field(default_factory=set)
    complexity_score: int = 0
    
    # Runtime state
    resolved_expression: Optional[str] = None
    last_evaluation: Optional[bool] = None
    evaluation_history: List[Tuple[int, str, bool, float]] = field(default_factory=list)  # iteration, resolved, result, timestamp
    
    # Performance tracking
    total_evaluations: int = 0
    avg_evaluation_time: float = 0.0
    cache_hits: int = 0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    loop_id: Optional[str] = None
    
    def record_evaluation(self, iteration: int, resolved_expr: str, result: bool, eval_time: float) -> None:
        """Record an evaluation result with performance metrics."""
        timestamp = time.time()
        
        # Update state
        self.resolved_expression = resolved_expr
        self.last_evaluation = result
        self.total_evaluations += 1
        
        # Update performance metrics
        if self.total_evaluations == 1:
            self.avg_evaluation_time = eval_time
        else:
            # Running average
            self.avg_evaluation_time = (
                (self.avg_evaluation_time * (self.total_evaluations - 1) + eval_time) 
                / self.total_evaluations
            )
        
        # Add to history (keep last 100 evaluations)
        self.evaluation_history.append((iteration, resolved_expr, result, timestamp))
        if len(self.evaluation_history) > 100:
            self.evaluation_history.pop(0)
        
        logger.debug(f"Recorded evaluation for {self.condition_type} condition: "
                    f"iteration={iteration}, result={result}, time={eval_time:.3f}s")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debugging information for this condition."""
        return {
            "expression": self.expression,
            "condition_type": self.condition_type,
            "analysis": {
                "has_auto_tags": self.has_auto_tags,
                "has_templates": self.has_templates,
                "dependencies": list(self.dependencies),
                "complexity_score": self.complexity_score,
            },
            "runtime_state": {
                "resolved_expression": self.resolved_expression,
                "last_evaluation": self.last_evaluation,
                "total_evaluations": self.total_evaluations,
            },
            "performance": {
                "avg_evaluation_time": self.avg_evaluation_time,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": self.cache_hits / max(self.total_evaluations, 1),
            },
            "recent_history": [
                {
                    "iteration": hist[0],
                    "resolved_expression": hist[1],
                    "result": hist[2],
                    "timestamp": hist[3]
                }
                for hist in self.evaluation_history[-5:]  # Last 5 evaluations
            ]
        }
    
    def should_terminate_loop(self, evaluation_result: bool) -> bool:
        """Determine if loop should terminate based on condition type and result."""
        if self.condition_type == "until":
            # Until condition: terminate when condition becomes TRUE
            return evaluation_result
        else:  # while
            # While condition: terminate when condition becomes FALSE
            return not evaluation_result
    
    def get_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for this condition with given context."""
        # Create deterministic key from condition and relevant context
        relevant_context = {k: v for k, v in context.items() if k in self.dependencies}
        context_str = str(sorted(relevant_context.items()))
        return f"{hash(self.expression)}_{hash(context_str)}"


@dataclass
class ConditionEvaluationResult:
    """Result of a condition evaluation with metadata."""
    
    condition: LoopCondition
    result: bool
    should_terminate: bool
    evaluation_time: float
    iteration: int
    resolved_expression: str
    cache_hit: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "condition_expression": self.condition.expression,
            "condition_type": self.condition.condition_type,
            "result": self.result,
            "should_terminate": self.should_terminate,
            "evaluation_time": self.evaluation_time,
            "iteration": self.iteration,
            "resolved_expression": self.resolved_expression,
            "cache_hit": self.cache_hit,
            "error": self.error,
        }


class ConditionParser:
    """Advanced parser for loop conditions with dependency analysis."""
    
    def __init__(self):
        self.auto_tag_pattern = re.compile(r'<AUTO>(.*?)</AUTO>', re.DOTALL)
        self.template_pattern = re.compile(r'\{\{\s*(.*?)\s*\}\}')
        self.comparison_ops = ['>=', '<=', '==', '!=', '>', '<']
        self.logical_ops = ['and', 'or', 'not']
        
        # Variable reference patterns
        self.variable_patterns = [
            r'\b([a-zA-Z_]\w*)\.[\w\[\]\.]+',  # object.property, object.method()
            r'\$([a-zA-Z_]\w*)',              # $variable (loop variables)
            r'\b([a-zA-Z_]\w*)\b',            # simple variables
        ]
    
    def parse(self, condition_str: str, condition_type: str = "until") -> LoopCondition:
        """Parse condition string into structured analysis."""
        condition = LoopCondition(
            expression=condition_str.strip(),
            condition_type=condition_type
        )
        
        # Analyze AUTO tags
        condition.has_auto_tags = bool(self.auto_tag_pattern.search(condition_str))
        
        # Analyze templates
        condition.has_templates = bool(self.template_pattern.search(condition_str))
        
        # Extract dependencies
        condition.dependencies = self._extract_dependencies(condition_str)
        
        # Calculate complexity
        condition.complexity_score = self._calculate_complexity(condition_str)
        
        logger.debug(f"Parsed {condition_type} condition: {condition.expression[:50]}{'...' if len(condition.expression) > 50 else ''}")
        logger.debug(f"Analysis: AUTO={condition.has_auto_tags}, templates={condition.has_templates}, "
                    f"deps={list(condition.dependencies)}, complexity={condition.complexity_score}")
        
        return condition
    
    def _extract_dependencies(self, expression: str) -> Set[str]:
        """Extract all variable dependencies from condition."""
        deps = set()
        
        # Extract from template variables: {{ variable.attribute }}
        template_matches = self.template_pattern.findall(expression)
        for match in template_matches:
            # Extract base variable name
            var_parts = match.split('.')[0].split('[')[0].strip()
            if var_parts and not var_parts.startswith('$'):
                deps.add(var_parts)
        
        # Extract from direct variable references
        for pattern in self.variable_patterns:
            matches = re.findall(pattern, expression)
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern with groups - take non-empty group
                    var_name = next((m for m in match if m), None)
                else:
                    var_name = match
                
                if var_name and var_name not in ['true', 'false', 'True', 'False', 'and', 'or', 'not']:
                    # Skip built-in keywords and operators
                    if not any(op in var_name for op in self.comparison_ops + self.logical_ops):
                        deps.add(var_name)
        
        # Extract loop variables: $item, $index, etc.
        loop_var_pattern = r'\$([a-zA-Z_]\w*)'
        loop_matches = re.findall(loop_var_pattern, expression)
        for var in loop_matches:
            deps.add(f"${var}")
        
        return deps
    
    def _calculate_complexity(self, expression: str) -> int:
        """Calculate expression complexity score."""
        score = 0
        
        # Logical operators add complexity
        score += expression.count('and') + expression.count('or')
        
        # Comparison operators
        score += sum(expression.count(op) for op in self.comparison_ops)
        
        # AUTO tags are complex
        score += len(self.auto_tag_pattern.findall(expression)) * 3
        
        # Template variables add some complexity
        score += len(self.template_pattern.findall(expression))
        
        # Parentheses indicate nested logic
        score += expression.count('(') + expression.count(')')
        
        # Function calls
        score += len(re.findall(r'\w+\s*\(', expression))
        
        return score
    
    def validate_condition(self, condition: LoopCondition) -> List[str]:
        """Validate condition for common issues."""
        issues = []
        
        # Check for balanced parentheses
        if condition.expression.count('(') != condition.expression.count(')'):
            issues.append("Unbalanced parentheses in condition")
        
        # Check for balanced template brackets
        if condition.expression.count('{{') != condition.expression.count('}}'):
            issues.append("Unbalanced template brackets {{ }} in condition")
        
        # Check for balanced AUTO tags
        if condition.expression.count('<AUTO>') != condition.expression.count('</AUTO>'):
            issues.append("Unbalanced AUTO tags in condition")
        
        # Check for potentially dangerous patterns
        dangerous_patterns = ['eval(', 'exec(', '__import__', 'open(']
        for pattern in dangerous_patterns:
            if pattern in condition.expression:
                issues.append(f"Potentially dangerous pattern detected: {pattern}")
        
        # Warn about very complex conditions
        if condition.complexity_score > 20:
            issues.append(f"Very complex condition (score: {condition.complexity_score}). Consider simplifying.")
        
        return issues


class ConditionCache:
    """Cache for condition evaluation results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[bool, float]] = {}  # key -> (result, timestamp)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[bool]:
        """Get cached result if valid."""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hits += 1
                return result
            else:
                # Expired - remove
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, result: bool) -> None:
        """Cache evaluation result."""
        # Clean up if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()
            
            # If still full, remove oldest entries
            if len(self.cache) >= self.max_size:
                sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
                for old_key, _ in sorted_items[:self.max_size // 4]:
                    del self.cache[old_key]
        
        self.cache[key] = (result, time.time())
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
        }