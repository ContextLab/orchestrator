"""Enhanced condition evaluator with structured evaluation and performance tracking."""

import ast
import time
import re
from typing import Any, Dict, Optional, Union
import logging

from .condition_models import LoopCondition, ConditionEvaluationResult, ConditionCache, ConditionParser
from .auto_resolver import ControlFlowAutoResolver

logger = logging.getLogger(__name__)


class EnhancedConditionEvaluator:
    """Advanced condition evaluator with caching, debugging, and performance tracking."""
    
    def __init__(self, auto_resolver: Optional[ControlFlowAutoResolver] = None):
        self.auto_resolver = auto_resolver
        self.cache = ConditionCache()
        self.parser = ConditionParser()
        
        # Safe evaluation environment
        self.safe_builtins = {
            'abs': abs, 'min': min, 'max': max, 'len': len,
            'str': str, 'int': int, 'float': float, 'bool': bool,
            'True': True, 'False': False, 'None': None,
            'sum': sum, 'round': round, 'sorted': sorted,
        }
        
        # Statistics
        self.total_evaluations = 0
        self.total_evaluation_time = 0.0
        self.error_count = 0
    
    async def evaluate_condition(
        self, 
        condition: Union[str, LoopCondition], 
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        iteration: int,
        condition_type: str = "until"
    ) -> ConditionEvaluationResult:
        """Evaluate condition with full metadata and performance tracking."""
        start_time = time.time()
        
        # Parse condition if string provided
        if isinstance(condition, str):
            condition = self.parser.parse(condition, condition_type)
        
        try:
            # Build evaluation context
            eval_context = self._build_evaluation_context(context, step_results, iteration)
            
            # Check cache first
            cache_key = condition.get_cache_key(eval_context)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                condition.cache_hits += 1
                evaluation_time = time.time() - start_time
                
                # Record evaluation in condition history
                condition.record_evaluation(
                    iteration=iteration,
                    resolved_expr=condition.resolved_expression or condition.expression,
                    result=cached_result,
                    eval_time=evaluation_time
                )
                
                return ConditionEvaluationResult(
                    condition=condition,
                    result=cached_result,
                    should_terminate=condition.should_terminate_loop(cached_result),
                    evaluation_time=evaluation_time,
                    iteration=iteration,
                    resolved_expression=condition.resolved_expression or condition.expression,
                    cache_hit=True
                )
            
            # Resolve templates first
            resolved = condition.expression
            if condition.has_templates:
                resolved = self._resolve_templates(resolved, eval_context)
            
            # Also resolve $ variables even without template brackets (for AST compatibility)
            resolved = self._resolve_dollar_variables(resolved, eval_context)
            
            # Resolve AUTO tags
            if condition.has_auto_tags and self.auto_resolver:
                resolved = await self._resolve_auto_tags(resolved, eval_context)
            
            # Evaluate boolean expression
            result = self._safe_evaluate_expression(resolved, eval_context)
            
            # Cache result
            self.cache.set(cache_key, result)
            
            # Calculate performance metrics
            evaluation_time = time.time() - start_time
            self.total_evaluations += 1
            self.total_evaluation_time += evaluation_time
            
            # Record evaluation in condition history
            condition.record_evaluation(
                iteration=iteration,
                resolved_expr=resolved,
                result=result,
                eval_time=evaluation_time
            )
            
            return ConditionEvaluationResult(
                condition=condition,
                result=result,
                should_terminate=condition.should_terminate_loop(result),
                evaluation_time=evaluation_time,
                iteration=iteration,
                resolved_expression=resolved,
                cache_hit=False
            )
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            self.error_count += 1
            
            logger.error(f"Condition evaluation failed: {e}, condition: {condition.expression}")
            
            # Return safe fallback
            fallback_result = self._get_fallback_result(condition.condition_type)
            
            return ConditionEvaluationResult(
                condition=condition,
                result=fallback_result,
                should_terminate=condition.should_terminate_loop(fallback_result),
                evaluation_time=evaluation_time,
                iteration=iteration,
                resolved_expression=condition.expression,
                cache_hit=False,
                error=str(e)
            )
    
    def _build_evaluation_context(
        self, 
        context: Dict[str, Any], 
        step_results: Dict[str, Any], 
        iteration: int
    ) -> Dict[str, Any]:
        """Build comprehensive evaluation context."""
        eval_context = context.copy()
        eval_context.update(step_results)
        
        # Add iteration metadata
        eval_context.update({
            "$iteration": iteration,
            "$index": iteration,
            "iteration": iteration,
            "step_results": step_results,
            "results": step_results,
        })
        
        # Add safe functions
        eval_context.update(self.safe_builtins)
        
        return eval_context
    
    def _resolve_templates(self, template: str, context: Dict[str, Any]) -> str:
        """Simple but robust template variable replacement."""
        def replace_var(match):
            var_expr = match.group(1).strip()
            
            try:
                # Handle dot notation (e.g., object.property)
                if '.' in var_expr:
                    parts = var_expr.split('.')
                    value = context
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        elif hasattr(value, part):
                            value = getattr(value, part)
                        else:
                            return match.group(0)  # Return original if not found
                    return str(value)
                
                # Check for $ variables first (like $iteration)
                if var_expr.startswith('$'):
                    if var_expr in context:
                        return str(context[var_expr])
                    # Also try without the $ prefix for fallback
                    bare_var = var_expr[1:]
                    if bare_var in context:
                        return str(context[bare_var])
                
                # Simple variable lookup
                if var_expr in context:
                    return str(context[var_expr])
                
                # Return original if not found
                return match.group(0)
                
            except Exception as e:
                logger.debug(f"Template resolution failed for {var_expr}: {e}")
                return match.group(0)
        
        # Replace {{ variable }} patterns
        template_pattern = r'\{\{\s*([^}]+)\s*\}\}'
        return re.sub(template_pattern, replace_var, template)
    
    def _resolve_dollar_variables(self, expression: str, context: Dict[str, Any]) -> str:
        """Resolve $variable references for AST compatibility."""
        def replace_dollar_var(match):
            var_name = match.group(1)
            full_var = f"${var_name}"
            
            try:
                if full_var in context:
                    return str(context[full_var])
                elif var_name in context:
                    return str(context[var_name])
                else:
                    # Return a safe Python variable name if not found
                    return f"__{var_name}__"
            except Exception as e:
                logger.debug(f"Dollar variable resolution failed for ${var_name}: {e}")
                return f"__{var_name}__"
        
        # Replace $variable patterns with their values
        dollar_pattern = r'\$([a-zA-Z_]\w*)'
        resolved = re.sub(dollar_pattern, replace_dollar_var, expression)
        
        # Also add any referenced variables to context with safe names
        dollar_matches = re.findall(r'\$([a-zA-Z_]\w*)', expression)
        for var_name in dollar_matches:
            full_var = f"${var_name}"
            if full_var in context:
                # Add as safe Python variable name
                context[f"__{var_name}__"] = context[full_var]
            elif var_name in context:
                context[f"__{var_name}__"] = context[var_name]
        
        return resolved
    
    async def _resolve_auto_tags(self, content: str, context: Dict[str, Any]) -> str:
        """Resolve AUTO tags using the auto resolver."""
        if not self.auto_resolver:
            # Fallback to default resolution
            return self._get_default_auto_resolution(content)
        
        try:
            # Use existing auto resolver functionality
            return await self.auto_resolver._resolve_auto_tags(content, context, {})
        except Exception as e:
            logger.warning(f"AUTO tag resolution failed: {e}")
            return self._get_default_auto_resolution(content)
    
    def _get_default_auto_resolution(self, content: str) -> str:
        """Provide default resolution when AUTO resolver fails."""
        content_lower = content.lower()
        
        # Boolean questions
        if any(word in content_lower for word in ['is', 'are', 'should', 'do', 'does']):
            if any(word in content_lower for word in ['not', 'false', 'no', 'never']):
                return "false"
            return "true"
        
        # Quality/threshold questions
        if any(word in content_lower for word in ['quality', 'score', 'rating']):
            if any(word in content_lower for word in ['high', 'good', 'excellent', '>= 0.8']):
                return "true"
            return "false"
        
        # Completion questions
        if any(word in content_lower for word in ['complete', 'finished', 'done', 'ready']):
            return "true"
        
        # Default to continue (false for until conditions)
        return "false"
    
    def _safe_evaluate_expression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate boolean expression using AST."""
        try:
            # Handle simple boolean strings first
            expr_lower = expression.strip().lower()
            if expr_lower in ['true', '1', 'yes', 'on']:
                return True
            elif expr_lower in ['false', '0', 'no', 'off', 'none', '']:
                return False
            
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')
            
            # Validate AST for safety
            self._validate_ast_safety(tree)
            
            # Compile and evaluate
            code = compile(tree, '<condition>', 'eval')
            result = eval(code, {"__builtins__": {}}, context)
            
            return bool(result)
            
        except Exception as e:
            logger.debug(f"Expression evaluation failed: {e}, expression: {expression}")
            # Try simple variable lookup as fallback
            return self._try_simple_evaluation(expression, context)
    
    def _validate_ast_safety(self, node):
        """Validate AST node to ensure it's safe to evaluate."""
        allowed_nodes = (
            ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp,
            ast.Compare, ast.Call, ast.Constant, ast.Name,
            ast.Load, ast.Attribute, ast.Subscript,
            ast.List, ast.Tuple, ast.Dict,
            # Operators
            ast.And, ast.Or, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
            ast.UAdd, ast.USub,
        )
        
        # For Python 3.8 compatibility
        if hasattr(ast, 'NameConstant'):
            allowed_nodes = allowed_nodes + (ast.NameConstant, ast.Num, ast.Str)
        
        for node in ast.walk(node):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsafe node type: {type(node).__name__}")
            
            # Check function calls - only allow safe ones
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self.safe_builtins:
                        raise ValueError(f"Unsafe function call: {node.func.id}")
                else:
                    raise ValueError(f"Complex function calls not allowed")
    
    def _try_simple_evaluation(self, expression: str, context: Dict[str, Any]) -> bool:
        """Try evaluating simple expressions as fallback."""
        expression = expression.strip().lower()
        
        # Direct boolean values
        if expression in ['true', '1', 'yes']:
            return True
        elif expression in ['false', '0', 'no']:
            return False
        
        # Simple variable lookup
        if expression in context:
            return bool(context[expression])
        
        # Default to False for safety (continue loop)
        return False
    
    def _get_fallback_result(self, condition_type: str) -> bool:
        """Get safe fallback result when evaluation fails."""
        if condition_type == "until":
            # For until conditions, return False (continue loop) on error
            return False
        else:  # while
            # For while conditions, return False (stop loop) on error
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this evaluator."""
        avg_time = (
            self.total_evaluation_time / self.total_evaluations 
            if self.total_evaluations > 0 else 0
        )
        
        return {
            "total_evaluations": self.total_evaluations,
            "total_time": self.total_evaluation_time,
            "avg_evaluation_time": avg_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_evaluations, 1),
            "cache_stats": self.cache.get_stats(),
        }
    
    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self.cache.clear()
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_evaluations = 0
        self.total_evaluation_time = 0.0
        self.error_count = 0
        self.cache.clear()