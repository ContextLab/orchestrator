"""Condition evaluation action handlers for control flow."""

import ast
import re
import operator
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import logging

from ..tools.base import Tool
from ..core.exceptions import TaskExecutionError


class ConditionEvaluationError(TaskExecutionError):
    """Raised when condition evaluation fails."""
    
    def __init__(self, message: str, condition: str = None):
        task_id = "evaluate_condition"
        super().__init__(task_id=task_id, reason=message)
        self.condition = condition


class ConditionEvaluator(Tool):
    """Base class for condition evaluation."""
    
    def __init__(self):
        super().__init__(
            name="evaluate_condition",
            description="Evaluate conditions for control flow"
        )
        self.add_parameter("condition", "string", "Condition to evaluate", required=True)
        self.add_parameter("context", "object", "Evaluation context", required=False)
        self.add_parameter("operator", "string", "Comparison operator", required=False)
        self.add_parameter("left", "any", "Left operand", required=False)
        self.add_parameter("right", "any", "Right operand", required=False)
        
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate the condition.
        
        Args:
            condition: Condition string to evaluate
            context: Context dict with variables for evaluation
            
        Returns:
            Boolean result of evaluation
        """
        pass
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute condition evaluation."""
        condition = kwargs.get("condition", "")
        context = kwargs.get("context", {})
        
        # Merge any additional parameters into context
        for key, value in kwargs.items():
            if key not in ["condition", "context"]:
                context[key] = value
        
        try:
            result = await self.evaluate(condition, context)
            self.logger.debug(f"Evaluated condition '{condition}' -> {result}")
            return {
                "result": result,
                "condition": condition,
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return {
                "result": False,
                "condition": condition,
                "status": "error",
                "error": str(e)
            }


class BooleanEvaluator(ConditionEvaluator):
    """Evaluate simple boolean conditions."""
    
    async def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate boolean conditions.
        
        Handles:
        - Simple strings: "true", "false", "yes", "no", etc.
        - Numbers: 0, 1
        - Python expressions that evaluate to bool
        """
        condition = condition.strip()
        condition_lower = condition.lower()
        
        # Check if it's a simple variable lookup first
        if condition in context:
            return bool(context[condition])
        
        # Handle simple boolean strings
        if condition_lower in ["true", "1", "yes", "on", "enabled"]:
            return True
        elif condition_lower in ["false", "0", "no", "off", "disabled", "none", ""]:
            return False
        
        # Try to evaluate as a simple Python expression
        try:
            # Use ast.literal_eval for safety (only allows literals)
            result = ast.literal_eval(condition)
            return bool(result)
        except (ValueError, SyntaxError):
            # As last resort, check if it's a simple expression
            try:
                # Create a safe evaluation environment
                safe_dict = {"__builtins__": {}}
                safe_dict.update(context)
                
                # Compile and evaluate
                code = compile(condition, "<condition>", "eval")
                result = eval(code, safe_dict)
                return bool(result)
            except Exception as e:
                raise ConditionEvaluationError(f"Cannot evaluate boolean condition '{condition}': {e}", condition)


class ComparisonEvaluator(ConditionEvaluator):
    """Evaluate comparison operations."""
    
    OPERATORS = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "in": lambda a, b: a in b,
        "not in": lambda a, b: a not in b,
        "is": operator.is_,
        "is not": operator.is_not,
    }
    
    # Regex to parse comparison expressions
    COMPARISON_PATTERN = re.compile(
        r'^\s*(.+?)\s*(==|!=|<=|>=|<|>|in|not\s+in|is\s+not|is)\s*(.+?)\s*$',
        re.IGNORECASE
    )
    
    async def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate comparison conditions."""
        # First check if operator and operands are provided as parameters
        if "operator" in context and "left" in context and "right" in context:
            op = context["operator"]
            left = context["left"]
            right = context["right"]
            
            if op in self.OPERATORS:
                try:
                    return self.OPERATORS[op](left, right)
                except Exception as e:
                    raise ConditionEvaluationError(f"Comparison failed: {e}", condition)
        
        # Otherwise, parse the condition string
        match = self.COMPARISON_PATTERN.match(condition)
        if not match:
            # Try to evaluate as boolean
            bool_evaluator = BooleanEvaluator()
            return await bool_evaluator.evaluate(condition, context)
        
        left_str, op, right_str = match.groups()
        op = op.lower()
        
        # Evaluate operands
        left = self._evaluate_operand(left_str.strip(), context)
        right = self._evaluate_operand(right_str.strip(), context)
        
        # Perform comparison
        if op in self.OPERATORS:
            try:
                return self.OPERATORS[op](left, right)
            except Exception as e:
                raise ConditionEvaluationError(f"Comparison '{left} {op} {right}' failed: {e}", condition)
        else:
            raise ConditionEvaluationError(f"Unknown operator: {op}", condition)
    
    def _evaluate_operand(self, operand: str, context: Dict[str, Any]) -> Any:
        """Evaluate a single operand."""
        # Remove quotes if it's a string literal
        if (operand.startswith('"') and operand.endswith('"')) or \
           (operand.startswith("'") and operand.endswith("'")):
            return operand[1:-1]
        
        # Handle special boolean literals
        if operand.lower() in ["true", "false"]:
            return operand.lower() == "true"
        
        # Try to parse as literal
        try:
            return ast.literal_eval(operand)
        except (ValueError, SyntaxError):
            pass
        
        # Check if it's a simple variable
        if operand in context:
            return context[operand]
        
        # Check for dot notation (e.g., user.name)
        if "." in operand:
            parts = operand.split(".")
            value = context
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    # Return the original string if we can't resolve it
                    return operand
            return value
        
        # Return as string if we can't evaluate it
        return operand


class LogicalEvaluator(ConditionEvaluator):
    """Evaluate logical operations (AND, OR, NOT)."""
    
    # Patterns for logical operations
    AND_PATTERN = re.compile(r'\s+and\s+', re.IGNORECASE)
    OR_PATTERN = re.compile(r'\s+or\s+', re.IGNORECASE)
    NOT_PATTERN = re.compile(r'^\s*not\s+', re.IGNORECASE)
    
    def _split_respecting_parentheses(self, text: str, pattern: re.Pattern) -> List[str]:
        """Split text by pattern but respect parentheses."""
        parts = []
        current_part = []
        paren_count = 0
        
        # Find all matches
        matches = list(pattern.finditer(text))
        if not matches:
            return [text]
        
        last_end = 0
        for match in matches:
            # Check if we're inside parentheses
            for i in range(last_end, match.start()):
                char = text[i]
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            
            if paren_count == 0:
                # We're not inside parentheses, so split here
                parts.append(text[last_end:match.start()])
                last_end = match.end()
        
        # Add the remaining part
        parts.append(text[last_end:])
        
        return [p for p in parts if p]  # Filter out empty parts
    
    async def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate logical conditions."""
        condition = condition.strip()
        
        # Handle OR (lower precedence than AND)
        or_parts = self._split_respecting_parentheses(condition, self.OR_PATTERN)
        if len(or_parts) > 1:
            for part in or_parts:
                # Each part might contain AND operations
                part_result = await self._evaluate_and_chain(part.strip(), context)
                if part_result:
                    return True
            return False
        
        # Handle AND
        return await self._evaluate_and_chain(condition, context)
    
    async def _evaluate_and_chain(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a chain of AND operations."""
        and_parts = self._split_respecting_parentheses(condition, self.AND_PATTERN)
        if len(and_parts) > 1:
            for part in and_parts:
                if not await self._evaluate_single(part.strip(), context):
                    return False
            return True
        
        # Single condition
        return await self._evaluate_single(condition, context)
    
    async def _evaluate_single(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition (no logical operators)."""
        # Remove parentheses if they wrap the entire condition
        condition = condition.strip()
        if condition.startswith('(') and condition.endswith(')'):
            # Check if parentheses are balanced
            count = 0
            for i, char in enumerate(condition):
                if char == '(':
                    count += 1
                elif char == ')':
                    count -= 1
                if count == 0 and i < len(condition) - 1:
                    # Parentheses don't wrap entire condition
                    break
            else:
                # Parentheses wrap entire condition
                condition = condition[1:-1].strip()
                # The content inside might have logical operators
                return await self.evaluate(condition, context)
        
        # Handle NOT operator
        if self.NOT_PATTERN.match(condition):
            inner_condition = self.NOT_PATTERN.sub('', condition, count=1)
            inner_result = await self._evaluate_single(inner_condition, context)
            return not inner_result
        
        # Try comparison evaluator
        comp_evaluator = ComparisonEvaluator()
        return await comp_evaluator.evaluate(condition, context)


class TemplateEvaluator(ConditionEvaluator):
    """Evaluate conditions with template variables."""
    
    def __init__(self, template_manager=None):
        super().__init__()
        self.template_manager = template_manager
        self._template_pattern = re.compile(r'\{\{.*?\}\}')
    
    async def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate conditions that may contain template variables."""
        # Check if condition contains templates
        if not self._template_pattern.search(condition):
            # No templates, use logical evaluator
            logical_evaluator = LogicalEvaluator()
            return await logical_evaluator.evaluate(condition, context)
        
        # Render templates if template manager is available
        if self.template_manager:
            try:
                rendered = self.template_manager.render(condition, additional_context=context)
                self.logger.debug(f"Rendered condition: '{condition}' -> '{rendered}'")
                condition = rendered
            except Exception as e:
                self.logger.warning(f"Failed to render template: {e}")
        else:
            # Simple template replacement without template manager
            condition = self._simple_template_replace(condition, context)
        
        # Evaluate the rendered condition
        logical_evaluator = LogicalEvaluator()
        return await logical_evaluator.evaluate(condition, context)
    
    def _simple_template_replace(self, template: str, context: Dict[str, Any]) -> str:
        """Simple template variable replacement."""
        def replace_var(match):
            var_expr = match.group(0)[2:-2].strip()  # Remove {{ }}
            
            # Handle dot notation
            if '.' in var_expr:
                parts = var_expr.split('.')
                value = context
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)  # Return original if not found
                return str(value)
            
            # Simple variable
            if var_expr in context:
                return str(context[var_expr])
            
            return match.group(0)  # Return original if not found
        
        return self._template_pattern.sub(replace_var, template)


class ExpressionEvaluator(ConditionEvaluator):
    """Evaluate complex expressions safely."""
    
    # Allowed names in expressions
    SAFE_NAMES = {
        'abs': abs,
        'len': len,
        'max': max,
        'min': min,
        'sum': sum,
        'round': round,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'True': True,
        'False': False,
        'None': None,
    }
    
    async def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate complex expressions safely."""
        try:
            # Parse the expression into an AST
            tree = ast.parse(condition, mode='eval')
            
            # Validate the AST to ensure it's safe
            self._validate_ast(tree)
            
            # Create evaluation namespace
            namespace = self.SAFE_NAMES.copy()
            namespace.update(context)
            
            # Compile and evaluate
            code = compile(tree, '<expression>', 'eval')
            result = eval(code, {"__builtins__": {}}, namespace)
            
            return bool(result)
        except ConditionEvaluationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Fall back to template evaluator
            template_evaluator = TemplateEvaluator()
            return await template_evaluator.evaluate(condition, context)
    
    def _validate_ast(self, node):
        """Validate AST node to ensure it's safe to evaluate."""
        allowed_nodes = (
            ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp,
            ast.Compare, ast.Call, ast.Constant, ast.Name,
            ast.Load, ast.Attribute, ast.Subscript, ast.Index,
            ast.List, ast.Tuple, ast.Dict, ast.Set,
            # Operators
            ast.And, ast.Or, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Is, ast.IsNot, ast.In, ast.NotIn,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.UAdd, ast.USub,
        )
        
        # For Python 3.8 compatibility
        if hasattr(ast, 'NameConstant'):
            allowed_nodes = allowed_nodes + (ast.NameConstant, ast.Num, ast.Str)
        
        for node in ast.walk(node):
            if not isinstance(node, allowed_nodes):
                raise ConditionEvaluationError(f"Unsafe node type: {type(node).__name__}", str(node))
            
            # Check function calls
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in self.SAFE_NAMES:
                    raise ConditionEvaluationError(f"Unsafe function call: {ast.dump(node.func)}", str(node))


# Factory function to get appropriate evaluator
def get_condition_evaluator(condition: str, context: Dict[str, Any] = None) -> ConditionEvaluator:
    """Get the appropriate evaluator for a condition."""
    condition = condition.strip()
    
    # Check for template variables
    if '{{' in condition and '}}' in condition:
        # Get template manager from context if available
        template_manager = context.get('_template_manager') if context else None
        return TemplateEvaluator(template_manager)
    
    # Check for function calls
    if re.search(r'\w+\s*\(', condition):
        return ExpressionEvaluator()
    
    # Check for mathematical operators (*, +, -, /, %, **)
    if any(op in condition for op in ['*', '+', '-', '/', '%', '**']) and \
       any(op in condition for op in ['==', '!=', '<=', '>=', '<', '>', ' in ', ' not in ']):
        # Has both math and comparison - needs expression evaluator
        return ExpressionEvaluator()
    
    # Check for logical operators (must come before comparison)
    if re.search(r'\b(and|or|not)\b', condition, re.IGNORECASE):
        return LogicalEvaluator()
    
    # Check for comparison operators
    if any(op in condition for op in ['==', '!=', '<=', '>=', '<', '>', ' in ', ' not in ']):
        return ComparisonEvaluator()
    
    # Check for complex expressions (parentheses, brackets, etc.)
    if any(char in condition for char in ['(', ')', '[', ']']):
        return ExpressionEvaluator()
    
    # Default to boolean evaluator
    return BooleanEvaluator()