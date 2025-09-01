"""
Rule execution engine for flexible validation processing.

This module provides the core execution engine that orchestrates validation rules,
manages execution context, and provides comprehensive validation results.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, Callable
from pathlib import Path

from .rules import (
    ValidationRule, QualityRule, RuleViolation, ValidationContext, 
    RuleRegistry, RuleSeverity, RuleCategory
)
from ...execution.state import ExecutionContext, ExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class RuleExecutionResult:
    """Result of executing a single rule."""
    rule_id: str
    rule_name: str
    success: bool
    violations: List[RuleViolation] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None


@dataclass  
class ValidationSession:
    """Represents a complete validation session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_rules: int = 0
    rules_executed: int = 0
    rules_failed: int = 0
    total_violations: int = 0
    violations_by_severity: Dict[RuleSeverity, int] = field(default_factory=lambda: {
        severity: 0 for severity in RuleSeverity
    })
    rule_results: List[RuleExecutionResult] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Get session duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000
    
    @property
    def success_rate(self) -> float:
        """Calculate rule success rate."""
        if self.rules_executed == 0:
            return 0.0
        return ((self.rules_executed - self.rules_failed) / self.rules_executed) * 100.0
    
    def add_result(self, result: RuleExecutionResult) -> None:
        """Add a rule execution result."""
        self.rule_results.append(result)
        self.rules_executed += 1
        
        if not result.success:
            self.rules_failed += 1
        
        # Count violations by severity
        for violation in result.violations:
            self.violations_by_severity[violation.severity] += 1
            self.total_violations += 1
    
    def finalize(self) -> None:
        """Mark session as complete."""
        self.end_time = time.time()


class RuleExecutionContext:
    """Context for rule execution with shared state and utilities."""
    
    def __init__(
        self,
        validation_context: ValidationContext,
        session: ValidationSession,
        rule_registry: RuleRegistry
    ):
        """Initialize rule execution context."""
        self.validation_context = validation_context
        self.session = session
        self.rule_registry = rule_registry
        self.shared_state: Dict[str, Any] = {}
        self.execution_metadata: Dict[str, Any] = {}
    
    def set_shared_state(self, key: str, value: Any) -> None:
        """Set shared state for rule communication."""
        self.shared_state[key] = value
    
    def get_shared_state(self, key: str, default: Any = None) -> Any:
        """Get shared state value."""
        return self.shared_state.get(key, default)
    
    def log_execution_metadata(self, rule_id: str, metadata: Dict[str, Any]) -> None:
        """Log execution metadata for a rule."""
        self.execution_metadata[rule_id] = metadata


class ValidationEngine:
    """
    Core validation engine for executing rules and managing validation sessions.
    
    Provides parallel rule execution, error handling, and comprehensive result
    reporting for pipeline output validation.
    """
    
    def __init__(
        self,
        rule_registry: Optional[RuleRegistry] = None,
        max_workers: int = 4,
        timeout_seconds: float = 60.0
    ):
        """
        Initialize validation engine.
        
        Args:
            rule_registry: Rule registry to use (creates default if None)
            max_workers: Maximum number of worker threads for parallel execution
            timeout_seconds: Timeout for individual rule execution
        """
        self.rule_registry = rule_registry or RuleRegistry()
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.execution_hooks: Dict[str, List[Callable]] = {
            'before_validation': [],
            'after_validation': [],
            'before_rule': [],
            'after_rule': [],
            'on_violation': []
        }
        
        logger.info(f"Initialized ValidationEngine with {len(self.rule_registry.rules)} rules")
    
    def add_hook(self, hook_type: str, callback: Callable) -> None:
        """Add execution hook callback."""
        if hook_type in self.execution_hooks:
            self.execution_hooks[hook_type].append(callback)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
    
    def validate_output(
        self,
        execution_context: ExecutionContext,
        output_path: str,
        output_content: Optional[Any] = None,
        rule_filters: Optional[List[str]] = None,
        category_filters: Optional[List[RuleCategory]] = None,
        parallel: bool = True
    ) -> ValidationSession:
        """
        Validate output using configured rules.
        
        Args:
            execution_context: Current execution context
            output_path: Path to output file/directory to validate
            output_content: Optional pre-loaded content
            rule_filters: Optional list of rule IDs to execute
            category_filters: Optional list of categories to filter by
            parallel: Whether to execute rules in parallel
            
        Returns:
            ValidationSession with complete results
        """
        session_id = f"validation_{execution_context.execution_id}_{int(time.time())}"
        session = ValidationSession(
            session_id=session_id,
            start_time=time.time()
        )
        
        # Create validation context
        validation_context = ValidationContext(
            execution_context=execution_context,
            output_path=output_path,
            output_content=output_content,
            output_metadata={
                "session_id": session_id,
                "pipeline_id": execution_context.pipeline_id,
                "step_id": execution_context.current_step_id
            }
        )
        
        # Create rule execution context
        rule_execution_context = RuleExecutionContext(
            validation_context=validation_context,
            session=session,
            rule_registry=self.rule_registry
        )
        
        # Execute hooks
        self._execute_hooks('before_validation', rule_execution_context)
        
        try:
            # Select rules to execute
            rules_to_execute = self._select_rules(rule_filters, category_filters)
            session.total_rules = len(rules_to_execute)
            
            logger.info(f"Starting validation session {session_id} with {len(rules_to_execute)} rules")
            
            # Execute rules
            if parallel and len(rules_to_execute) > 1:
                self._execute_rules_parallel(rules_to_execute, rule_execution_context)
            else:
                self._execute_rules_sequential(rules_to_execute, rule_execution_context)
            
        except Exception as e:
            logger.error(f"Validation session {session_id} failed: {e}")
            logger.error(traceback.format_exc())
        finally:
            session.finalize()
            self._execute_hooks('after_validation', rule_execution_context)
            
            logger.info(
                f"Completed validation session {session_id}: "
                f"{session.rules_executed}/{session.total_rules} rules, "
                f"{session.total_violations} violations, "
                f"{session.duration_ms:.2f}ms"
            )
        
        return session
    
    def _select_rules(
        self,
        rule_filters: Optional[List[str]] = None,
        category_filters: Optional[List[RuleCategory]] = None
    ) -> List[ValidationRule]:
        """Select rules based on filters."""
        enabled_rules = self.rule_registry.get_enabled_rules()
        
        # Apply rule ID filters
        if rule_filters:
            enabled_rules = [rule for rule in enabled_rules if rule.rule_id in rule_filters]
        
        # Apply category filters  
        if category_filters:
            enabled_rules = [rule for rule in enabled_rules if rule.category in category_filters]
        
        return enabled_rules
    
    def _execute_rules_sequential(
        self,
        rules: List[ValidationRule],
        context: RuleExecutionContext
    ) -> None:
        """Execute rules sequentially."""
        for rule in rules:
            result = self._execute_single_rule(rule, context)
            context.session.add_result(result)
    
    def _execute_rules_parallel(
        self,
        rules: List[ValidationRule],
        context: RuleExecutionContext
    ) -> None:
        """Execute rules in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all rule executions
            future_to_rule = {
                executor.submit(self._execute_single_rule, rule, context): rule
                for rule in rules
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_rule, timeout=self.timeout_seconds * len(rules)):
                try:
                    result = future.result()
                    context.session.add_result(result)
                except Exception as e:
                    rule = future_to_rule[future]
                    error_result = RuleExecutionResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        success=False,
                        error_message=f"Rule execution failed: {e}"
                    )
                    context.session.add_result(error_result)
                    logger.error(f"Rule {rule.rule_id} failed: {e}")
    
    def _execute_single_rule(
        self,
        rule: ValidationRule,
        context: RuleExecutionContext
    ) -> RuleExecutionResult:
        """Execute a single validation rule."""
        start_time = time.time()
        result = RuleExecutionResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            success=True
        )
        
        try:
            # Execute hooks
            self._execute_hooks('before_rule', context, rule=rule)
            
            # Check if rule is applicable
            if not rule.is_applicable(context.validation_context):
                logger.debug(f"Rule {rule.rule_id} not applicable, skipping")
                return result
            
            # Execute rule validation
            violations = rule.validate(context.validation_context)
            result.violations = violations
            
            # Execute violation hooks
            for violation in violations:
                self._execute_hooks('on_violation', context, violation=violation)
            
            logger.debug(f"Rule {rule.rule_id} executed: {len(violations)} violations")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Rule {rule.rule_id} execution failed: {e}")
            logger.error(traceback.format_exc())
        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000
            self._execute_hooks('after_rule', context, rule=rule, result=result)
        
        return result
    
    def _execute_hooks(self, hook_type: str, context: RuleExecutionContext, **kwargs) -> None:
        """Execute registered hooks."""
        for callback in self.execution_hooks.get(hook_type, []):
            try:
                callback(context, **kwargs)
            except Exception as e:
                logger.error(f"Hook {hook_type} failed: {e}")
    
    def get_validation_summary(self, session: ValidationSession) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        violations_by_category = {}
        violations_by_rule = {}
        
        for result in session.rule_results:
            violations_by_rule[result.rule_id] = len(result.violations)
            
            for violation in result.violations:
                category = violation.category.value
                if category not in violations_by_category:
                    violations_by_category[category] = 0
                violations_by_category[category] += 1
        
        return {
            "session_id": session.session_id,
            "duration_ms": session.duration_ms,
            "rules_executed": session.rules_executed,
            "rules_failed": session.rules_failed, 
            "success_rate": session.success_rate,
            "total_violations": session.total_violations,
            "violations_by_severity": {
                severity.value: count 
                for severity, count in session.violations_by_severity.items()
            },
            "violations_by_category": violations_by_category,
            "violations_by_rule": violations_by_rule,
            "quality_score": self._calculate_quality_score(session),
            "recommendations": self._generate_recommendations(session)
        }
    
    def _calculate_quality_score(self, session: ValidationSession) -> float:
        """Calculate overall quality score (0-100)."""
        if session.total_violations == 0:
            return 100.0
        
        # Weight violations by severity
        severity_weights = {
            RuleSeverity.CRITICAL: 10,
            RuleSeverity.ERROR: 5,
            RuleSeverity.WARNING: 2,
            RuleSeverity.INFO: 1
        }
        
        weighted_score = 0
        max_possible_score = 100
        
        for severity, count in session.violations_by_severity.items():
            weight = severity_weights.get(severity, 1)
            weighted_score += count * weight
        
        # Calculate score (higher violations = lower score)
        quality_score = max(0, max_possible_score - weighted_score)
        return min(100.0, quality_score)
    
    def _generate_recommendations(self, session: ValidationSession) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for high violation counts
        if session.total_violations > 10:
            recommendations.append("Consider reviewing content quality standards - high violation count detected")
        
        # Check for critical violations
        critical_count = session.violations_by_severity.get(RuleSeverity.CRITICAL, 0)
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical violations immediately")
        
        # Check for failed rules
        if session.rules_failed > 0:
            recommendations.append(f"Investigate {session.rules_failed} failed validation rules")
        
        # Check for performance issues
        slow_rules = [r for r in session.rule_results if r.execution_time_ms > 5000]  # > 5 seconds
        if slow_rules:
            recommendations.append(f"Optimize performance for {len(slow_rules)} slow validation rules")
        
        return recommendations
    
    def export_results(
        self,
        session: ValidationSession,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """Export validation results to file."""
        import json
        
        output_path = Path(output_path)
        summary = self.get_validation_summary(session)
        
        # Add detailed results
        detailed_results = {
            "summary": summary,
            "rule_results": [
                {
                    "rule_id": result.rule_id,
                    "rule_name": result.rule_name,
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                    "error_message": result.error_message,
                    "violations": [
                        {
                            "rule_id": v.rule_id,
                            "message": v.message,
                            "severity": v.severity.value,
                            "category": v.category.value,
                            "file_path": v.file_path,
                            "line_number": v.line_number,
                            "column_number": v.column_number,
                            "metadata": v.metadata
                        }
                        for v in result.violations
                    ]
                }
                for result in session.rule_results
            ]
        }
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported validation results to {output_path}")