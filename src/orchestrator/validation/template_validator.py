"""Template validation for compile-time template checking.

This module provides comprehensive template validation to detect issues
before pipeline execution, including:
- Template syntax validation
- Variable reference checking against available context
- Undefined variable detection
- Clear error messages and suggestions

Issue #229: Compile-time template validation
"""

import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from jinja2 import TemplateSyntaxError, meta
from jinja2.sandbox import SandboxedEnvironment

from ..core.runtime_context import BARE_RUNTIME_NAMES, RUNTIME_NAMESPACE
from ..core.loop_contracts import ALL_BINDINGS, LoopContract, contracts_for
from ..core.step_fields import (
    INERT_PIPELINE_FIELDS,
    INERT_PROSE_STEP_FIELDS,
    NON_RENDERED_STRUCTURAL_STEP_FIELDS,
    OPERATIONAL_METADATA_KEYS,
)
from ..core.template_globals import (
    ALL_LOOP_VARIABLES,
    DOLLAR_LOOP_VARIABLES,
    find_global_misuse,
)
from ..core.template_sandbox import pipeline_global_names

logger = logging.getLogger(__name__)

#: The namespaces a pipeline's own parameters can be reached through.
#: `{{ topic }}`, `{{ parameters.topic }}` and `{{ inputs.topic }}` all render
#: the same value, so all three must validate. Deliberately does not include
#: `execution` -- the data-flow validator accepts that one, but the runtime
#: does not populate it, and papering over that here would hide a real bug.
PARAMETER_NAMESPACES = frozenset({"parameters", "inputs"})

#: Where a step sits in a pipeline document. A loop is declared by a step, so
#: this is where a loop construct is looked for -- see `_validate_object_templates`.
_STEP_PATH = re.compile(r"steps\[\d+\]$")

#: What a template in an inert field actually does. Validation used to report
#: `{{ b.result }}` in a step's `name:` as "references step results - will be
#: resolved at runtime", which is the opposite of true: nothing renders `name`,
#: so the braces reach the log verbatim. Worse, `{{ nosuch }}` in a
#: `description:` was a hard error, so a stray brace in prose rejected a
#: pipeline that runs correctly.
_INERT_TEMPLATE_MESSAGE = (
    "'{field}' is copied verbatim, so this template is never rendered -- "
    "the braces appear literally in the output"
)

#: How a field being unrendered should be reported. "Unrendered" alone does
#: not settle it: the first version of this treated every such field as prose
#: and warned, which said a `goto` sending execution to a step literally named
#: `{{ nosuch }}` was a wording problem.
_PROSE = "prose"
_STRUCTURAL = "structural"
_OPERATIONAL = "operational"

_INERT_ERROR_TYPE = {
    _STRUCTURAL: "template_in_structural_field",
    _OPERATIONAL: "template_in_operational_metadata",
}

_INERT_ERROR_MESSAGE = {
    _STRUCTURAL: (
        "'{field}' names a step, tool or dependency and is never rendered, so "
        "this template cannot resolve to the name it is standing in for"
    ),
    _OPERATIONAL: (
        "metadata '{field}' is read by the runtime and is never rendered, so "
        "the literal template text is what the runtime would act on"
    ),
}

_INERT_ERROR_SUGGESTION = {
    _STRUCTURAL: "Write the literal name here",
    _OPERATIONAL: "Write a literal value, or compute it in a rendered field",
}


def _classify(path: str, key: str, is_step: bool):
    """How a field's unrenderedness should be reported, or None if it renders."""
    if is_step:
        if key in INERT_PROSE_STEP_FIELDS:
            return (key, _PROSE)
        if key in NON_RENDERED_STRUCTURAL_STEP_FIELDS:
            return (key, _STRUCTURAL)
        if key == "metadata":
            return (key, _PROSE)
    if not path and key in INERT_PIPELINE_FIELDS:
        return (key, _PROSE)
    return None


def _binding_set(value: Union[bool, FrozenSet[str], None]) -> FrozenSet[str]:
    """Normalise the loop-scope argument to a set of names.

    `True` and `False` are still accepted because the parameter began life as
    a boolean. `True` means "inside some loop, construct unknown" and admits
    the union -- the imprecision this module is moving away from, kept only so
    an external caller that has not been updated does not start reporting
    every loop variable as undefined.
    """
    if value is True:
        return ALL_BINDINGS
    if not value:
        return frozenset()
    return frozenset(value)


@dataclass
class TemplateValidationError:
    """Represents a template validation error."""
    
    template: str
    error_type: str
    message: str
    context_path: Optional[str] = None
    suggestions: List[str] = None
    severity: str = "error"  # error, warning, info
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
    
    def __str__(self) -> str:
        """String representation of the validation error."""
        path_str = f" (at {self.context_path})" if self.context_path else ""
        result = f"{self.severity.upper()}{path_str}: {self.message}"
        if self.suggestions:
            result += f"\nSuggestions: {', '.join(self.suggestions)}"
        return result


@dataclass
class TemplateValidationResult:
    """Result of template validation."""
    
    is_valid: bool
    errors: List[TemplateValidationError]
    warnings: List[TemplateValidationError]
    available_variables: Set[str]
    used_variables: Set[str]
    undefined_variables: Set[str]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid and not self.has_warnings:
            return "Template validation passed"
        
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
            
        return f"Template validation: {', '.join(parts)}"


class TemplateValidator:
    """Validates templates at compile time to prevent runtime errors.
    
    This validator checks:
    1. Template syntax correctness
    2. Variable references against available context
    3. Loop variable usage patterns
    4. Filter and function usage
    5. Control structure syntax
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the template validator.
        
        Args:
            debug_mode: Enable debug logging
        """
        self.debug_mode = debug_mode
        
        # Use sandboxed environment for safety
        self.env = SandboxedEnvironment()
        self._register_custom_filters()
        
        # Patterns for different types of templates
        self.variable_pattern = re.compile(r'{{\s*([^}]+)\s*}}')
        self.control_pattern = re.compile(r'{%\s*([^%]+)\s*%}')
        self.comment_pattern = re.compile(r'{#\s*([^#]+)\s*#}')
        
        # Loop variable patterns
        # Both spellings. The runtime registers `item` and `$item` alike, and
        # knowing only the `$` form meant `{{ item.name }}` -- the form every
        # example actually uses -- was reported as an undefined variable
        # (#469). Declared in `core.template_globals` so the data-flow
        # validator and this one cannot drift apart about them.
        self.loop_vars = ALL_LOOP_VARIABLES
        self.step_result_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\.(result|results|output|outputs|content|data)')
        
        logger.info("TemplateValidator initialized")
    
    def validate_template(
        self,
        template: str,
        available_context: Optional[Dict[str, Any]] = None,
        context_path: Optional[str] = None,
        step_ids: Optional[List[str]] = None,
        loop_bindings: Union[bool, FrozenSet[str]] = frozenset(),
    ) -> TemplateValidationResult:
        """Validate a single template string.

        Args:
            template: Template string to validate
            available_context: Context variables available at compile time
            context_path: Path to this template (for error reporting)
            step_ids: List of step IDs in the pipeline
            loop_bindings: The names this template's loop binds. Empty means
                not inside a loop. `True` is accepted as "some loop, construct
                unknown" and admits the union of every construct's bindings --
                which cannot tell `{{ is_last }}` in a parallel queue from the
                same text in a `while` loop, so pass the real set where the
                construct is known.

        Returns:
            TemplateValidationResult with validation details
        """
        loop_bindings = _binding_set(loop_bindings)
        if available_context is None:
            available_context = {}
        if step_ids is None:
            step_ids = []
            
        errors = []
        warnings = []
        available_variables = set(available_context.keys())
        used_variables = set()
        undefined_variables = set()
        
        # Skip validation for empty or non-string templates
        if not isinstance(template, str) or not template.strip():
            return TemplateValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                available_variables=available_variables,
                used_variables=used_variables,
                undefined_variables=undefined_variables
            )
        
        if self.debug_mode:
            logger.debug(f"Validating template: {template[:100]}...")
            logger.debug(f"Available context: {list(available_context.keys())}")
        
        # 1. Check template syntax
        syntax_errors = self._validate_syntax(template, context_path)
        errors.extend(syntax_errors)
        
        # If syntax errors exist, can't proceed with further validation
        if syntax_errors:
            return TemplateValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                available_variables=available_variables,
                used_variables=used_variables,
                undefined_variables=undefined_variables
            )
        
        # 2. Extract and validate variable references
        var_results = self._validate_variables(
            template, available_context, context_path, step_ids, loop_bindings
        )
        errors.extend(var_results['errors'])
        warnings.extend(var_results['warnings'])
        used_variables.update(var_results['used_variables'])
        undefined_variables.update(var_results['undefined_variables'])
        
        # 3. Validate control structures
        control_results = self._validate_control_structures(template, context_path)
        errors.extend(control_results['errors'])
        warnings.extend(control_results['warnings'])
        
        # 4. Validate filters and functions
        filter_results = self._validate_filters(template, context_path)
        errors.extend(filter_results['errors'])
        warnings.extend(filter_results['warnings'])
        
        is_valid = len(errors) == 0
        
        if self.debug_mode:
            logger.debug(f"Validation result: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")
        
        return TemplateValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            available_variables=available_variables,
            used_variables=used_variables,
            undefined_variables=undefined_variables
        )
    
    def validate_pipeline_templates(
        self,
        pipeline_def: Dict[str, Any],
        compile_context: Optional[Dict[str, Any]] = None
    ) -> TemplateValidationResult:
        """Validate all templates in a pipeline definition.
        
        Args:
            pipeline_def: Pipeline definition dictionary
            compile_context: Context available at compile time
            
        Returns:
            Combined validation result for all templates
        """
        if compile_context is None:
            compile_context = {}
            
        # Collect all step IDs
        step_ids = []
        if "steps" in pipeline_def:
            for step in pipeline_def["steps"]:
                if isinstance(step, dict) and "id" in step:
                    step_ids.append(step["id"])
        
        # Add pipeline inputs and parameters to context
        full_context = compile_context.copy()
        
        # Add inputs and parameters.
        #
        # A declared name counts as available whether or not it has a default.
        # These entries are only ever tested for membership -- they answer "is
        # this name declared", not "what is its value" -- and a parameter
        # without a default is still perfectly well declared; its value simply
        # arrives at run time from `-i name=value`.
        #
        # Registering only the ones with defaults meant a pipeline declaring
        # `output_path` with no default was told `Undefined variable:
        # 'output_path'`, which is the name it just declared. That single case
        # accounted for the largest remaining cluster of false rejections.
        for section in ("inputs", "parameters"):
            for name, spec in (pipeline_def.get(section) or {}).items():
                if isinstance(spec, dict):
                    full_context[name] = spec.get("default")
                else:
                    full_context[name] = spec
        
        # Validate all templates in pipeline
        all_errors = []
        all_warnings = []
        all_used_variables = set()
        all_undefined_variables = set()
        
        self._validate_object_templates(
            pipeline_def, full_context, step_ids, "", 
            all_errors, all_warnings, all_used_variables, all_undefined_variables
        )
        
        is_valid = len(all_errors) == 0
        
        return TemplateValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            available_variables=set(full_context.keys()),
            used_variables=all_used_variables,
            undefined_variables=all_undefined_variables
        )
    
    def _validate_syntax(self, template: str, context_path: Optional[str]) -> List[TemplateValidationError]:
        """Validate Jinja2 template syntax."""
        errors = []
        
        try:
            # Try to parse the template
            self.env.parse(template)
        except TemplateSyntaxError as e:
            errors.append(TemplateValidationError(
                template=template,
                error_type="syntax_error",
                message=f"Template syntax error: {e.message}",
                context_path=context_path,
                suggestions=self._suggest_syntax_fixes(str(e))
            ))
        except Exception as e:
            errors.append(TemplateValidationError(
                template=template,
                error_type="parse_error",
                message=f"Template parsing failed: {str(e)}",
                context_path=context_path
            ))
        
        return errors
    
    def _validate_variables(
        self,
        template: str,
        available_context: Dict[str, Any],
        context_path: Optional[str],
        step_ids: List[str],
        loop_bindings: FrozenSet[str],
    ) -> Dict[str, Any]:
        """Validate variable references in template."""
        errors = []
        warnings = []
        used_variables = set()
        undefined_variables = set()
        
        try:
            # Parse template to get AST
            ast = self.env.parse(template)

            # Knowing the name is not knowing the use. `{{ now }}` names a
            # global correctly and still cannot work: it renders the function
            # object itself, so the artifact receives "<function ...>" and
            # nothing fails. The AST is what tells a call apart from an
            # attribute access or a bare mention; the text does not.
            for misuse in find_global_misuse(ast):
                reported = TemplateValidationError(
                    template=template,
                    error_type=misuse.code,
                    message=misuse.message,
                    context_path=context_path,
                    suggestions=[misuse.suggestion],
                    severity=misuse.severity,
                )
                # A deprecated global still works, so saying so must not stop
                # the pipeline; a misused one cannot work, so it must.
                if misuse.severity == "error":
                    errors.append(reported)
                else:
                    warnings.append(reported)

            # Find all variable references
            var_names = meta.find_undeclared_variables(ast)

            # `$item` is not a name Jinja can parse, so it never appears in
            # the AST and has to be matched as raw text. That substring scan
            # stays confined to the `$` spellings: applied to the bare names
            # it would match `item` inside `items` and `item_count`, which is
            # the text-matching class of bug #458 removed.
            loop_var_matches = [
                loop_var for loop_var in DOLLAR_LOOP_VARIABLES
                if loop_var in template
            ]
            
            # Combine both sets of variables.
            #
            # Sorted, because findings are emitted in this order and a set of
            # strings iterates by hash. Two identical `orchestrator validate`
            # runs on the same file produced the same 44 findings in different
            # orders, so any caller diffing runs, pinning output, or reporting
            # "the first problem" saw noise. `PYTHONHASHSEED` differs per
            # process, which is why the in-process check missed it.
            all_var_names = sorted(set(var_names) | set(loop_var_matches))

            for var_name in all_var_names:
                used_variables.add(var_name)
                
                # Check if it's a loop variable. A pipeline that declares an
                # input of its own called `item` means that input, so a
                # declared name wins -- otherwise adding these would reject
                # the pipeline that named its parameter after a loop word.
                if var_name in self.loop_vars and var_name not in available_context:
                    if var_name in loop_bindings:
                        continue
                    if loop_bindings:
                        # Inside a loop, but not one that binds this name.
                        # `while` has no `item`; only a parallel queue has
                        # `queue`. Accepting the union here is what let
                        # `{{ is_last }}` pass inside a `while` loop and then
                        # fail to render.
                        bound = ", ".join(
                            sorted(n for n in loop_bindings if not n.startswith("$"))
                        )
                        errors.append(TemplateValidationError(
                            template=template,
                            error_type="loop_variable_wrong_construct",
                            message=(
                                f"Loop variable '{var_name}' is not bound by this "
                                f"loop construct"
                            ),
                            context_path=context_path,
                            suggestions=[f"This loop binds: {bound}"],
                        ))
                        continue
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="loop_variable_outside_loop",
                        message=f"Loop variable '{var_name}' used outside of loop context",
                        context_path=context_path,
                        suggestions=["Move this template inside a for_each loop"]
                    ))
                    continue
                
                # Check if it's a step result reference
                if self._is_step_result_reference(var_name, step_ids):
                    # This is valid - step results are runtime variables
                    warnings.append(TemplateValidationError(
                        template=template,
                        error_type="runtime_variable",
                        message=f"Variable '{var_name}' references step results - will be resolved at runtime",
                        context_path=context_path,
                        severity="info"
                    ))
                    continue
                
                # A pipeline's own parameters can be named three ways --
                # `{{ topic }}`, `{{ parameters.topic }}`, `{{ inputs.topic }}`
                # -- and the runtime renders all three identically. Only the
                # bare form appears in `available_context`, so the other two
                # were reported undefined in pipelines that run correctly.
                if var_name in PARAMETER_NAMESPACES:
                    continue

                # `now()`, `file_exists()` and the loop helpers are part of the
                # language, but only the runtime can answer them, so they are
                # deliberately absent from this environment. Knowing the name
                # is what stops `{{ now() }}` -- which runs correctly -- from
                # being reported as an undefined variable.
                if var_name in pipeline_global_names():
                    continue

                # The run's own context. Populated by the runtime, so it is
                # not an undefined variable -- `{{ execution.timestamp }}` is
                # used by 32 catalogue pipelines, every one of which ran
                # correctly and failed validation. Which *fields* it offers is
                # checked by the data-flow validator, which sees the whole
                # dotted reference; this only sees the base name.
                if var_name == RUNTIME_NAMESPACE or var_name in BARE_RUNTIME_NAMES:
                    continue

                # Check if variable is available in context
                if var_name not in available_context:
                    undefined_variables.add(var_name)
                    
                    # Generate suggestions
                    suggestions = self._suggest_variable_alternatives(var_name, available_context, step_ids)
                    
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="undefined_variable",
                        message=f"Undefined variable: '{var_name}'",
                        context_path=context_path,
                        suggestions=suggestions
                    ))
        
        except Exception as e:
            errors.append(TemplateValidationError(
                template=template,
                error_type="variable_analysis_error",
                message=f"Failed to analyze variables: {str(e)}",
                context_path=context_path
            ))
        
        return {
            'errors': errors,
            'warnings': warnings,
            'used_variables': used_variables,
            'undefined_variables': undefined_variables
        }
    
    def _validate_control_structures(self, template: str, context_path: Optional[str]) -> Dict[str, List]:
        """Validate Jinja2 control structures."""
        errors = []
        warnings = []
        
        # Find all control structures
        controls = self.control_pattern.findall(template)
        
        if self.debug_mode:
            logger.debug(f"Found controls: {controls}")
        
        for control in controls:
            control = control.strip()
            
            # Check for common control structure issues
            if control.startswith('for '):
                # Validate for loop syntax
                if ' in ' not in control:
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="invalid_for_loop",
                        message=f"Invalid for loop syntax: '{control}'",
                        context_path=context_path,
                        suggestions=["Use format: 'for item in items'"]
                    ))
            
            elif control.startswith('if '):
                # Validate if condition syntax
                if len(control.split()) < 2:
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="invalid_if_statement",
                        message=f"Invalid if statement syntax: '{control}'",
                        context_path=context_path,
                        suggestions=["Provide a condition after 'if'"]
                    ))
            
            elif control.startswith('set '):
                # Validate set statement syntax
                if '=' not in control:
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="invalid_set_statement",
                        message=f"Invalid set statement syntax: '{control}'",
                        context_path=context_path,
                        suggestions=["Use format: 'set variable = value'"]
                    ))
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_filters(self, template: str, context_path: Optional[str]) -> Dict[str, List]:
        """Validate filter usage in template."""
        errors = []
        warnings = []
        
        # Find all variable expressions with filters
        var_matches = self.variable_pattern.findall(template)
        
        for var_expr in var_matches:
            if '|' in var_expr:
                # Extract filters
                parts = var_expr.split('|')
                for i, part in enumerate(parts[1:], 1):  # Skip variable name
                    # The filter name is the leading identifier, and nothing
                    # more. Splitting only on "(" treated everything after the
                    # name as part of it, so `{{ content | length > 10 }}` --
                    # which Jinja reads as `(content | length) > 10` -- was
                    # reported as an unknown filter called "length > 10".
                    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", part)
                    if not match:
                        continue
                    filter_name = match.group(1)
                    
                    # Check if filter exists
                    if filter_name not in self.env.filters:
                        errors.append(TemplateValidationError(
                            template=template,
                            error_type="unknown_filter",
                            message=f"Unknown filter: '{filter_name}'",
                            context_path=context_path,
                            suggestions=self._suggest_filter_alternatives(filter_name)
                        ))
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_object_templates(
        self,
        obj: Any,
        context: Dict[str, Any],
        step_ids: List[str],
        path: str,
        errors: List,
        warnings: List,
        used_variables: Set,
        undefined_variables: Set,
        loop_bindings: FrozenSet[str] = frozenset(),
        loop_scope: Optional[Tuple[LoopContract, str, FrozenSet[str]]] = None,
        inert_field: Optional[Tuple[str, str]] = None,
    ):
        """Recursively validate templates in an object.

        `loop_bindings` is what the enclosing loops bind. `loop_scope` is the
        innermost loop construct still being walked, paired with the field
        path reached inside it, because scope is a property of the field and
        not of the step: a `create_parallel_queue`'s `on` resolves before any
        item exists while the action list beside it runs per item.

        `inert_field` is `(field, kind)` for the unrendered step field this
        walk is inside, if any -- see `_classify`. It names the field rather
        than a flag so a nested value reports `metadata` instead of whichever
        key it sits under, and carries the kind because being unrendered is a
        warning in prose and an error in a field the runtime acts on.
        """
        if isinstance(obj, str):
            # Check if this contains templates
            if '{{' in obj or '{%' in obj:
                if inert_field:
                    field, kind = inert_field
                    if kind == _PROSE:
                        warnings.append(TemplateValidationError(
                            template=obj,
                            error_type="inert_field_template",
                            message=_INERT_TEMPLATE_MESSAGE.format(field=field),
                            context_path=path,
                            severity="warning",
                            suggestions=[
                                "Move the reference to a field that is rendered "
                                "(parameters, action, location), or remove the braces"
                            ],
                        ))
                        return
                    errors.append(TemplateValidationError(
                        template=obj,
                        error_type=_INERT_ERROR_TYPE[kind],
                        message=_INERT_ERROR_MESSAGE[kind].format(field=field),
                        context_path=path,
                        suggestions=[_INERT_ERROR_SUGGESTION[kind]],
                    ))
                    return
                result = self.validate_template(
                    obj, context, path, step_ids, loop_bindings
                )
                errors.extend(result.errors)
                warnings.extend(result.warnings)
                used_variables.update(result.used_variables)
                undefined_variables.update(result.undefined_variables)

        elif isinstance(obj, dict):
            # Only a *step* declares a loop. Matching any dict that happens to
            # hold a loop key made `create_parallel_queue`'s own nested
            # `action_loop` look like a second, separate loop, which replaced
            # the queue's scope with the action loop's and let `{{ item }}`
            # through in the `on` expression that generates the queue.
            # Nothing inside an inert field is a step, whatever it looks like.
            # `metadata` holds arbitrary author data, so a `metadata.steps`
            # list carrying `for_each` and `while` keys was being read as
            # pipeline structure and reported as an ambiguous loop -- inside a
            # subtree this module has just declared the runtime copies
            # verbatim.
            is_step = inert_field is None and bool(_STEP_PATH.search(path))
            declared = contracts_for(obj) if is_step else ()
            if len(declared) > 1:
                # Which construct wins is decided by declaration order in
                # `loop_contracts`, and no engine agreed to that order. The
                # step binds one set of names or another depending on an
                # implementation detail, so it has no meaning to validate.
                errors.append(TemplateValidationError(
                    template="",
                    error_type="ambiguous_loop_construct",
                    message=(
                        "Step declares more than one loop construct: "
                        + ", ".join(sorted(c.key for c in declared))
                    ),
                    context_path=path,
                    suggestions=["Split these into separate steps"],
                ))
            if len(declared) == 1:
                # Entering a loop. What the enclosing loops bind is kept
                # separately rather than subtracted back out later: an inner
                # `for_each` inside an outer one shares every name with it, so
                # subtracting the inner contract's names would take the outer
                # loop's `item` away from the inner iterable that is normally
                # written from exactly that.
                loop_scope = (declared[0], "", loop_bindings)
                loop_bindings = loop_bindings | declared[0].all_bindings()

            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                child_inert = inert_field or _classify(path, key, is_step)
                if (
                    inert_field is not None
                    and path.endswith(".metadata")
                    and key in OPERATIONAL_METADATA_KEYS
                ):
                    # A reserved key *inside* metadata. The object itself is
                    # arbitrary author data; these particular keys are read by
                    # control code, so an unrendered template in one is handed
                    # to it as a literal string.
                    child_inert = (key, _OPERATIONAL)
                child_bindings, child_scope = loop_bindings, loop_scope
                if loop_scope is not None:
                    contract, prefix, enclosing = loop_scope
                    relative = f"{prefix}.{key}" if prefix else key
                    child_scope = (contract, relative, enclosing)
                    child_bindings = enclosing | contract.bindings_for(relative)
                self._validate_object_templates(
                    value, context, step_ids, new_path,
                    errors, warnings, used_variables, undefined_variables,
                    child_bindings, child_scope, child_inert,
                )

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                self._validate_object_templates(
                    item, context, step_ids, new_path,
                    errors, warnings, used_variables, undefined_variables,
                    loop_bindings, loop_scope, inert_field,
                )
    
    def _is_step_result_reference(self, var_name: str, step_ids: List[str]) -> bool:
        """Check if a variable name references step results."""
        # Check direct step ID references
        if var_name in step_ids:
            return True
        
        # Check step.property references
        parts = var_name.split('.')
        if len(parts) >= 2 and parts[0] in step_ids:
            return True
        
        # Check for common result patterns
        if self.step_result_pattern.match(var_name):
            return True
        
        return False
    
    def _suggest_variable_alternatives(
        self,
        var_name: str,
        available_context: Dict[str, Any],
        step_ids: List[str]
    ) -> List[str]:
        """Suggest alternative variable names for undefined variables."""
        suggestions = []
        
        # Look for similar names in context
        for ctx_var in available_context.keys():
            if self._similar_strings(var_name, ctx_var):
                suggestions.append(f"Did you mean '{ctx_var}'?")
        
        # Look for similar step IDs
        for step_id in step_ids:
            if self._similar_strings(var_name, step_id):
                suggestions.append(f"Did you mean '{step_id}' (step result)?")
        
        # Common patterns
        if var_name.endswith('_text'):
            suggestions.append("Consider using 'text' or 'content'")
        elif var_name.endswith('_data'):
            suggestions.append("Consider using 'data' or 'result'")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _suggest_syntax_fixes(self, error_msg: str) -> List[str]:
        """Suggest fixes for syntax errors."""
        suggestions = []
        
        if 'unexpected' in error_msg.lower():
            suggestions.append("Check for unmatched brackets or quotes")
        
        if 'expected' in error_msg.lower():
            suggestions.append("Check template syntax - missing closing tags?")
        
        if 'filter' in error_msg.lower():
            suggestions.append("Check filter syntax: {{ variable | filter_name }}")
        
        return suggestions
    
    def _suggest_filter_alternatives(self, filter_name: str) -> List[str]:
        """Suggest alternative filter names."""
        suggestions = []
        
        # Common filter alternatives
        filter_alternatives = {
            'lower': ['lower'],
            'upper': ['upper'],
            'title': ['title'],
            'capitalize': ['capitalize'],
            'default': ['default'],
            'length': ['length', 'count'],
            'first': ['first'],
            'last': ['last'],
            'join': ['join'],
            'replace': ['replace'],
            'split': ['split'],
            'format': ['format']
        }
        
        for known_filter in self.env.filters.keys():
            if self._similar_strings(filter_name, known_filter):
                suggestions.append(f"Did you mean '{known_filter}'?")
        
        return suggestions[:3]
    
    def _similar_strings(self, s1: str, s2: str, threshold: float = 0.6) -> bool:
        """Check if two strings are similar using simple edit distance."""
        if len(s1) == 0 or len(s2) == 0:
            return False
        
        s1, s2 = s1.lower(), s2.lower()
        
        # Check for partial matches first
        if s1 in s2 or s2 in s1:
            return True
        
        # Simple character-by-character similarity
        max_len = max(len(s1), len(s2))
        min_len = min(len(s1), len(s2))
        
        if max_len == 0:
            return True
        
        # Count matching characters at same positions
        matches = sum(1 for i, (a, b) in enumerate(zip(s1, s2)) if a == b)
        
        # Also check if the strings have similar length and many shared characters
        shared_chars = len(set(s1) & set(s2))
        total_chars = len(set(s1) | set(s2))
        
        position_similarity = matches / min_len if min_len > 0 else 0
        char_similarity = shared_chars / total_chars if total_chars > 0 else 0
        
        return position_similarity >= threshold or char_similarity >= threshold
    
    def _register_custom_filters(self):
        """Take the runtime's filters verbatim.

        This used to define its own small set -- `default`, `length`, `json`,
        `lower`, `upper`, `replace` -- in parallel with the ones
        `TemplateManager` registers. The two drifted: the runtime grew to 70
        filters and this environment knew 56, so `{{ title | slugify }}` was
        reported as an unknown filter in a pipeline that renders it correctly.

        A validator that rejects working pipelines is worse than no validator,
        so there is one source of truth and this is not it.
        """
        from ..core.template_sandbox import create_pipeline_environment

        self.env = create_pipeline_environment()

        if self.debug_mode:
            logger.debug(f"Registered {len(self.env.filters)} template filters")
    
    def get_available_filters(self) -> List[str]:
        """Get list of available template filters."""
        return list(self.env.filters.keys())
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the validator state."""
        return {
            "debug_mode": self.debug_mode,
            "available_filters": len(self.env.filters),
            "filter_names": list(self.env.filters.keys())
        }