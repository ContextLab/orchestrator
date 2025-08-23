Validation API
==============

The orchestrator framework provides a comprehensive validation system to ensure pipeline integrity, catch configuration errors, and provide detailed feedback for debugging.

Overview
--------

The validation framework consists of multiple specialized validators:

* **TemplateValidator**: Validates Jinja2 templates and variable resolution
* **ToolValidator**: Validates tool configurations and availability
* **DependencyValidator**: Validates task dependencies and execution order
* **ModelValidator**: Validates model configurations and availability
* **OutputValidator**: Validates pipeline outputs and data flow
* **DataFlowValidator**: Validates data flow between pipeline steps
* **ValidationReport**: Unified reporting system for validation results

Core Validation Classes
----------------------

ValidationReport
~~~~~~~~~~~~~~~

.. automodule:: orchestrator.validation.validation_report
   :members:
   :undoc-members:
   :show-inheritance:

The ValidationReport system provides unified reporting across all validators with structured output formats.

Usage Example::

    from orchestrator.validation import ValidationReport, ValidationLevel, OutputFormat
    
    # Create report with specific validation level
    report = ValidationReport(
        validation_level=ValidationLevel.STRICT,
        output_format=OutputFormat.DETAILED
    )
    
    # Add issues from validators
    report.add_issue(ValidationIssue(
        severity=ValidationSeverity.ERROR,
        category="template",
        component="step1.action",
        message="Undefined variable 'user_input' in template",
        code="TEMPLATE_UNDEFINED_VAR",
        path="steps[0].action",
        suggestions=["Add 'user_input' to pipeline inputs", "Use default value: {{user_input | default('N/A')}}"]
    ))
    
    # Generate report
    if report.has_errors():
        print("Validation failed:")
        print(report.to_text())
    else:
        print("Validation passed!")

Classes and Enums
~~~~~~~~~~~~~~~~~

.. class:: ValidationLevel

   Validation strictness levels.

   .. attribute:: STRICT
   
      All validations must pass, no bypasses allowed.

   .. attribute:: PERMISSIVE
   
      Some validations can be warnings instead of errors.

   .. attribute:: DEVELOPMENT
   
      Maximum bypasses for development workflow.

.. class:: OutputFormat

   Validation report output formats.

   .. attribute:: TEXT
   
      Human-readable text format.

   .. attribute:: JSON
   
      Machine-readable JSON format.

   .. attribute:: DETAILED
   
      Detailed text with full context.

   .. attribute:: SUMMARY
   
      Brief summary format.

.. class:: ValidationSeverity

   Issue severity levels.

   .. attribute:: ERROR
   
      Blocking issues that prevent execution.

   .. attribute:: WARNING
   
      Non-blocking issues that should be addressed.

   .. attribute:: INFO
   
      Informational messages.

   .. attribute:: DEBUG
   
      Debug-level information.

.. class:: ValidationIssue

   Represents a single validation issue.

   :param severity: Issue severity level
   :type severity: ValidationSeverity
   :param category: Issue category (e.g., "template", "tool", "dependency")
   :type category: str
   :param component: Component identifier (e.g., task ID, parameter name)
   :type component: str
   :param message: Human-readable description
   :type message: str
   :param code: Machine-readable error code (optional)
   :type code: Optional[str]
   :param path: Context path (e.g., "steps[0].parameters.prompt")
   :type path: Optional[str]
   :param suggestions: List of suggested fixes
   :type suggestions: List[str]
   :param metadata: Additional metadata
   :type metadata: Dict[str, Any]

Specialized Validators
---------------------

TemplateValidator
~~~~~~~~~~~~~~~~

.. automodule:: orchestrator.validation.template_validator
   :members:
   :undoc-members:
   :show-inheritance:

Validates Jinja2 templates for syntax errors, undefined variables, and template resolution issues.

Usage Example::

    from orchestrator.validation import TemplateValidator
    
    validator = TemplateValidator()
    
    # Validate single template
    result = validator.validate_template(
        template="Hello {{name}}, your score is {{score}}",
        context={"name": "Alice"},  # Missing 'score' variable
        template_id="greeting"
    )
    
    if not result.is_valid:
        for issue in result.issues:
            print(f"Template error: {issue.message}")
    
    # Validate pipeline templates
    pipeline_result = validator.validate_pipeline(pipeline_dict)

ToolValidator
~~~~~~~~~~~~

.. automodule:: orchestrator.validation.tool_validator
   :members:
   :undoc-members:
   :show-inheritance:

Validates tool configurations, availability, and parameter schemas.

Usage Example::

    from orchestrator.validation import ToolValidator
    
    validator = ToolValidator()
    
    # Validate single tool configuration
    tool_config = {
        "tool": "web_search",
        "action": "search",
        "parameters": {
            "query": "{{search_term}}",
            "max_results": 10
        }
    }
    
    result = validator.validate_tool_config(tool_config)
    
    if not result.is_valid:
        for issue in result.issues:
            print(f"Tool validation error: {issue.message}")

DependencyValidator
~~~~~~~~~~~~~~~~~~

.. automodule:: orchestrator.validation.dependency_validator
   :members:
   :undoc-members:
   :show-inheritance:

Validates task dependencies, detects circular dependencies, and ensures execution order feasibility.

Usage Example::

    from orchestrator.validation import DependencyValidator
    
    validator = DependencyValidator()
    
    # Validate pipeline dependencies
    pipeline = {
        "steps": [
            {"id": "step1", "depends_on": []},
            {"id": "step2", "depends_on": ["step1"]},
            {"id": "step3", "depends_on": ["step1", "step2"]},
            {"id": "step4", "depends_on": ["step3", "step1"]}  # Complex but valid
        ]
    }
    
    result = validator.validate_dependencies(pipeline)
    
    if result.has_cycles:
        print("Circular dependencies detected:")
        for cycle in result.cycles:
            print(f"  {' -> '.join(cycle)}")

ModelValidator
~~~~~~~~~~~~~

.. automodule:: orchestrator.validation.model_validator
   :members:
   :undoc-members:
   :show-inheritance:

Validates model configurations, availability, and compatibility with pipeline requirements.

Usage Example::

    from orchestrator.validation import ModelValidator
    
    validator = ModelValidator()
    
    # Validate model configuration
    model_config = {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    result = validator.validate_model_config(model_config)
    
    if not result.is_valid:
        for issue in result.issues:
            print(f"Model validation error: {issue.message}")

OutputValidator
~~~~~~~~~~~~~~

.. automodule:: orchestrator.validation.output_validator
   :members:
   :undoc-members:
   :show-inheritance:

Validates pipeline outputs, data types, and format compliance.

Usage Example::

    from orchestrator.validation import OutputValidator, ValidationRule
    
    validator = OutputValidator()
    
    # Define validation rules
    rules = [
        ValidationRule(
            name="required_fields",
            rule_type="consistency",
            parameters={"required": ["title", "content"]}
        ),
        ValidationRule(
            name="content_format",
            rule_type="format",
            parameters={"format": "markdown"}
        )
    ]
    
    # Validate output
    output_data = {
        "title": "Analysis Report",
        "content": "# Analysis\n\nResults show...",
        "metadata": {"timestamp": "2024-01-01"}
    }
    
    result = validator.validate_output(output_data, rules)

DataFlowValidator
~~~~~~~~~~~~~~~~

.. automodule:: orchestrator.validation.data_flow_validator
   :members:
   :undoc-members:
   :show-inheritance:

Validates data flow between pipeline steps, ensuring data types match and required data is available.

Usage Example::

    from orchestrator.validation import DataFlowValidator
    
    validator = DataFlowValidator()
    
    # Validate data flow in pipeline
    pipeline = {
        "steps": [
            {
                "id": "extract",
                "outputs": {"data": {"type": "list", "items": "dict"}}
            },
            {
                "id": "process", 
                "depends_on": ["extract"],
                "inputs": {"raw_data": "{{extract.data}}"},
                "expected_input_types": {"raw_data": "list"}
            }
        ]
    }
    
    result = validator.validate_data_flow(pipeline)

Validation Rules
---------------

The validation framework uses a flexible rule system for custom validations.

.. class:: ValidationRule

   Defines a validation rule for output validation.

   :param name: Rule name/identifier
   :type name: str
   :param rule_type: Rule type ("consistency", "format", "dependency", "filesystem")
   :type rule_type: str
   :param parameters: Rule-specific parameters
   :type parameters: Dict[str, Any]

Built-in Rule Types
~~~~~~~~~~~~~~~~~~

**ConsistencyValidationRule**: Validates data consistency and required fields.

Parameters:
- ``required``: List of required fields
- ``forbidden``: List of forbidden fields
- ``unique``: Fields that must have unique values

**FormatValidationRule**: Validates data formats and schemas.

Parameters:
- ``format``: Expected format ("json", "yaml", "markdown", "xml")
- ``schema``: JSON schema for validation
- ``encoding``: Expected text encoding

**DependencyValidationRule**: Validates cross-reference dependencies.

Parameters:
- ``references``: Expected reference fields
- ``circular_refs``: Whether circular references are allowed

**FileSystemValidationRule**: Validates file system outputs.

Parameters:
- ``required_files``: List of required output files
- ``file_patterns``: File name patterns to validate
- ``permissions``: Expected file permissions

Custom Validation Rules
~~~~~~~~~~~~~~~~~~~~~~

Create custom validation rules by extending the base ValidationRule::

    from orchestrator.validation import ValidationRule, ValidationResult
    
    class CustomFormatRule(ValidationRule):
        def __init__(self, expected_format: str):
            super().__init__(
                name="custom_format",
                rule_type="format",
                parameters={"format": expected_format}
            )
        
        def validate(self, data: Any, context: Dict[str, Any]) -> ValidationResult:
            # Custom validation logic
            if self.check_format(data):
                return ValidationResult(is_valid=True)
            else:
                return ValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="format",
                        component="data",
                        message=f"Data does not match expected format: {self.parameters['format']}"
                    )]
                )

Integration Example
------------------

Complete validation workflow for a pipeline::

    from orchestrator.validation import (
        ValidationReport, ValidationLevel, OutputFormat,
        TemplateValidator, ToolValidator, DependencyValidator,
        ModelValidator, OutputValidator
    )
    
    def validate_pipeline(pipeline_config: Dict[str, Any]) -> ValidationReport:
        """Comprehensive pipeline validation."""
        
        # Create validation report
        report = ValidationReport(
            validation_level=ValidationLevel.STRICT,
            output_format=OutputFormat.DETAILED
        )
        
        # Initialize validators
        template_validator = TemplateValidator()
        tool_validator = ToolValidator()
        dependency_validator = DependencyValidator()
        model_validator = ModelValidator()
        output_validator = OutputValidator()
        
        # Run template validation
        template_result = template_validator.validate_pipeline(pipeline_config)
        report.add_issues(template_result.issues)
        
        # Run tool validation
        tool_result = tool_validator.validate_pipeline(pipeline_config)
        report.add_issues(tool_result.issues)
        
        # Run dependency validation
        dep_result = dependency_validator.validate_dependencies(pipeline_config)
        report.add_issues(dep_result.issues)
        
        # Run model validation
        if "model" in pipeline_config:
            model_result = model_validator.validate_model_config(pipeline_config["model"])
            report.add_issues(model_result.issues)
        
        # Generate final report
        if report.has_errors():
            print("Pipeline validation failed:")
            print(report.to_text())
            return report
        
        print("Pipeline validation passed!")
        return report
    
    # Usage
    pipeline = {
        "name": "Data Processing Pipeline",
        "inputs": {"data_file": {"type": "string", "required": True}},
        "steps": [
            {
                "id": "load_data",
                "tool": "filesystem",
                "action": "read",
                "parameters": {"path": "{{data_file}}"}
            },
            {
                "id": "process_data",
                "depends_on": ["load_data"],
                "action": "Process the data: {{load_data.result}}"
            }
        ],
        "outputs": {
            "result": "{{process_data.result}}"
        }
    }
    
    validation_report = validate_pipeline(pipeline)
    
    if validation_report.is_valid:
        # Pipeline is ready for execution
        pass
    else:
        # Handle validation errors
        for issue in validation_report.get_errors():
            print(f"Error: {issue.message}")
            if issue.suggestions:
                print(f"Suggestions: {', '.join(issue.suggestions)}")

Validation Best Practices
-------------------------

1. **Validate Early**: Run validation during pipeline compilation, not execution
2. **Use Appropriate Levels**: Choose validation level based on environment (strict for production, permissive for development)
3. **Handle Warnings**: Address warnings even if they don't block execution
4. **Custom Rules**: Create domain-specific validation rules for specialized requirements
5. **Comprehensive Reports**: Use detailed output format for debugging, summary for monitoring
6. **Error Recovery**: Provide actionable suggestions in validation issues
7. **Performance**: Cache validation results for repeated pipeline executions