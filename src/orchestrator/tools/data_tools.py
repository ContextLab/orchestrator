"""Data processing and validation tools."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict

from .base import Tool


class DataProcessingTool(Tool):
    """Tool for data processing operations."""

    def __init__(self):
        super().__init__(
            name="data-processing",
            description="Process and transform data in various formats",
        )
        self.add_parameter(
            "action", "string", "Action: 'convert', 'filter', 'aggregate', 'transform'"
        )
        self.add_parameter("data", "object", "Input data or file path")
        self.add_parameter(
            "format",
            "string",
            "Data format: 'json', 'csv', 'yaml'",
            required=False,
            default="json",
        )
        self.add_parameter(
            "operation", "object", "Operation details", required=False, default={}
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute data processing operation."""
        action = kwargs.get("action", "")
        data = kwargs.get("data", {})
        format = kwargs.get("format", "json")
        operation = kwargs.get("operation", {})

        try:
            if action == "convert":
                return await self._convert_data(data, format, operation)
            elif action == "filter":
                return await self._filter_data(data, operation, format)
            elif action == "aggregate":
                return await self._aggregate_data(data, operation)
            elif action == "transform":
                return await self._transform_data(data, operation)
            else:
                raise ValueError(f"Unknown data processing action: {action}")

        except Exception as e:
            return {"action": action, "success": False, "error": str(e)}

    async def _convert_data(
        self, data: Any, target_format: str, operation: Dict
    ) -> Dict[str, Any]:
        """Convert data between formats."""
        # Handle file paths
        if isinstance(data, str) and Path(data).exists():
            with open(data, "r") as f:
                if data.endswith(".json"):
                    data = json.load(f)
                elif data.endswith(".csv"):
                    reader = csv.DictReader(f)
                    data = list(reader)
                else:
                    data = f.read()

        # Convert to target format
        if target_format == "json":
            result = json.dumps(data, indent=2)
        elif target_format == "csv":
            if isinstance(data, list) and data:
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                result = output.getvalue()
            else:
                result = str(data)
        else:
            result = str(data)

        return {
            "action": "convert",
            "target_format": target_format,
            "result": result,
            "original_type": type(data).__name__,
            "success": True,
        }

    async def _filter_data(self, data: Any, operation: Dict, format: str = "json") -> Dict[str, Any]:
        """Filter data based on criteria."""
        criteria = operation.get("criteria", {})
        
        # If data is a string and looks like CSV, parse it first
        if isinstance(data, str) and format == "csv":
            import csv
            import io
            reader = csv.DictReader(io.StringIO(data))
            data = list(reader)

        if isinstance(data, list):
            filtered_data = []
            for item in data:
                if self._matches_criteria(item, criteria):
                    filtered_data.append(item)
        else:
            filtered_data = data if self._matches_criteria(data, criteria) else None

        # Convert back to CSV if input format was CSV
        result = filtered_data
        if isinstance(filtered_data, list) and filtered_data and format == 'csv':
            import io
            output = io.StringIO()
            if filtered_data:
                writer = csv.DictWriter(output, fieldnames=filtered_data[0].keys())
                writer.writeheader()
                writer.writerows(filtered_data)
                result = output.getvalue()
        
        return {
            "action": "filter",
            "criteria": criteria,
            "result": result,
            "original_count": len(data) if isinstance(data, list) else 1,
            "filtered_count": (
                len(filtered_data)
                if isinstance(filtered_data, list)
                else (1 if filtered_data else 0)
            ),
            "success": True,
        }

    async def _aggregate_data(self, data: Any, operation: Dict) -> Dict[str, Any]:
        """Aggregate data."""
        operation.get("group_by", [])
        aggregations = operation.get("aggregations", {})

        if not isinstance(data, list):
            raise ValueError("Aggregation requires list data")

        # Simple aggregation
        result = {"count": len(data), "aggregations": {}}

        for field, agg_type in aggregations.items():
            values = [item.get(field) for item in data if field in item]
            numeric_values = [v for v in values if isinstance(v, (int, float))]

            if agg_type == "sum" and numeric_values:
                result["aggregations"][f"{field}_sum"] = sum(numeric_values)
            elif agg_type == "avg" and numeric_values:
                result["aggregations"][f"{field}_avg"] = sum(numeric_values) / len(
                    numeric_values
                )
            elif agg_type == "count":
                result["aggregations"][f"{field}_count"] = len(values)

        return {"action": "aggregate", "result": result, "success": True}

    async def _transform_data(self, data: Any, operation: Dict) -> Dict[str, Any]:
        """Transform data structure."""
        import json
        
        # Parse JSON string if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass
        
        transformations = operation.get("transformations", [])

        result = data
        for transform in transformations:
            transform_type = transform.get("type", "")

            if transform_type == "filter":
                field = transform.get("field", "")
                value = transform.get("value")
                
                # Handle filtering on nested data
                if isinstance(result, dict) and "records" in result:
                    records = result["records"]
                    if isinstance(records, list):
                        filtered = [r for r in records if r.get(field) == value]
                        result = {"records": filtered}
                elif isinstance(result, list):
                    result = [r for r in result if r.get(field) == value]
                    
            elif transform_type == "aggregate":
                agg_op = transform.get("operation", "")
                field = transform.get("field", "")
                
                # Handle aggregation on nested data
                records = result
                if isinstance(result, dict) and "records" in result:
                    records = result["records"]
                    
                if isinstance(records, list) and agg_op == "sum":
                    total = sum(r.get(field, 0) for r in records if isinstance(r.get(field, 0), (int, float)))
                    # Include both the filtered records and the aggregation
                    result = {
                        "filtered_records": records,
                        "aggregation": {"operation": agg_op, "field": field, "result": total}
                    }
                    
            elif transform_type == "rename_fields":
                mapping = transform.get("mapping", {})
                if isinstance(result, dict):
                    result = {mapping.get(k, k): v for k, v in result.items()}
                elif isinstance(result, list):
                    result = [
                        {mapping.get(k, k): v for k, v in item.items()}
                        for item in result
                    ]

            elif transform_type == "add_field":
                field_name = transform.get("field", "")
                field_value = transform.get("value", "")
                if isinstance(result, dict):
                    result[field_name] = field_value
                elif isinstance(result, list):
                    for item in result:
                        item[field_name] = field_value

        return {
            "action": "transform",
            "transformations": transformations,
            "result": result,
            "success": True,
        }

    def _matches_criteria(self, item: Any, criteria: Dict) -> bool:
        """Check if item matches filter criteria."""
        if not isinstance(item, dict):
            return True

        for field, expected in criteria.items():
            if field not in item:
                return False
            if item[field] != expected:
                return False

        return True


# NOTE: ValidationTool has been moved to validation.py with comprehensive implementation
# The old basic implementation below is kept for reference but should not be used

# class ValidationTool(Tool):
#     """Tool for data validation."""
#
#     def __init__(self):
#         super().__init__(
#             name="validation", description="Validate data against schemas and rules"
#         )
#         self.add_parameter("data", "object", "Data to validate")
#         self.add_parameter("schema", "object", "Validation schema", required=False)
#         self.add_parameter(
#             "rules", "array", "Validation rules", required=False, default=[]
#         )

#     async def execute(self, **kwargs) -> Dict[str, Any]:
#         """Execute validation."""
#         data = kwargs.get("data", {})
#         schema = kwargs.get("schema", {})
#         rules = kwargs.get("rules", [])
#
#         validation_results = {"valid": True, "errors": [], "warnings": []}
#
#         # Schema validation
#         if schema:
#             schema_errors = self._validate_schema(data, schema)
#             validation_results["errors"].extend(schema_errors)
#             if schema_errors:
#                 validation_results["valid"] = False
#
#         # Rules validation
#         for rule in rules:
#             rule_result = self._validate_rule(data, rule)
#             if rule_result["severity"] == "error":
#                 validation_results["errors"].append(rule_result)
#                 validation_results["valid"] = False
#             elif rule_result["severity"] == "warning":
#                 validation_results["warnings"].append(rule_result)
#
#         return {
#             "action": "validation",
#             "result": validation_results,
#             "data_type": type(data).__name__,
#             "success": True,
#         }
#
#     def _validate_schema(self, data: Any, schema: Dict) -> List[Dict]:
#         """Validate data against schema."""
#         errors = []
#
#         required_fields = schema.get("required", [])
#         properties = schema.get("properties", {})
#
#         if isinstance(data, dict):
#             # Check required fields
#             for field in required_fields:
#                 if field not in data:
#                     errors.append(
#                         {
#                             "field": field,
#                             "message": f"Required field '{field}' is missing",
#                             "severity": "error",
#                         }
#                     )
#
#             # Check field types
#             for field, value in data.items():
#                 if field in properties:
#                     expected_type = properties[field].get("type", "")
#                     if expected_type == "string" and not isinstance(value, str):
#                         errors.append(
#                             {
#                                 "field": field,
#                                 "message": f"Field '{field}' should be string, got {type(value).__name__}",
#                                 "severity": "error",
#                             }
#                         )
#                     elif expected_type == "integer" and not isinstance(value, int):
#                         errors.append(
#                             {
#                                 "field": field,
#                                 "message": f"Field '{field}' should be integer, got {type(value).__name__}",
#                                 "severity": "error",
#                             }
#                         )
#
#         return errors
#
#     def _validate_rule(self, data: Any, rule: Dict) -> Dict:
#         """Validate data against a single rule."""
#         rule_type = rule.get("type", "")
#         severity = rule.get("severity", "warning")
#
#         if rule_type == "not_empty":
#             field = rule.get("field", "")
#             if isinstance(data, dict) and field in data:
#                 if not data[field] or (
#                     isinstance(data[field], str) and not data[field].strip()
#                 ):
#                     return {
#                         "rule": rule_type,
#                         "field": field,
#                         "message": f"Field '{field}' should not be empty",
#                         "severity": severity,
#                     }
#
#         elif rule_type == "min_length":
#             field = rule.get("field", "")
#             min_length = rule.get("value", 0)
#             if isinstance(data, dict) and field in data:
#                 if len(str(data[field])) < min_length:
#                     return {
#                         "rule": rule_type,
#                         "field": field,
#                         "message": f"Field '{field}' should be at least {min_length} characters",
#                         "severity": severity,
#                     }
#
#         # Rule passed
#         return {"rule": rule_type, "message": "Rule passed", "severity": "info"}
