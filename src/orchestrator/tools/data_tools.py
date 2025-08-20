"""Data processing and validation tools."""

import csv
import io
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import Tool


class DataProcessingTool(Tool):
    """Tool for data processing operations."""

    def __init__(self):
        super().__init__(
            name="data-processing",
            description="Process and transform data in various formats",
        )
        self.add_parameter(
            "action", "string", "Action: 'convert', 'filter', 'aggregate', 'transform', 'profile', 'pivot'"
        )
        # Support both 'data' and 'input_data' parameter names
        self.add_parameter("data", "object", "Input data or file path", required=False)
        self.add_parameter("input_data", "object", "Input data or file path (alias for data)", required=False)
        
        # Format parameters
        self.add_parameter(
            "format",
            "string",
            "Data format: 'json', 'csv', 'yaml', 'parquet'",
            required=False,
            default="json",
        )
        self.add_parameter(
            "input_format",
            "string",
            "Input data format (alias for format)",
            required=False,
        )
        self.add_parameter(
            "output_format",
            "string",
            "Output data format",
            required=False,
        )
        
        # Operation parameters
        self.add_parameter(
            "operation", "object", "Operation details", required=False, default={}
        )
        self.add_parameter(
            "operations", "array", "List of operations for transform action", required=False
        )
        
        # Aggregation parameters
        self.add_parameter(
            "group_by", "array", "Fields to group by for aggregation", required=False
        )
        self.add_parameter(
            "aggregations", "object", "Aggregation specifications", required=False
        )
        
        # Pivot parameters
        self.add_parameter(
            "index", "array", "Index columns for pivot", required=False
        )
        self.add_parameter(
            "columns", "array", "Column fields for pivot", required=False
        )
        self.add_parameter(
            "values", "array", "Value fields for pivot", required=False
        )
        self.add_parameter(
            "aggfunc", "string", "Aggregation function for pivot", required=False, default="sum"
        )
        self.add_parameter(
            "fill_value", "any", "Fill value for missing pivot combinations", required=False, default=0
        )
        
        # Profile parameters
        self.add_parameter(
            "profiling_options", "array", "Profiling options", required=False
        )
        
        # Other parameters
        self.add_parameter(
            "compression", "string", "Compression type for parquet", required=False
        )

    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute data processing operation."""
        action = kwargs.get("action", "")
        
        # Support both 'data' and 'input_data' parameter names
        data = kwargs.get("data") or kwargs.get("input_data", {})
        
        # Support format parameter aliases
        format = kwargs.get("input_format") or kwargs.get("format", "json")
        output_format = kwargs.get("output_format", format)
        
        # Get operation-specific parameters
        operation = kwargs.get("operation", {})
        operations = kwargs.get("operations", [])
        
        # Merge operations into operation dict for backward compatibility
        if operations and not operation:
            operation = {"operations": operations}

        try:
            if action == "convert":
                # For convert action, use output_format if specified
                target_format = output_format or format
                return await self._convert_data(data, target_format, operation)
            elif action == "filter":
                return await self._filter_data(data, operation, format)
            elif action == "aggregate":
                # Pass additional parameters for enhanced aggregation
                operation["group_by"] = kwargs.get("group_by", operation.get("group_by", []))
                operation["aggregations"] = kwargs.get("aggregations", operation.get("aggregations", {}))
                return await self._aggregate_data(data, operation, format, output_format)
            elif action == "transform":
                return await self._transform_data(data, operation, format, output_format)
            elif action == "profile":
                profiling_options = kwargs.get("profiling_options", [])
                return await self._profile_data(data, format, profiling_options)
            elif action == "pivot":
                pivot_params = {
                    "index": kwargs.get("index", []),
                    "columns": kwargs.get("columns", []),
                    "values": kwargs.get("values", []),
                    "aggfunc": kwargs.get("aggfunc", "sum"),
                    "fill_value": kwargs.get("fill_value", 0)
                }
                return await self._pivot_data(data, format, output_format, pivot_params)
            else:
                raise ValueError(f"Unknown data processing action: {action}")

        except Exception as e:
            return {"action": action, "success": False, "error": str(e)}

    async def _convert_data(
        self, data: Any, target_format: str, operation: Dict
    ) -> Dict[str, Any]:
        """Convert data between formats."""
        # Handle file paths (check length first to avoid "File name too long" error)
        if isinstance(data, str) and len(data) < 255 and Path(data).exists():
            with open(data, "r") as f:
                if data.endswith(".json"):
                    data = json.load(f)
                elif data.endswith(".csv"):
                    reader = csv.DictReader(f)
                    data = list(reader)
                else:
                    data = f.read()
        # Try to parse JSON string
        elif isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass

        # Convert to target format
        if target_format == "json":
            result = json.dumps(data, indent=2) if not isinstance(data, str) else data
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

    async def _aggregate_data(self, data: Any, operation: Dict, input_format: str = "json", output_format: str = "json") -> Dict[str, Any]:
        """Aggregate data with enhanced grouping support."""
        # Parse CSV if needed
        if isinstance(data, str) and input_format == "csv":
            reader = csv.DictReader(io.StringIO(data))
            data = list(reader)
        # Parse JSON string if needed
        elif isinstance(data, str) and input_format == "json":
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return {"action": "aggregate", "success": False, "error": f"Could not parse JSON data"}
        
        group_by = operation.get("group_by", [])
        aggregations = operation.get("aggregations", {})

        if not isinstance(data, list):
            raise ValueError("Aggregation requires list data")

        # If no group_by specified, do simple aggregation
        if not group_by:
            result = {"count": len(data), "aggregations": {}}
            
            for field, agg_spec in aggregations.items():
                # Handle both string (legacy) and dict (new) aggregation specs
                if isinstance(agg_spec, str):
                    agg_type = agg_spec
                    column = field
                else:
                    agg_type = agg_spec.get("function", "sum")
                    column = agg_spec.get("column", field)
                
                values = [self._parse_numeric(item.get(column)) for item in data if column in item]
                numeric_values = [v for v in values if v is not None]
                
                result_key = field if isinstance(agg_spec, dict) else f"{field}_{agg_type}"
                
                if agg_type in ["sum", "total"] and numeric_values:
                    result["aggregations"][result_key] = sum(numeric_values)
                elif agg_type in ["avg", "mean", "average"] and numeric_values:
                    result["aggregations"][result_key] = sum(numeric_values) / len(numeric_values)
                elif agg_type == "count":
                    result["aggregations"][result_key] = len(values)
                elif agg_type == "count_distinct":
                    result["aggregations"][result_key] = len(set(str(v) for v in values if v is not None))
                elif agg_type == "min" and numeric_values:
                    result["aggregations"][result_key] = min(numeric_values)
                elif agg_type == "max" and numeric_values:
                    result["aggregations"][result_key] = max(numeric_values)
                elif agg_type == "median" and numeric_values:
                    result["aggregations"][result_key] = statistics.median(numeric_values)
                
            return {"action": "aggregate", "result": result, "success": True}
        
        # Group-by aggregation
        grouped_data = defaultdict(list)
        for item in data:
            # Create group key from specified fields
            if len(group_by) == 1:
                key = str(item.get(group_by[0], ""))
            else:
                key = tuple(str(item.get(field, "")) for field in group_by)
            grouped_data[key].append(item)
        
        # Aggregate each group
        result = []
        for group_key, group_items in grouped_data.items():
            group_result = {}
            
            # Add group-by fields to result
            if len(group_by) == 1:
                group_result[group_by[0]] = group_key
            else:
                for i, field in enumerate(group_by):
                    group_result[field] = group_key[i]
            
            # Calculate aggregations for this group
            for agg_name, agg_spec in aggregations.items():
                if isinstance(agg_spec, str):
                    # Legacy format
                    agg_type = agg_spec
                    column = agg_name
                else:
                    # New format from pipeline
                    agg_type = agg_spec.get("function", "sum")
                    column = agg_spec.get("column", agg_name)
                
                values = [self._parse_numeric(item.get(column)) for item in group_items if column in item]
                numeric_values = [v for v in values if v is not None]
                
                if agg_type in ["sum", "total"] and numeric_values:
                    group_result[agg_name] = sum(numeric_values)
                elif agg_type in ["avg", "mean", "average"] and numeric_values:
                    group_result[agg_name] = sum(numeric_values) / len(numeric_values)
                elif agg_type == "count":
                    group_result[agg_name] = len(group_items)
                elif agg_type == "count_distinct":
                    distinct_values = set(str(item.get(column, "")) for item in group_items if column in item)
                    group_result[agg_name] = len(distinct_values)
                elif agg_type == "min" and numeric_values:
                    group_result[agg_name] = min(numeric_values)
                elif agg_type == "max" and numeric_values:
                    group_result[agg_name] = max(numeric_values)
                elif agg_type == "median" and numeric_values:
                    group_result[agg_name] = statistics.median(numeric_values)
            
            result.append(group_result)
        
        # Format output
        if output_format == "csv" and result:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=result[0].keys())
            writer.writeheader()
            writer.writerows(result)
            formatted_result = output.getvalue()
        elif output_format == "json":
            formatted_result = result
        else:
            formatted_result = result
            
        return {"action": "aggregate", "result": formatted_result, "row_count": len(result), "success": True}

    async def _transform_data(self, data: Any, operation: Dict, input_format: str = "json", output_format: str = "json") -> Dict[str, Any]:
        """Transform data structure with enhanced operations support."""
        import json
        
        # Parse input based on format
        if isinstance(data, str):
            if input_format == "csv":
                reader = csv.DictReader(io.StringIO(data))
                data = list(reader)
            else:
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass
        
        # Support both 'transformations' and 'operations' parameters
        transformations = operation.get("transformations", operation.get("operations", []))
        
        # If operations came from the pipeline's 'operations' parameter directly
        if not transformations and "operations" in operation:
            transformations = operation["operations"]

        result = data
        
        # Handle operations from the pipeline format
        for transform in transformations:
            # Support both dict format and pipeline operation format
            if not isinstance(transform, dict):
                continue
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
            
            # Support pipeline-specific operations
            elif transform_type == "deduplicate":
                columns = transform.get("columns", [])
                keep = transform.get("keep", "first")
                if isinstance(result, list) and columns:
                    seen = set()
                    deduped = []
                    for item in result:
                        key = tuple(str(item.get(col, "")) for col in columns)
                        if key not in seen:
                            seen.add(key)
                            deduped.append(item)
                        elif keep == "last":
                            # Remove previous and add this one
                            deduped = [i for i in deduped if tuple(str(i.get(col, "")) for col in columns) != key]
                            deduped.append(item)
                    result = deduped
            
            elif transform_type == "cast":
                columns = transform.get("columns", {})
                if isinstance(result, list):
                    for item in result:
                        for col, dtype in columns.items():
                            if col in item:
                                try:
                                    if dtype == "integer":
                                        item[col] = int(float(item[col])) if item[col] else 0
                                    elif dtype == "float":
                                        item[col] = float(item[col]) if item[col] else 0.0
                                    elif dtype == "datetime":
                                        # Keep as string for now, could parse to datetime
                                        item[col] = str(item[col])
                                except (ValueError, TypeError):
                                    pass
            
            elif transform_type == "fill_missing":
                strategy = transform.get("strategy", {})
                if isinstance(result, list):
                    for item in result:
                        for col, fill_value in strategy.items():
                            if col not in item or item[col] is None or item[col] == "":
                                item[col] = fill_value
            
            elif transform_type == "calculate":
                expressions = transform.get("expressions", {})
                if isinstance(result, list):
                    for item in result:
                        for new_col, expr in expressions.items():
                            try:
                                # Simple expression evaluation (multiply only for now)
                                if "*" in expr:
                                    parts = expr.split("*")
                                    if len(parts) == 2:
                                        left = parts[0].strip()
                                        right = parts[1].strip()
                                        left_val = float(item.get(left, 0))
                                        right_val = float(item.get(right, 0))
                                        item[new_col] = left_val * right_val
                                elif "DATE_FORMAT" in expr:
                                    # Extract date formatting
                                    import re
                                    match = re.search(r"DATE_FORMAT\((\w+),\s*'([^']+)'\)", expr)
                                    if match:
                                        date_col = match.group(1)
                                        format_str = match.group(2)
                                        if date_col in item:
                                            # Simple date extraction - just get year-month for now
                                            date_str = str(item[date_col])
                                            if "-" in date_str:
                                                parts = date_str.split("-")
                                                if len(parts) >= 2:
                                                    item[new_col] = f"{parts[0]}-{parts[1]}"
                            except Exception:
                                pass
            
            elif transform_type == "filter" and "condition" in transform:
                # Handle condition-based filtering from pipeline
                condition = transform.get("condition", "")
                if isinstance(result, list):
                    filtered = []
                    for item in result:
                        # Simple condition evaluation
                        if "!=" in condition:
                            parts = condition.split("!=")
                            if len(parts) == 2:
                                field = parts[0].strip()
                                value = parts[1].strip().strip("'\"")
                                if item.get(field) != value:
                                    filtered.append(item)
                        elif "==" in condition:
                            parts = condition.split("==")
                            if len(parts) == 2:
                                field = parts[0].strip()
                                value = parts[1].strip().strip("'\"")
                                if item.get(field) == value:
                                    filtered.append(item)
                        else:
                            filtered.append(item)
                    result = filtered

        # Format output
        if output_format == "csv" and isinstance(result, list) and result:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=result[0].keys())
            writer.writeheader()
            writer.writerows(result)
            formatted_result = output.getvalue()
        elif output_format == "json":
            formatted_result = json.dumps(result, indent=2) if not isinstance(result, str) else result
        else:
            formatted_result = result

        # Add row_count for list results
        row_count = len(result) if isinstance(result, list) else None
        
        return_val = {
            "action": "transform",
            "transformations": transformations,
            "result": formatted_result,
            "success": True,
        }
        
        if row_count is not None:
            return_val["row_count"] = row_count
            
        return return_val

    async def _profile_data(self, data: Any, input_format: str, profiling_options: List[str]) -> Dict[str, Any]:
        """Profile data to analyze quality and statistics."""
        # Parse CSV if needed
        if isinstance(data, str) and input_format == "csv":
            reader = csv.DictReader(io.StringIO(data))
            data = list(reader)
        elif isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return {"action": "profile", "success": False, "error": "Could not parse data"}
        
        if not isinstance(data, list) or not data:
            return {"action": "profile", "success": False, "error": "Data must be a non-empty list"}
        
        profile_result = {
            "row_count": len(data),
            "column_count": len(data[0].keys()) if data else 0,
            "columns": {}
        }
        
        # Default to all profiling options if none specified
        if not profiling_options:
            profiling_options = ["missing_values", "data_types", "statistical_summary", "outlier_detection", "duplicate_detection"]
        
        # Analyze each column
        for column in data[0].keys():
            col_profile = {"name": column}
            values = [row.get(column) for row in data]
            
            # Missing values analysis
            if "missing_values" in profiling_options:
                missing_count = sum(1 for v in values if v is None or v == "" or v == "null")
                col_profile["missing_count"] = missing_count
                col_profile["missing_percentage"] = (missing_count / len(values)) * 100
            
            # Data type detection
            if "data_types" in profiling_options:
                numeric_values = []
                date_count = 0
                for v in values:
                    if v is not None and v != "":
                        # Try to parse as number
                        parsed = self._parse_numeric(v)
                        if parsed is not None:
                            numeric_values.append(parsed)
                        # Check for date patterns
                        if isinstance(v, str) and ("-" in v or "/" in v):
                            date_count += 1
                
                if len(numeric_values) > len(values) * 0.8:
                    col_profile["data_type"] = "numeric"
                elif date_count > len(values) * 0.8:
                    col_profile["data_type"] = "date"
                else:
                    col_profile["data_type"] = "string"
                
                # Statistical summary for numeric columns
                if "statistical_summary" in profiling_options and numeric_values:
                    col_profile["min"] = min(numeric_values)
                    col_profile["max"] = max(numeric_values)
                    col_profile["mean"] = sum(numeric_values) / len(numeric_values)
                    if len(numeric_values) > 1:
                        col_profile["median"] = statistics.median(numeric_values)
                        col_profile["std_dev"] = statistics.stdev(numeric_values)
                    
                    # Outlier detection using IQR method
                    if "outlier_detection" in profiling_options and len(numeric_values) > 3:
                        sorted_vals = sorted(numeric_values)
                        q1_idx = len(sorted_vals) // 4
                        q3_idx = 3 * len(sorted_vals) // 4
                        q1 = sorted_vals[q1_idx]
                        q3 = sorted_vals[q3_idx]
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = [v for v in numeric_values if v < lower_bound or v > upper_bound]
                        col_profile["outlier_count"] = len(outliers)
                        col_profile["outlier_percentage"] = (len(outliers) / len(numeric_values)) * 100
            
            # Value statistics for all types
            unique_values = set(str(v) for v in values if v is not None and v != "")
            col_profile["unique_count"] = len(unique_values)
            col_profile["unique_percentage"] = (len(unique_values) / len(values)) * 100 if values else 0
            
            profile_result["columns"][column] = col_profile
        
        # Duplicate detection
        if "duplicate_detection" in profiling_options:
            # Check for duplicate rows
            row_strings = []
            for row in data:
                row_str = "|".join(str(row.get(col, "")) for col in sorted(row.keys()))
                row_strings.append(row_str)
            
            unique_rows = set(row_strings)
            duplicate_count = len(row_strings) - len(unique_rows)
            profile_result["duplicate_rows"] = duplicate_count
            profile_result["duplicate_percentage"] = (duplicate_count / len(data)) * 100
        
        return {
            "action": "profile",
            "result": profile_result,
            "success": True
        }
    
    async def _pivot_data(self, data: Any, input_format: str, output_format: str, pivot_params: Dict) -> Dict[str, Any]:
        """Create pivot table from data."""
        # Parse CSV if needed
        if isinstance(data, str) and input_format == "csv":
            reader = csv.DictReader(io.StringIO(data))
            data = list(reader)
        # Parse JSON string if needed
        elif isinstance(data, str) and input_format == "json":
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return {"action": "pivot", "success": False, "error": "Could not parse JSON data"}
        
        if not isinstance(data, list):
            return {"action": "pivot", "success": False, "error": "Data must be a list"}
        
        index_cols = pivot_params.get("index", [])
        column_cols = pivot_params.get("columns", [])
        value_cols = pivot_params.get("values", [])
        aggfunc = pivot_params.get("aggfunc", "sum")
        fill_value = pivot_params.get("fill_value", 0)
        
        if not index_cols or not value_cols:
            return {"action": "pivot", "success": False, "error": "Index and values parameters are required"}
        
        # Build pivot structure
        pivot_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for row in data:
            # Get index key
            if len(index_cols) == 1:
                index_key = str(row.get(index_cols[0], ""))
            else:
                index_key = tuple(str(row.get(col, "")) for col in index_cols)
            
            # Get column key (if specified)
            if column_cols:
                if len(column_cols) == 1:
                    col_key = str(row.get(column_cols[0], ""))
                else:
                    col_key = tuple(str(row.get(col, "")) for col in column_cols)
            else:
                col_key = "value"
            
            # Collect values
            for value_col in value_cols:
                val = self._parse_numeric(row.get(value_col))
                if val is not None:
                    pivot_data[index_key][col_key][value_col].append(val)
        
        # Aggregate values and build result
        result = []
        all_column_keys = set()
        for index_key in pivot_data:
            all_column_keys.update(pivot_data[index_key].keys())
        
        for index_key, columns in pivot_data.items():
            row_result = {}
            
            # Add index columns
            if len(index_cols) == 1:
                row_result[index_cols[0]] = index_key
            else:
                for i, col_name in enumerate(index_cols):
                    row_result[col_name] = index_key[i]
            
            # Add pivoted values
            for col_key in all_column_keys:
                for value_col in value_cols:
                    # Create column name
                    if column_cols:
                        if len(value_cols) == 1:
                            result_col = str(col_key)
                        else:
                            result_col = f"{col_key}_{value_col}"
                    else:
                        result_col = value_col
                    
                    # Aggregate values
                    values = columns[col_key][value_col] if col_key in columns else []
                    
                    if values:
                        if aggfunc == "sum":
                            row_result[result_col] = sum(values)
                        elif aggfunc == "mean":
                            row_result[result_col] = sum(values) / len(values)
                        elif aggfunc == "count":
                            row_result[result_col] = len(values)
                        elif aggfunc == "min":
                            row_result[result_col] = min(values)
                        elif aggfunc == "max":
                            row_result[result_col] = max(values)
                    else:
                        row_result[result_col] = fill_value
            
            result.append(row_result)
        
        # Format output
        if output_format == "csv" and result:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=result[0].keys())
            writer.writeheader()
            writer.writerows(result)
            formatted_result = output.getvalue()
        elif output_format == "json":
            formatted_result = result
        else:
            formatted_result = result
        
        return {
            "action": "pivot",
            "result": formatted_result,
            "row_count": len(result),
            "success": True
        }
    
    def _parse_numeric(self, value: Any) -> Optional[float]:
        """Parse a value to numeric, returning None if not possible."""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                # Remove common formatting
                cleaned = value.replace(",", "").replace("$", "").strip()
                return float(cleaned)
            except (ValueError, AttributeError):
                return None
        return None

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
