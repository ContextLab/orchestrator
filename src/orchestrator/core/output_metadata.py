"""
Output metadata models and utilities for task output tracking.
Provides comprehensive output specification, tracking, and validation capabilities.
"""

from __future__ import annotations

import mimetypes
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class OutputMetadata:
    """Metadata specification for task outputs."""
    
    produces: Optional[str] = None              # Output type descriptor (e.g., "pdf", "markdown-file", "json-data")
    location: Optional[str] = None              # File/URL/path location (supports templates)
    format: Optional[str] = None                # MIME type or format (e.g., "application/pdf", "text/markdown")
    schema: Optional[Dict[str, Any]] = None     # JSON schema for structured outputs
    size_limit: Optional[int] = None            # Maximum file size in bytes
    validation_rules: Optional[List[str]] = None  # Custom validation rules
    description: Optional[str] = None           # Human-readable description of output
    tags: Optional[List[str]] = None            # Tags for categorizing outputs
    
    def __post_init__(self):
        """Validate output metadata after initialization."""
        if self.tags is None:
            self.tags = []
        if self.validation_rules is None:
            self.validation_rules = []
        
        # Validate size limit
        if self.size_limit is not None and self.size_limit <= 0:
            raise ValueError("size_limit must be positive")
        
        # Auto-infer format from location if not specified
        if self.location and not self.format:
            self.format = self._infer_format_from_location(self.location)
    
    def _infer_format_from_location(self, location: str) -> Optional[str]:
        """Infer MIME type from file location."""
        try:
            # Handle template variables by extracting the base path pattern
            clean_location = location.split('{{')[0].split('}}')[-1] if '{{' in location else location
            
            # Get file extension
            _, ext = os.path.splitext(clean_location)
            if ext:
                mime_type, _ = mimetypes.guess_type(f"file{ext}")
                return mime_type
        except Exception:
            pass
        return None
    
    def is_file_output(self) -> bool:
        """Check if this output represents a file."""
        return bool(self.location and (
            self.location.startswith('./') or 
            self.location.startswith('/') or
            '/' in self.location or
            '\\' in self.location
        ))
    
    def is_structured_output(self) -> bool:
        """Check if this output has a defined schema."""
        return bool(self.schema)
    
    def get_expected_extension(self) -> Optional[str]:
        """Get expected file extension based on format."""
        if not self.format:
            return None
        
        # Common MIME type to extension mappings
        format_to_ext = {
            'application/pdf': '.pdf',
            'text/markdown': '.md',
            'text/plain': '.txt',
            'application/json': '.json',
            'text/csv': '.csv',
            'text/html': '.html',
            'application/xml': '.xml',
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/gif': '.gif',
            'application/zip': '.zip',
            'application/vnd.ms-excel': '.xlsx',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
        }
        
        return format_to_ext.get(self.format)
    
    def validate_consistency(self) -> List[str]:
        """Validate consistency between produces, location, and format."""
        issues = []
        
        if self.produces and self.location:
            # Check if produces type matches location extension
            expected_ext = self.get_expected_extension()
            if expected_ext and self.location.endswith(expected_ext):
                pass  # Consistent
            elif 'pdf' in self.produces.lower() and not (self.location.endswith('.pdf') or '.pdf' in self.location):
                issues.append(f"produces type '{self.produces}' suggests PDF but location '{self.location}' doesn't indicate PDF file")
            elif 'json' in self.produces.lower() and not (self.location.endswith('.json') or '.json' in self.location):
                issues.append(f"produces type '{self.produces}' suggests JSON but location '{self.location}' doesn't indicate JSON file")
            elif 'markdown' in self.produces.lower() and not (self.location.endswith('.md') or '.md' in self.location):
                issues.append(f"produces type '{self.produces}' suggests Markdown but location '{self.location}' doesn't indicate Markdown file")
        
        return issues


@dataclass
class OutputInfo:
    """Information about an actual task output with tracking metadata."""
    
    task_id: str
    output_type: Optional[str] = None           # From produces field
    location: Optional[str] = None              # Resolved location (no templates)
    format: Optional[str] = None                # Detected/specified format
    result: Any = None                          # Actual result data
    file_size: Optional[int] = None             # File size in bytes (for file outputs)
    checksum: Optional[str] = None              # File checksum for integrity
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: Optional[datetime] = None      # Last modification time
    accessed_at: Optional[datetime] = None      # Last access time
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    validation_status: Optional[str] = None     # "valid", "invalid", "pending"
    validation_errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize output info after creation."""
        if self.accessed_at is None:
            self.accessed_at = self.created_at
    
    def mark_accessed(self) -> None:
        """Mark output as accessed."""
        self.accessed_at = datetime.now()
    
    def mark_modified(self) -> None:
        """Mark output as modified."""
        self.modified_at = datetime.now()
    
    def is_file_output(self) -> bool:
        """Check if this output represents a file."""
        return bool(self.location and os.path.exists(self.location))
    
    def get_file_stats(self) -> Optional[Dict[str, Any]]:
        """Get file system statistics for file outputs."""
        if not self.is_file_output():
            return None
        
        try:
            stat = os.stat(self.location)
            return {
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'accessed': datetime.fromtimestamp(stat.st_atime),
                'permissions': oct(stat.st_mode)
            }
        except OSError:
            return None
    
    def validate_against_metadata(self, output_metadata: OutputMetadata) -> bool:
        """Validate actual output against expected metadata."""
        self.validation_errors.clear()
        
        # Check size limit
        if output_metadata.size_limit and self.file_size:
            if self.file_size > output_metadata.size_limit:
                self.validation_errors.append(
                    f"File size {self.file_size} exceeds limit {output_metadata.size_limit}"
                )
        
        # Check format consistency
        if output_metadata.format and self.format:
            if self.format != output_metadata.format:
                self.validation_errors.append(
                    f"Expected format '{output_metadata.format}' but got '{self.format}'"
                )
        
        # Check file existence for file outputs
        if output_metadata.is_file_output() and self.location:
            if not os.path.exists(self.location):
                self.validation_errors.append(f"Expected file at '{self.location}' does not exist")
        
        # Validate against schema if provided
        if output_metadata.schema and isinstance(self.result, dict):
            try:
                import jsonschema
                jsonschema.validate(self.result, output_metadata.schema)
            except ImportError:
                self.validation_errors.append("jsonschema not available for schema validation")
            except Exception as e:
                self.validation_errors.append(f"Schema validation failed: {str(e)}")
        
        # Set validation status
        self.validation_status = "valid" if not self.validation_errors else "invalid"
        return self.validation_status == "valid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'task_id': self.task_id,
            'output_type': self.output_type,
            'location': self.location,
            'format': self.format,
            'result': self.result,
            'file_size': self.file_size,
            'checksum': self.checksum,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'modified_at': self.modified_at.isoformat() if self.modified_at else None,
            'accessed_at': self.accessed_at.isoformat() if self.accessed_at else None,
            'metadata': self.metadata,
            'validation_status': self.validation_status,
            'validation_errors': self.validation_errors
        }


@dataclass
class OutputReference:
    """Reference to an output from another task."""
    
    task_id: str
    field: Optional[str] = None                 # Specific field to reference (e.g., "location", "result.data")
    default_value: Any = None                   # Default if reference cannot be resolved
    
    def __post_init__(self):
        """Validate reference after initialization."""
        if not self.task_id:
            raise ValueError("task_id is required for output reference")
    
    def resolve(self, output_info: OutputInfo) -> Any:
        """Resolve reference against actual output info."""
        if not self.field:
            return output_info.result
        
        # Handle nested field access (e.g., "result.data", "location")
        parts = self.field.split('.')
        current = output_info
        
        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return self.default_value
            return current
        except (AttributeError, KeyError, TypeError):
            return self.default_value
    
    def to_template_string(self) -> str:
        """Convert to template string format."""
        if self.field:
            return f"{{{{{self.task_id}.{self.field}}}}}"
        else:
            return f"{{{{{self.task_id}.result}}}}"


class OutputFormatDetector:
    """Utility class for detecting and working with output formats."""
    
    # Comprehensive format mappings
    EXTENSION_TO_MIME = {
        '.pdf': 'application/pdf',
        '.md': 'text/markdown',
        '.markdown': 'text/markdown',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.csv': 'text/csv',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.xml': 'application/xml',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.zip': 'application/zip',
        '.tar': 'application/x-tar',
        '.gz': 'application/gzip',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    }
    
    PRODUCES_TO_MIME = {
        'pdf': 'application/pdf',
        'markdown-file': 'text/markdown',
        'json-data': 'application/json',
        'csv-data': 'text/csv',
        'html-file': 'text/html',
        'text-file': 'text/plain',
        'image-png': 'image/png',
        'image-jpeg': 'image/jpeg',
        'excel-file': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'word-document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    @classmethod
    def detect_from_location(cls, location: str) -> Optional[str]:
        """Detect MIME type from file location."""
        if not location:
            return None
        
        # Handle template variables
        clean_location = location
        if '{{' in location:
            # Extract the static parts to infer format
            parts = location.split('{{')
            if len(parts) > 1:
                # Use the last part after template resolution
                clean_location = parts[-1].split('}}')[-1]
        
        # Get extension
        _, ext = os.path.splitext(clean_location.lower())
        return cls.EXTENSION_TO_MIME.get(ext)
    
    @classmethod
    def detect_from_produces(cls, produces: str) -> Optional[str]:
        """Detect MIME type from produces field."""
        if not produces:
            return None
        
        produces_lower = produces.lower().replace(' ', '-').replace('_', '-')
        return cls.PRODUCES_TO_MIME.get(produces_lower)
    
    @classmethod
    def detect_from_content(cls, content: Any, location: Optional[str] = None) -> Optional[str]:
        """Detect format from actual content."""
        if isinstance(content, dict):
            return 'application/json'
        elif isinstance(content, str):
            # Check if it looks like specific formats
            content_lower = content.lower().strip()
            if content_lower.startswith('<!doctype html') or content_lower.startswith('<html'):
                return 'text/html'
            elif content_lower.startswith('{') and content_lower.endswith('}'):
                try:
                    import json
                    json.loads(content)
                    return 'application/json'
                except:
                    pass
            elif location:
                return cls.detect_from_location(location)
            else:
                return 'text/plain'
        elif isinstance(content, bytes):
            # Check for binary format signatures
            if content.startswith(b'%PDF'):
                return 'application/pdf'
            elif content.startswith(b'\x89PNG'):
                return 'image/png'
            elif content.startswith(b'\xff\xd8\xff'):
                return 'image/jpeg'
            elif location:
                return cls.detect_from_location(location)
        
        return None
    
    @classmethod
    def get_extension_for_mime(cls, mime_type: str) -> Optional[str]:
        """Get file extension for MIME type."""
        mime_to_ext = {v: k for k, v in cls.EXTENSION_TO_MIME.items()}
        return mime_to_ext.get(mime_type)
    
    @classmethod
    def validate_consistency(cls, produces: Optional[str], location: Optional[str], 
                           format: Optional[str]) -> List[str]:
        """Validate consistency between produces, location, and format."""
        issues = []
        
        # Detect formats from different sources
        format_from_produces = cls.detect_from_produces(produces) if produces else None
        format_from_location = cls.detect_from_location(location) if location else None
        
        # Check consistency
        formats = [f for f in [format, format_from_produces, format_from_location] if f]
        
        if len(set(formats)) > 1:
            issues.append(
                f"Inconsistent formats detected: explicit={format}, "
                f"from_produces={format_from_produces}, from_location={format_from_location}"
            )
        
        return issues


def create_output_metadata(produces: Optional[str] = None, 
                          location: Optional[str] = None,
                          format: Optional[str] = None,
                          **kwargs) -> OutputMetadata:
    """Convenience function to create OutputMetadata with validation."""
    # Auto-detect format if not provided
    if not format:
        format = (OutputFormatDetector.detect_from_produces(produces) or 
                 OutputFormatDetector.detect_from_location(location))
    
    return OutputMetadata(
        produces=produces,
        location=location,
        format=format,
        **kwargs
    )


def validate_output_specification(produces: Optional[str] = None,
                                location: Optional[str] = None,
                                format: Optional[str] = None) -> List[str]:
    """Validate output specification for consistency."""
    issues = []
    
    # Use format detector for consistency check
    detector_issues = OutputFormatDetector.validate_consistency(produces, location, format)
    issues.extend(detector_issues)
    
    # Check for template syntax in location
    if location and '{{' in location:
        # Basic template syntax validation
        open_count = location.count('{{')
        close_count = location.count('}}')
        if open_count != close_count:
            issues.append(f"Template syntax error in location: mismatched braces in '{location}'")
    
    return issues