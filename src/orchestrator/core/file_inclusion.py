"""
File inclusion system for external content in pipeline definitions.
Provides secure file inclusion with caching, recursive processing, and comprehensive security.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class FileIncludeDirective:
    """Configuration for a file inclusion directive."""
    
    syntax: Literal["template", "bracket"]  # {{ file:... }} or << ... >>
    path: str
    base_dir: Optional[str] = None
    encoding: str = "utf-8"
    required: bool = True
    recursive: bool = True  # Allow nested includes
    max_size: Optional[int] = None  # Maximum file size in bytes
    
    def __post_init__(self):
        """Validate directive configuration."""
        if not self.path:
            raise ValueError("File path cannot be empty")
        
        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("max_size must be positive")


@dataclass
class FileIncludeResult:
    """Result of a file inclusion operation."""
    
    content: str
    resolved_path: str
    size: int
    encoding: str
    cached: bool
    nested_includes: List[str] = field(default_factory=list)
    load_time: float = 0.0
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "resolved_path": self.resolved_path,
            "size": self.size,
            "encoding": self.encoding,
            "cached": self.cached,
            "nested_includes": self.nested_includes,
            "load_time": self.load_time,
            "cache_hit": self.cache_hit
        }


class FileInclusionError(Exception):
    """Base exception for file inclusion errors."""
    pass


class SecurityError(FileInclusionError):
    """Raised when file inclusion violates security policies."""
    pass


class FileNotFoundError(FileInclusionError):
    """Raised when included file is not found."""
    pass


class FileSizeError(FileInclusionError):
    """Raised when file exceeds size limits."""
    pass


class CircularInclusionError(FileInclusionError):
    """Raised when circular file inclusion is detected."""
    pass


class FileInclusionProcessor:
    """
    Secure file inclusion processor with caching and recursive support.
    
    Provides comprehensive file inclusion capabilities with security controls,
    performance optimization, and integration with template systems.
    """
    
    def __init__(
        self,
        base_dirs: Optional[List[str]] = None,
        cache_enabled: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB default
        max_inclusion_depth: int = 10,
        allowed_extensions: Optional[Set[str]] = None,
        cache_ttl: int = 3600  # 1 hour cache TTL
    ):
        """
        Initialize file inclusion processor.
        
        Args:
            base_dirs: Base directories for file resolution
            cache_enabled: Whether to enable file caching
            max_file_size: Maximum allowed file size in bytes
            max_inclusion_depth: Maximum recursive inclusion depth
            allowed_extensions: Set of allowed file extensions (None = all allowed)
            cache_ttl: Cache time-to-live in seconds
        """
        self.base_dirs = base_dirs or [".", "prompts", "templates", "includes", "content"]
        self.cache_enabled = cache_enabled
        self.max_file_size = max_file_size
        self.max_inclusion_depth = max_inclusion_depth
        self.allowed_extensions = allowed_extensions
        self.cache_ttl = cache_ttl
        
        # File cache: path -> (result, timestamp)
        self.file_cache: Dict[str, Tuple[FileIncludeResult, float]] = {}
        
        # Inclusion tracking for circular reference detection
        self.inclusion_stack: List[str] = []
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "files_loaded": 0,
            "total_bytes_loaded": 0,
            "security_violations": 0,
            "circular_references_detected": 0
        }
        
        # Regex patterns for file inclusion syntax
        self.template_pattern = re.compile(r'\{\{\s*file:\s*([^}]+)\s*\}\}', re.IGNORECASE)
        self.bracket_pattern = re.compile(r'<<\s*([^>]+)\s*>>', re.IGNORECASE)
    
    async def process_content(self, content: str, base_dir: Optional[str] = None) -> str:
        """
        Process all file inclusions in content.
        
        Args:
            content: Content to process
            base_dir: Base directory for relative path resolution
            
        Returns:
            Content with all file inclusions processed
            
        Raises:
            FileInclusionError: If inclusion processing fails
        """
        if not isinstance(content, str):
            return str(content)
        
        logger.debug(f"Processing content with {len(content)} characters")
        
        # Only reset inclusion stack if we're at the top level (empty stack)
        reset_stack = len(self.inclusion_stack) == 0
        
        try:
            # Process template syntax: {{ file:path }}
            content = await self._process_template_syntax(content, base_dir)
            
            # Process bracket syntax: << path >>
            content = await self._process_bracket_syntax(content, base_dir)
            
            logger.debug(f"Content processing complete, result: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error processing file inclusions: {e}")
            raise FileInclusionError(f"Content processing failed: {e}") from e
        finally:
            # Only clear stack if we reset it at the beginning
            if reset_stack:
                self.inclusion_stack.clear()
    
    async def include_file(self, directive: FileIncludeDirective) -> FileIncludeResult:
        """
        Load and process a single file inclusion.
        
        Args:
            directive: File inclusion directive
            
        Returns:
            File inclusion result
            
        Raises:
            FileInclusionError: If file inclusion fails
        """
        start_time = time.time()
        
        try:
            # Resolve file path
            resolved_path = self._resolve_path(directive)
            logger.debug(f"Resolved path: {directive.path} -> {resolved_path}")
            
            # Security validation
            if not self._validate_path_security(resolved_path):
                self.metrics["security_violations"] += 1
                raise SecurityError(f"Path security violation: {directive.path}")
            
            # Check cache first
            if self.cache_enabled:
                cached_result = self._get_cached_result(resolved_path)
                if cached_result:
                    self.metrics["cache_hits"] += 1
                    cached_result.cache_hit = True
                    cached_result.load_time = time.time() - start_time
                    logger.debug(f"Cache hit for: {resolved_path}")
                    return cached_result
            
            self.metrics["cache_misses"] += 1
            
            # Check for circular inclusion
            if resolved_path in self.inclusion_stack:
                self.metrics["circular_references_detected"] += 1
                circular_path = " -> ".join(self.inclusion_stack + [resolved_path])
                raise CircularInclusionError(f"Circular inclusion detected: {circular_path}")
            
            # Load file
            result = await self._load_file(resolved_path, directive)
            
            # Process nested inclusions if enabled
            if directive.recursive and self._has_inclusions(result.content):
                logger.debug(f"Processing nested inclusions in: {resolved_path}")
                
                # Add to inclusion stack
                self.inclusion_stack.append(resolved_path)
                
                try:
                    # Check inclusion depth
                    if len(self.inclusion_stack) > self.max_inclusion_depth:
                        raise FileInclusionError(
                            f"Maximum inclusion depth ({self.max_inclusion_depth}) exceeded"
                        )
                    
                    # Process nested inclusions
                    processed_content = await self.process_content(
                        result.content, 
                        os.path.dirname(resolved_path)
                    )
                    result.content = processed_content
                    result.nested_includes = self._extract_nested_includes(processed_content)
                    
                finally:
                    # Remove from inclusion stack only if we added it
                    if self.inclusion_stack and self.inclusion_stack[-1] == resolved_path:
                        self.inclusion_stack.pop()
            
            # Cache result
            if self.cache_enabled:
                self._cache_result(resolved_path, result)
            
            # Update metrics
            self.metrics["files_loaded"] += 1
            self.metrics["total_bytes_loaded"] += result.size
            
            result.load_time = time.time() - start_time
            logger.info(f"Successfully included file: {resolved_path} ({result.size} bytes)")
            
            return result
            
        except FileInclusionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error including file {directive.path}: {e}")
            raise FileInclusionError(f"Failed to include file {directive.path}: {e}") from e
    
    def _resolve_path(self, directive: FileIncludeDirective) -> str:
        """
        Resolve file path against base directories.
        
        Args:
            directive: File inclusion directive
            
        Returns:
            Resolved absolute path
            
        Raises:
            FileNotFoundError: If file cannot be found
        """
        path = directive.path.strip()
        
        # If path is absolute, use as-is (after security validation)
        if os.path.isabs(path):
            if os.path.exists(path):
                return os.path.abspath(path)
            else:
                raise FileNotFoundError(f"Absolute path not found: {path}")
        
        # Try each base directory
        search_dirs = []
        
        # Add directive-specific base directory if provided
        if directive.base_dir:
            search_dirs.append(directive.base_dir)
        
        # Add configured base directories
        search_dirs.extend(self.base_dirs)
        
        for base_dir in search_dirs:
            candidate_path = os.path.join(base_dir, path)
            abs_candidate = os.path.abspath(candidate_path)
            
            if os.path.exists(abs_candidate):
                logger.debug(f"Found file: {path} in base directory: {base_dir}")
                return abs_candidate
        
        # File not found in any base directory
        searched_paths = [os.path.join(base_dir, path) for base_dir in search_dirs]
        raise FileNotFoundError(
            f"File not found: {path}. Searched in: {searched_paths}"
        )
    
    def _validate_path_security(self, resolved_path: str) -> bool:
        """
        Validate file path for security compliance.
        
        Args:
            resolved_path: Resolved absolute file path
            
        Returns:
            True if path is secure, False otherwise
        """
        try:
            # Convert to Path object for easier manipulation
            path = Path(resolved_path).resolve()
            
            # Check if path exists
            if not path.exists():
                logger.warning(f"Path does not exist: {resolved_path}")
                return False
            
            # Check if it's a regular file
            if not path.is_file():
                logger.warning(f"Path is not a regular file: {resolved_path}")
                return False
            
            # Validate against base directories
            path_str = str(path)
            
            # Check if path is within allowed base directories
            allowed = False
            for base_dir in self.base_dirs:
                abs_base = Path(base_dir).resolve()
                try:
                    # Check if the file is within this base directory
                    path.relative_to(abs_base)
                    allowed = True
                    break
                except ValueError:
                    # Path is not relative to this base directory
                    continue
            
            if not allowed:
                logger.warning(f"Path outside allowed directories: {resolved_path}")
                return False
            
            # Check file extension if restrictions are configured
            if self.allowed_extensions:
                file_extension = path.suffix.lower()
                if file_extension not in self.allowed_extensions:
                    logger.warning(f"File extension not allowed: {file_extension}")
                    return False
            
            # Additional security checks
            
            # Check for suspicious path components
            suspicious_components = ['..', '.git', '.env', 'passwd', 'shadow']
            path_parts = path.parts
            
            for component in path_parts:
                if any(suspicious in component.lower() for suspicious in suspicious_components):
                    logger.warning(f"Suspicious path component detected: {component}")
                    return False
            
            # Check file size
            try:
                file_size = path.stat().st_size
                if file_size > self.max_file_size:
                    logger.warning(f"File too large: {file_size} bytes > {self.max_file_size}")
                    return False
            except OSError as e:
                logger.warning(f"Could not stat file: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security validation error for path {resolved_path}: {e}")
            return False
    
    async def _load_file(self, resolved_path: str, directive: FileIncludeDirective) -> FileIncludeResult:
        """
        Load file content from disk.
        
        Args:
            resolved_path: Resolved absolute file path
            directive: File inclusion directive
            
        Returns:
            File inclusion result
            
        Raises:
            FileInclusionError: If file loading fails
        """
        try:
            # Get file stats
            stat_result = os.stat(resolved_path)
            file_size = stat_result.st_size
            
            # Check size limits
            max_size = directive.max_size or self.max_file_size
            if file_size > max_size:
                raise FileSizeError(
                    f"File too large: {file_size} bytes > {max_size} bytes"
                )
            
            # Read file content
            async with aiofiles.open(resolved_path, 'r', encoding=directive.encoding) as f:
                content = await f.read()
            
            logger.debug(f"Loaded file: {resolved_path} ({len(content)} characters)")
            
            return FileIncludeResult(
                content=content,
                resolved_path=resolved_path,
                size=file_size,
                encoding=directive.encoding,
                cached=False
            )
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {resolved_path} with {directive.encoding}: {e}")
            raise FileInclusionError(
                f"Could not decode file {resolved_path} with encoding {directive.encoding}"
            ) from e
        except OSError as e:
            logger.error(f"OS error reading {resolved_path}: {e}")
            raise FileInclusionError(f"Could not read file {resolved_path}: {e}") from e
    
    def _get_cached_result(self, resolved_path: str) -> Optional[FileIncludeResult]:
        """
        Get cached file result if available and valid.
        
        Args:
            resolved_path: Resolved file path
            
        Returns:
            Cached result if available, None otherwise
        """
        if not self.cache_enabled or resolved_path not in self.file_cache:
            return None
        
        cached_result, cache_time = self.file_cache[resolved_path]
        
        # Check cache TTL
        if time.time() - cache_time > self.cache_ttl:
            logger.debug(f"Cache expired for: {resolved_path}")
            del self.file_cache[resolved_path]
            return None
        
        # Check if file has been modified since caching
        try:
            file_mtime = os.path.getmtime(resolved_path)
            if file_mtime > cache_time:
                logger.debug(f"File modified since cache: {resolved_path}")
                del self.file_cache[resolved_path]
                return None
        except OSError:
            # File might have been deleted
            del self.file_cache[resolved_path]
            return None
        
        return cached_result
    
    def _cache_result(self, resolved_path: str, result: FileIncludeResult) -> None:
        """
        Cache file inclusion result.
        
        Args:
            resolved_path: Resolved file path
            result: File inclusion result to cache
        """
        if not self.cache_enabled:
            return
        
        # Create a copy of the result for caching
        cached_result = FileIncludeResult(
            content=result.content,
            resolved_path=result.resolved_path,
            size=result.size,
            encoding=result.encoding,
            cached=True,
            nested_includes=result.nested_includes.copy()
        )
        
        self.file_cache[resolved_path] = (cached_result, time.time())
        
        # Limit cache size
        if len(self.file_cache) > 1000:  # Max 1000 cached files
            # Remove oldest entries
            oldest_entries = sorted(
                self.file_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )[:100]  # Remove 100 oldest
            
            for path, _ in oldest_entries:
                del self.file_cache[path]
    
    def _has_inclusions(self, content: str) -> bool:
        """Check if content contains file inclusion directives."""
        return bool(
            self.template_pattern.search(content) or 
            self.bracket_pattern.search(content)
        )
    
    def _extract_nested_includes(self, content: str) -> List[str]:
        """Extract list of nested file inclusions from content."""
        nested = []
        
        # Find template syntax inclusions
        for match in self.template_pattern.finditer(content):
            nested.append(match.group(1).strip())
        
        # Find bracket syntax inclusions
        for match in self.bracket_pattern.finditer(content):
            nested.append(match.group(1).strip())
        
        return nested
    
    async def _process_template_syntax(self, content: str, base_dir: Optional[str]) -> str:
        """Process {{ file:path }} syntax in content."""
        async def replace_template_match(match):
            path = match.group(1).strip()
            directive = FileIncludeDirective(
                syntax="template",
                path=path,
                base_dir=base_dir
            )
            
            try:
                result = await self.include_file(directive)
                return result.content
            except FileInclusionError as e:
                if directive.required:
                    raise
                logger.warning(f"Optional file not found: {path} - {e}")
                return f"<!-- File not found: {path} -->"
        
        # Process all matches
        processed_content = content
        for match in self.template_pattern.finditer(content):
            replacement = await replace_template_match(match)
            processed_content = processed_content.replace(match.group(0), replacement, 1)
        
        return processed_content
    
    async def _process_bracket_syntax(self, content: str, base_dir: Optional[str]) -> str:
        """Process << path >> syntax in content."""
        async def replace_bracket_match(match):
            path = match.group(1).strip()
            directive = FileIncludeDirective(
                syntax="bracket",
                path=path,
                base_dir=base_dir
            )
            
            try:
                result = await self.include_file(directive)
                return result.content
            except FileInclusionError as e:
                if directive.required:
                    raise
                logger.warning(f"Optional file not found: {path} - {e}")
                return f"<!-- File not found: {path} -->"
        
        # Process all matches
        processed_content = content
        for match in self.bracket_pattern.finditer(content):
            replacement = await replace_bracket_match(match)
            processed_content = processed_content.replace(match.group(0), replacement, 1)
        
        return processed_content
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and usage metrics."""
        cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        cache_hit_rate = (
            self.metrics["cache_hits"] / cache_total 
            if cache_total > 0 else 0.0
        )
        
        return {
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "files_loaded": self.metrics["files_loaded"],
            "total_bytes_loaded": self.metrics["total_bytes_loaded"],
            "security_violations": self.metrics["security_violations"],
            "circular_references_detected": self.metrics["circular_references_detected"],
            "cached_files": len(self.file_cache),
            "inclusion_depth": len(self.inclusion_stack)
        }
    
    def clear_cache(self) -> None:
        """Clear the file cache."""
        self.file_cache.clear()
        logger.info("File inclusion cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        cache_info = {}
        
        for path, (result, timestamp) in self.file_cache.items():
            cache_info[path] = {
                "size": result.size,
                "encoding": result.encoding,
                "cached_at": timestamp,
                "age_seconds": time.time() - timestamp,
                "nested_includes": len(result.nested_includes)
            }
        
        return cache_info


# Global instance for convenience
_default_processor: Optional[FileInclusionProcessor] = None


def get_default_processor() -> FileInclusionProcessor:
    """Get the default file inclusion processor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = FileInclusionProcessor()
    return _default_processor


async def include_file(path: str, base_dir: Optional[str] = None, **kwargs) -> str:
    """
    Convenience function to include a single file.
    
    Args:
        path: File path to include
        base_dir: Base directory for resolution
        **kwargs: Additional directive options
        
    Returns:
        File content
    """
    processor = get_default_processor()
    directive = FileIncludeDirective(
        syntax="template",
        path=path,
        base_dir=base_dir,
        **kwargs
    )
    
    result = await processor.include_file(directive)
    return result.content


async def process_content(content: str, base_dir: Optional[str] = None) -> str:
    """
    Convenience function to process content with file inclusions.
    
    Args:
        content: Content to process
        base_dir: Base directory for resolution
        
    Returns:
        Processed content
    """
    processor = get_default_processor()
    return await processor.process_content(content, base_dir)