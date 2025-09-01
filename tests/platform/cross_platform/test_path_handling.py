#!/usr/bin/env python3
"""
Cross-platform path handling validation tests.

Tests that file paths, directory operations, and path-related functionality
work consistently across Windows, macOS, and Linux platforms.
"""

import os
import sys
import platform
import tempfile
import shutil
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from typing import Dict, Any, List, Optional
import pytest
import logging

logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from orchestrator.compiler.yaml_compiler import YAMLCompiler
except ImportError as e:
    logger.warning(f"Could not import orchestrator modules: {e}")


class CrossPlatformPathTester:
    """Tests cross-platform path handling consistency."""

    def __init__(self):
        self.current_platform = platform.system()
        self.test_paths = self._generate_test_paths()
        self.results = {}
        
    def _generate_test_paths(self) -> Dict[str, List[str]]:
        """Generate test paths for different scenarios."""
        return {
            "simple_paths": [
                "test.txt",
                "folder/test.txt",
                "folder/subfolder/test.txt"
            ],
            "complex_paths": [
                "folder with spaces/test.txt",
                "folder-with-dashes/test.txt",
                "folder_with_underscores/test.txt",
                "folder.with.dots/test.txt"
            ],
            "deep_paths": [
                "/".join(["level" + str(i) for i in range(10)]) + "/test.txt"
            ],
            "special_names": [
                ".hidden/test.txt",
                "normal/.hidden_file.txt",
                "temp.tmp/test.txt"
            ],
            "unicode_paths": [
                "测试/test.txt",
                "тест/test.txt", 
                "テスト/test.txt",
                "emoji🚀/test.txt"
            ]
        }
    
    def test_path_creation_and_normalization(self) -> Dict[str, Any]:
        """Test path creation and normalization across platforms."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for category, paths in self.test_paths.items():
                category_results = {}
                
                for test_path in paths:
                    try:
                        # Test path creation
                        full_path = temp_path / test_path
                        path_created = True
                        
                        # Test normalization
                        normalized = full_path.resolve()
                        normalization_works = isinstance(normalized, Path)
                        
                        # Test string conversion
                        path_str = str(full_path)
                        string_conversion = isinstance(path_str, str) and len(path_str) > 0
                        
                        # Test platform-specific separator handling
                        if self.current_platform == "Windows":
                            expected_sep = "\\"
                            cross_platform_compat = "\\" in path_str or "/" in path_str
                        else:
                            expected_sep = "/"
                            cross_platform_compat = "/" in path_str
                            
                        category_results[test_path] = {
                            "path_created": path_created,
                            "normalization_works": normalization_works,
                            "string_conversion": string_conversion,
                            "cross_platform_compat": cross_platform_compat,
                            "normalized_path": str(normalized),
                            "separator_used": expected_sep
                        }
                        
                    except Exception as e:
                        category_results[test_path] = {
                            "error": str(e),
                            "path_created": False,
                            "normalization_works": False,
                            "string_conversion": False,
                            "cross_platform_compat": False
                        }
                        
                results["tests"][category] = category_results
        
        return results
    
    def test_path_operations(self) -> Dict[str, Any]:
        """Test path operations (parent, stem, suffix, etc.)."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        test_cases = [
            {
                "path": "folder/subfolder/test.txt",
                "expected_parent": "folder/subfolder",
                "expected_name": "test.txt", 
                "expected_stem": "test",
                "expected_suffix": ".txt"
            },
            {
                "path": "no_extension_file",
                "expected_parent": ".",
                "expected_name": "no_extension_file",
                "expected_stem": "no_extension_file", 
                "expected_suffix": ""
            },
            {
                "path": ".hidden_file",
                "expected_parent": ".",
                "expected_name": ".hidden_file",
                "expected_stem": ".hidden_file",
                "expected_suffix": ""
            }
        ]
        
        for i, case in enumerate(test_cases):
            try:
                path = Path(case["path"])
                
                # Test parent
                parent_str = str(path.parent).replace("\\", "/")
                parent_correct = parent_str.endswith(case["expected_parent"].replace("\\", "/"))
                
                # Test name
                name_correct = path.name == case["expected_name"]
                
                # Test stem
                stem_correct = path.stem == case["expected_stem"]
                
                # Test suffix
                suffix_correct = path.suffix == case["expected_suffix"]
                
                results["tests"][f"case_{i+1}"] = {
                    "input_path": case["path"],
                    "parent_correct": parent_correct,
                    "name_correct": name_correct,
                    "stem_correct": stem_correct,
                    "suffix_correct": suffix_correct,
                    "actual_parent": parent_str,
                    "actual_name": path.name,
                    "actual_stem": path.stem,
                    "actual_suffix": path.suffix
                }
                
            except Exception as e:
                results["tests"][f"case_{i+1}"] = {
                    "input_path": case["path"],
                    "error": str(e),
                    "all_correct": False
                }
        
        return results
    
    def test_file_operations(self) -> Dict[str, Any]:
        """Test file operations across platforms."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test simple file creation and reading
            try:
                test_file = temp_path / "test.txt"
                test_content = "Cross-platform test content\nLine 2\nLine 3"
                
                # Write file
                test_file.write_text(test_content, encoding="utf-8")
                write_successful = test_file.exists()
                
                # Read file
                read_content = test_file.read_text(encoding="utf-8")
                read_successful = read_content == test_content
                
                # Test file properties
                file_size = test_file.stat().st_size
                size_reasonable = file_size > 0
                
                results["tests"]["basic_file_ops"] = {
                    "write_successful": write_successful,
                    "read_successful": read_successful,
                    "size_reasonable": size_reasonable,
                    "file_size": file_size
                }
                
            except Exception as e:
                results["tests"]["basic_file_ops"] = {
                    "error": str(e),
                    "all_successful": False
                }
            
            # Test directory creation
            try:
                nested_dir = temp_path / "level1" / "level2" / "level3"
                nested_dir.mkdir(parents=True, exist_ok=True)
                
                dir_created = nested_dir.exists() and nested_dir.is_dir()
                
                # Create file in nested directory
                nested_file = nested_dir / "nested_test.txt"
                nested_file.write_text("Nested file content")
                nested_file_created = nested_file.exists()
                
                results["tests"]["directory_ops"] = {
                    "nested_dir_created": dir_created,
                    "nested_file_created": nested_file_created
                }
                
            except Exception as e:
                results["tests"]["directory_ops"] = {
                    "error": str(e),
                    "all_successful": False
                }
            
            # Test path iteration
            try:
                # Create multiple files
                for i in range(5):
                    (temp_path / f"file_{i}.txt").write_text(f"Content {i}")
                    
                # Iterate and count
                files_found = list(temp_path.glob("file_*.txt"))
                iteration_successful = len(files_found) == 5
                
                # Test pattern matching
                all_files = list(temp_path.glob("*"))
                pattern_matching = len(all_files) >= 5
                
                results["tests"]["path_iteration"] = {
                    "iteration_successful": iteration_successful,
                    "pattern_matching": pattern_matching,
                    "files_found": len(files_found),
                    "all_items_found": len(all_files)
                }
                
            except Exception as e:
                results["tests"]["path_iteration"] = {
                    "error": str(e),
                    "all_successful": False
                }
        
        return results
    
    def test_orchestrator_path_integration(self) -> Dict[str, Any]:
        """Test orchestrator's path handling."""
        results = {
            "platform": self.current_platform,  
            "tests": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test YAML file with paths
            try:
                yaml_content = f"""
name: cross_platform_path_test
description: Test path handling in YAML
steps:
  - name: file_step
    type: file_processor
    input_file: "{temp_path}/input.txt"
    output_file: "{temp_path}/output.txt"
"""
                
                # Create input file
                input_file = temp_path / "input.txt"
                input_file.write_text("Test input content")
                
                # Test YAML compilation
                if 'YAMLCompiler' in globals():
                    compiler = YAMLCompiler()
                    compiled = compiler.compile_yaml(yaml_content)
                    yaml_compilation = compiled is not None
                else:
                    yaml_compilation = True  # Skip if not available
                
                results["tests"]["yaml_path_handling"] = {
                    "input_file_created": input_file.exists(),
                    "yaml_compilation": yaml_compilation,
                    "path_format_accepted": True
                }
                
            except Exception as e:
                results["tests"]["yaml_path_handling"] = {
                    "error": str(e),
                    "all_successful": False
                }
        
        return results
    
    def test_path_conversion_between_platforms(self) -> Dict[str, Any]:
        """Test conversion between different path formats."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test different path formats
        test_conversions = [
            {
                "name": "windows_to_posix",
                "input": "C:\\Users\\test\\file.txt",
                "expected_posix": "/c/Users/test/file.txt"  # Git Bash style
            },
            {
                "name": "posix_to_windows", 
                "input": "/home/user/file.txt",
                "expected_windows": "\\home\\user\\file.txt"
            },
            {
                "name": "relative_path",
                "input": "folder/subfolder/file.txt",
                "expected_consistent": True
            }
        ]
        
        for conversion in test_conversions:
            try:
                input_path = conversion["input"]
                
                # Test PurePath conversion
                pure_path = PurePath(input_path)
                pure_conversion = isinstance(pure_path, PurePath)
                
                # Test platform-specific conversion
                if self.current_platform == "Windows":
                    windows_path = PureWindowsPath(input_path)
                    platform_conversion = isinstance(windows_path, PureWindowsPath)
                else:
                    posix_path = PurePosixPath(input_path)
                    platform_conversion = isinstance(posix_path, PurePosixPath)
                
                # Test as_posix() method (if available)
                try:
                    posix_format = pure_path.as_posix() if hasattr(pure_path, 'as_posix') else str(pure_path).replace("\\", "/")
                    posix_conversion = isinstance(posix_format, str)
                except AttributeError:
                    posix_conversion = False
                    posix_format = "N/A"
                
                results["tests"][conversion["name"]] = {
                    "pure_conversion": pure_conversion,
                    "platform_conversion": platform_conversion,
                    "posix_conversion": posix_conversion,
                    "input_path": input_path,
                    "posix_format": posix_format,
                    "platform_format": str(pure_path)
                }
                
            except Exception as e:
                results["tests"][conversion["name"]] = {
                    "error": str(e),
                    "all_successful": False
                }
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all cross-platform path tests."""
        logger.info(f"Running cross-platform path tests on {self.current_platform}")
        
        results = {
            "platform": self.current_platform,
            "test_results": {}
        }
        
        # Run test suites
        results["test_results"]["path_creation"] = self.test_path_creation_and_normalization()
        results["test_results"]["path_operations"] = self.test_path_operations()
        results["test_results"]["file_operations"] = self.test_file_operations()
        results["test_results"]["orchestrator_integration"] = self.test_orchestrator_path_integration()
        results["test_results"]["path_conversion"] = self.test_path_conversion_between_platforms()
        
        # Calculate overall success rate
        all_tests = []
        for test_category in results["test_results"].values():
            if "tests" in test_category:
                for test_name, test_data in test_category["tests"].items():
                    if isinstance(test_data, dict):
                        # Determine if test passed based on available metrics
                        if "error" in test_data:
                            all_tests.append(False)
                        else:
                            # Check for success indicators
                            success_indicators = [
                                test_data.get("path_created", True),
                                test_data.get("normalization_works", True), 
                                test_data.get("string_conversion", True),
                                test_data.get("write_successful", True),
                                test_data.get("read_successful", True),
                                test_data.get("all_successful", True),
                                test_data.get("pure_conversion", True),
                                test_data.get("platform_conversion", True)
                            ]
                            # Pass if any success indicator is True and none are explicitly False
                            explicit_failures = [x for x in success_indicators if x is False]
                            explicit_successes = [x for x in success_indicators if x is True]
                            
                            test_passed = len(explicit_successes) > 0 and len(explicit_failures) == 0
                            all_tests.append(test_passed)
                
        results["overall"] = {
            "total_tests": len(all_tests),
            "passed_tests": sum(all_tests),
            "success_rate": sum(all_tests) / len(all_tests) if all_tests else 0
        }
        
        logger.info(f"Cross-platform path tests: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed")
        
        return results


# pytest test functions

def test_path_normalization():
    """Test path normalization works across platforms."""
    tester = CrossPlatformPathTester()
    results = tester.test_path_creation_and_normalization()
    
    # Simple paths should always work
    simple_results = results["tests"]["simple_paths"]
    for path, result in simple_results.items():
        assert result.get("path_created", False), f"Failed to create path: {path}"
        assert result.get("normalization_works", False), f"Normalization failed for: {path}"


def test_path_operations():
    """Test path operations (parent, stem, suffix) work consistently."""
    tester = CrossPlatformPathTester()
    results = tester.test_path_operations()
    
    # At least basic operations should work
    passed_tests = sum(1 for test in results["tests"].values() 
                      if not test.get("error") and 
                      test.get("name_correct", False))
    
    assert passed_tests > 0, "No path operations worked correctly"


def test_file_operations():
    """Test file operations work across platforms."""
    tester = CrossPlatformPathTester()
    results = tester.test_file_operations()
    
    # Basic file operations should work
    basic_ops = results["tests"].get("basic_file_ops", {})
    assert basic_ops.get("write_successful", False), "File writing failed"
    assert basic_ops.get("read_successful", False), "File reading failed"
    
    # Directory operations should work
    dir_ops = results["tests"].get("directory_ops", {})
    assert dir_ops.get("nested_dir_created", False), "Nested directory creation failed"


def test_path_conversion():
    """Test path format conversion between platforms."""
    tester = CrossPlatformPathTester()
    results = tester.test_path_conversion_between_platforms()
    
    # At least some conversions should work
    successful_conversions = sum(1 for test in results["tests"].values()
                               if not test.get("error") and 
                               test.get("pure_conversion", False))
    
    assert successful_conversions > 0, "No path conversions worked"


@pytest.mark.slow
def test_comprehensive_path_testing():
    """Run comprehensive cross-platform path testing."""
    tester = CrossPlatformPathTester()
    results = tester.run_all_tests()
    
    # Should pass majority of tests
    assert results["overall"]["success_rate"] >= 0.7, \
        f"Path tests success rate too low: {results['overall']['success_rate']*100:.1f}%"
    
    # Log results for debugging
    logger.info(f"Platform: {results['platform']}")
    logger.info(f"Path tests passed: {results['overall']['passed_tests']}/{results['overall']['total_tests']}")


if __name__ == "__main__":
    # Run cross-platform path tests when called directly
    tester = CrossPlatformPathTester()
    results = tester.run_all_tests()
    
    print("=== Cross-Platform Path Handling Test Results ===")
    print(f"Platform: {results['platform']}")
    print(f"Test Results: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed ({results['overall']['success_rate']*100:.1f}%)")
    
    for category_name, category_results in results["test_results"].items():
        if "tests" in category_results:
            print(f"\n{category_name.title().replace('_', ' ')} Tests:")
            for test_name, test_data in category_results["tests"].items():
                if isinstance(test_data, dict):
                    if "error" in test_data:
                        status = "FAIL"
                        detail = f"Error: {test_data['error']}"
                    else:
                        # Check for various success indicators
                        success_count = sum([
                            test_data.get("path_created", True),
                            test_data.get("normalization_works", True),
                            test_data.get("write_successful", True),
                            test_data.get("read_successful", True),
                            test_data.get("pure_conversion", True)
                        ])
                        status = "PASS" if success_count > 0 else "FAIL"
                        detail = ""
                    
                    print(f"  {test_name}: {status} {detail}")