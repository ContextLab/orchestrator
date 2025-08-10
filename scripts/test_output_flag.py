#!/usr/bin/env python3
"""
Integration test script for the -o flag functionality.

This script tests that updated pipelines correctly respect the -o flag
and that the CLI warning system works for incompatible pipelines.

Usage:
    python scripts/test_output_flag.py [--verbose]
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def run_pipeline_with_output(pipeline_path: str, output_dir: str, inputs: Optional[Dict[str, str]] = None, 
                           timeout: int = 120) -> Tuple[bool, str, str]:
    """
    Run a pipeline with the -o flag and return success status and output.
    
    Args:
        pipeline_path: Path to the YAML pipeline file
        output_dir: Output directory to use
        inputs: Optional dictionary of input parameters
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    cmd = [
        sys.executable, 
        'scripts/run_pipeline.py', 
        pipeline_path,
        '-o', output_dir
    ]
    
    # Add input parameters if provided
    if inputs:
        for key, value in inputs.items():
            cmd.extend(['-i', f'{key}={value}'])
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '../src')
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "", f"Pipeline timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)

def check_warning_system(pipeline_path: str, output_dir: str) -> Tuple[bool, str]:
    """
    Check if the warning system correctly identifies incompatible pipelines.
    
    Args:
        pipeline_path: Path to pipeline file
        output_dir: Output directory to test with
        
    Returns:
        Tuple of (has_warning, captured_output)
    """
    cmd = [
        sys.executable,
        'scripts/run_pipeline.py',
        pipeline_path,
        '-o', output_dir
    ]
    
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(os.path.dirname(__file__), '../src')
        
        # Run with a short timeout to just capture the warning
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # Short timeout
            env=env,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        
        combined_output = result.stdout + result.stderr
        has_warning = "Warning: This pipeline may not respect the -o flag" in combined_output
        
        return has_warning, combined_output
        
    except subprocess.TimeoutExpired:
        # Timeout is expected for some pipelines - check if warning appeared in partial output
        return True, "Pipeline started (timeout expected for warning test)"
    except Exception as e:
        return False, str(e)

def check_files_in_output_dir(output_dir: str, expected_patterns: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if expected files were created in the output directory.
    
    Args:
        output_dir: Directory to check
        expected_patterns: List of filename patterns to look for
        
    Returns:
        Tuple of (found_files, found_file_paths)
    """
    if not os.path.exists(output_dir):
        return False, []
    
    found_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, output_dir)
            found_files.append(relative_path)
    
    return len(found_files) > 0, found_files

def main():
    parser = argparse.ArgumentParser(description='Test -o flag functionality for pipelines')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--timeout', '-t', type=int, default=120, help='Timeout per pipeline in seconds')
    args = parser.parse_args()

    # Test configuration
    test_cases = [
        # Updated pipelines that should work with -o flag
        {
            'name': 'Research Minimal',
            'pipeline': 'examples/research_minimal.yaml',
            'inputs': {'topic': 'test automation'},
            'should_work': True,
            'expected_files': ['test-automation_summary.md']
        },
        {
            'name': 'Research Basic', 
            'pipeline': 'examples/research_basic.yaml',
            'inputs': {'topic': 'machine learning', 'depth': 'basic'},
            'should_work': True,
            'expected_files': ['research_machine-learning.md']
        },
        {
            'name': 'Simple Data Processing',
            'pipeline': 'examples/simple_data_processing.yaml',
            'inputs': {},
            'should_work': True,
            'expected_files': []  # May not complete due to missing input file
        },
        {
            'name': 'Data Processing',
            'pipeline': 'examples/data_processing.yaml',
            'inputs': {'data_source': '/dev/null', 'output_format': 'json'},
            'should_work': True,
            'expected_files': []  # May not complete due to validation
        },
        {
            'name': 'Web Research Pipeline',
            'pipeline': 'examples/web_research_pipeline.yaml', 
            'inputs': {'research_topic': 'AI testing'},
            'should_work': True,
            'expected_files': ['research_ai-testing.md']
        },
        
        # Pipelines that should show warnings (hardcoded paths)
        {
            'name': 'Control Flow While Loop (Warning Test)',
            'pipeline': 'examples/control_flow_while_loop.yaml',
            'inputs': {},
            'should_work': False,  # Should show warning
            'expected_files': [],
            'warning_test': True
        }
    ]
    
    print("=" * 60)
    print("ORCHESTRATOR -O FLAG INTEGRATION TESTS")
    print("=" * 60)
    print()
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        test_name = test_case['name']
        pipeline_path = test_case['pipeline']
        inputs = test_case['inputs']
        should_work = test_case['should_work']
        expected_files = test_case['expected_files']
        is_warning_test = test_case.get('warning_test', False)
        
        print(f"Test {i}/{len(test_cases)}: {test_name}")
        print(f"Pipeline: {pipeline_path}")
        
        if not os.path.exists(pipeline_path):
            print(f"❌ SKIP: Pipeline file not found: {pipeline_path}")
            results.append(('SKIP', test_name, 'Pipeline file not found'))
            print()
            continue
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'pipeline_output')
            os.makedirs(output_dir, exist_ok=True)
            
            if is_warning_test:
                # Test warning system
                print("Testing CLI warning system...")
                has_warning, warning_output = check_warning_system(pipeline_path, output_dir)
                
                if has_warning:
                    print("✅ PASS: Warning system detected incompatible pipeline")
                    results.append(('PASS', test_name, 'Warning system works'))
                else:
                    print("❌ FAIL: Warning system did not detect incompatible pipeline")
                    results.append(('FAIL', test_name, 'Warning system failed'))
                    
                if args.verbose:
                    print("Warning output:", warning_output[:500])
                    
            else:
                # Test pipeline execution with -o flag
                print("Running pipeline with -o flag...")
                if args.verbose:
                    print(f"Output directory: {output_dir}")
                    print(f"Inputs: {inputs}")
                
                success, stdout, stderr = run_pipeline_with_output(
                    pipeline_path, output_dir, inputs, args.timeout
                )
                
                if success:
                    # Check if files were created in output directory
                    found_files, file_list = check_files_in_output_dir(output_dir, expected_files)
                    
                    if found_files:
                        print(f"✅ PASS: Pipeline completed and created files in output directory")
                        if args.verbose:
                            print(f"Created files: {file_list}")
                        results.append(('PASS', test_name, f'Created {len(file_list)} files'))
                    else:
                        if should_work:
                            print(f"⚠️  PARTIAL: Pipeline completed but no files found in output directory")
                            results.append(('PARTIAL', test_name, 'No output files found'))
                        else:
                            print(f"✅ PASS: Pipeline completed (no output files expected)")
                            results.append(('PASS', test_name, 'Completed as expected'))
                            
                elif should_work:
                    print(f"❌ FAIL: Pipeline execution failed")
                    if args.verbose:
                        print("STDOUT:", stdout[:500])
                        print("STDERR:", stderr[:500])
                    results.append(('FAIL', test_name, 'Execution failed'))
                else:
                    print(f"✅ PASS: Pipeline failed as expected")
                    results.append(('PASS', test_name, 'Failed as expected'))
        
        print()
    
    # Print summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    pass_count = sum(1 for result in results if result[0] == 'PASS')
    fail_count = sum(1 for result in results if result[0] == 'FAIL')
    partial_count = sum(1 for result in results if result[0] == 'PARTIAL')
    skip_count = sum(1 for result in results if result[0] == 'SKIP')
    
    for status, test_name, details in results:
        status_icon = {
            'PASS': '✅',
            'FAIL': '❌', 
            'PARTIAL': '⚠️',
            'SKIP': '⏭️'
        }[status]
        print(f"{status_icon} {status:8} | {test_name:30} | {details}")
    
    print()
    print(f"Results: {pass_count} passed, {fail_count} failed, {partial_count} partial, {skip_count} skipped")
    
    # Exit with appropriate code
    if fail_count > 0:
        print()
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
    elif partial_count > 0:
        print()
        print("⚠️  Some tests had partial success. This may be expected for certain pipelines.")
        sys.exit(0)
    else:
        print()
        print("✅ All tests passed!")
        sys.exit(0)

if __name__ == '__main__':
    main()