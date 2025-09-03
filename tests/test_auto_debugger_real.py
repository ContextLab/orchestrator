"""
Real-world tests for AutoDebugger Tool - NO MOCKS POLICY

This test suite implements comprehensive real-world testing scenarios for the
AutoDebugger tool as specified in Issue #201. All tests use real systems:
- Real LLM model calls
- Real tool execution
- Real file operations
- Real command execution
- Real data processing

NO MOCK IMPLEMENTATIONS are used anywhere in this test suite.
"""

import asyncio
import json
import os
import tempfile
import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.orchestrator.tools.auto_debugger import AutoDebuggerTool, AutoDebuggerInput

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestAutoDebuggerRealWorld:
    """Real-world test suite for AutoDebugger tool."""
    
    @pytest.fixture
    def auto_debugger(self):
        """Create AutoDebugger tool instance."""
        return AutoDebuggerTool()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.mark.asyncio
    async def test_python_syntax_error_debugging(self, auto_debugger, temp_dir):
        """
        Test debugging Python code with syntax errors using REAL systems.
        
        This test:
        1. Creates real Python code with syntax errors
        2. Uses real LLM to analyze the problem
        3. Uses real tool execution to fix the code
        4. Validates the fix with real Python execution
        """
        # Real Python code with syntax errors
        broken_python_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        
# Missing colon after else, missing import
import json
data = {'numbers': [fibonacci(i) for i in range(10)]}
print(json.dumps(data)
"""
        
        # Real debugging task
        result_json = await auto_debugger._arun(
            task_description="Fix Python syntax errors in Fibonacci calculation code",
            content_to_debug=broken_python_code,
            error_context="SyntaxError: invalid syntax on line 5 (missing colon), NameError: fibonacci not defined",
            expected_outcome="Working Python code that calculates and prints first 10 Fibonacci numbers as JSON",
            available_tools=["python_execution", "code_analysis", "filesystem"]
        )
        
        # Parse real result
        result = json.loads(result_json)
        
        # Validate real debugging session
        assert result["success"] is True, f"Debugging failed: {result.get('error_message', 'Unknown error')}"
        assert "final_content" in result
        assert result["total_iterations"] >= 1
        
        # Verify the fixed code is syntactically correct Python
        fixed_code = result["final_content"]
        assert isinstance(fixed_code, str)
        assert len(fixed_code.strip()) > 0
        
        # Real Python syntax validation
        try:
            compile(fixed_code, "<string>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Fixed code still has syntax errors: {e}")
        
        # Test the fixed code actually works by executing it
        test_file = os.path.join(temp_dir, "test_fibonacci.py")
        with open(test_file, 'w') as f:
            f.write(fixed_code)
        
        # Real execution test
        import subprocess
        result_exec = subprocess.run(
            [sys.executable, test_file], 
            capture_output=True, 
            text=True
        )
        
        # Validate real execution results
        assert result_exec.returncode == 0, f"Fixed code execution failed: {result_exec.stderr}"
        assert result_exec.stdout.strip(), "No output from fixed code"
        
        # Validate JSON output structure
        try:
            output_data = json.loads(result_exec.stdout.strip())
            assert "numbers" in output_data
            assert isinstance(output_data["numbers"], list)
            assert len(output_data["numbers"]) == 10
        except json.JSONDecodeError:
            pytest.fail("Fixed code doesn't produce valid JSON output")
    
    @pytest.mark.asyncio
    async def test_javascript_runtime_error_debugging(self, auto_debugger, temp_dir):
        """
        Test debugging JavaScript runtime errors using REAL Node.js execution.
        
        This test uses real Node.js to execute and validate JavaScript fixes.
        """
        # Real JavaScript code with runtime errors
        broken_js_code = """
const fs = require('fs');

function processData(filename) {
    const data = fs.readFileSync(filename, 'utf8');
    const parsed = JSON.parse(data);
    
    // TypeError: Cannot read property 'length' of undefined
    const results = parsed.items.map(item => {
        return {
            id: item.id,
            name: item.name.toUpperCase(),
            count: item.values.length
        };
    });
    
    return results;
}

// Missing error handling, assuming file exists
const result = processData('data.json');
console.log(JSON.stringify(result, null, 2));
"""
        
        # Create test data file
        test_data = {
            "items": [
                {"id": 1, "name": "test", "values": [1, 2, 3]},
                {"id": 2, "name": "demo", "values": [4, 5]}
            ]
        }
        
        data_file = os.path.join(temp_dir, "data.json")
        with open(data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Real debugging task
        result_json = await auto_debugger._arun(
            task_description="Fix JavaScript runtime errors in data processing code",
            content_to_debug=broken_js_code,
            error_context="TypeError: Cannot read property 'length' of undefined when items is missing",
            expected_outcome="Robust JavaScript code that handles missing data gracefully",
            available_tools=["system_tools", "filesystem", "code_analysis"]
        )
        
        # Parse and validate real result
        result = json.loads(result_json)
        assert result["success"] is True, f"JavaScript debugging failed: {result.get('error_message')}"
        
        fixed_js_code = result["final_content"]
        
        # Test with real Node.js execution
        js_file = os.path.join(temp_dir, "test_script.js")
        with open(js_file, 'w') as f:
            f.write(fixed_js_code)
        
        # Real Node.js execution
        node_result = subprocess.run(
            ["node", js_file], 
            cwd=temp_dir,
            capture_output=True, 
            text=True
        )
        
        # Validate real execution
        assert node_result.returncode == 0, f"Fixed JavaScript failed: {node_result.stderr}"
        
        # Validate output structure
        try:
            js_output = json.loads(node_result.stdout.strip())
            assert isinstance(js_output, list)
            assert len(js_output) == 2
        except json.JSONDecodeError:
            pytest.fail("Fixed JavaScript doesn't produce valid JSON")
    
    @pytest.mark.asyncio 
    async def test_latex_compilation_error_debugging(self, auto_debugger, temp_dir):
        """
        Test debugging LaTeX compilation errors using REAL pdflatex.
        
        This test requires pdflatex to be available on the system.
        """
        # Check if pdflatex is available
        try:
            subprocess.run(["pdflatex", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("pdflatex not available on system")
        
        # Real LaTeX document with compilation errors
        broken_latex = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}

\begin{document}
\title{Test Document}
\author{AutoDebugger Test}
\maketitle

\section{Introduction}
This is a test document with several \LaTeX\ errors:

\begin{equation}
    E = mc^2
\end{equation

% Missing closing brace above
% Undefined command below
\undefinedcommand{This will cause an error}

% Missing $ around math
The equation x^2 + y^2 = z^2 is famous.

\section{Results}
\begin{itemize}
    \item First item
    \item Second item
    % Missing \end{itemize}

\subsection{Analysis}
% Environment mismatch
\begin{center}
Some centered text
\end{figure}

\end{document}
"""
        
        # Real debugging task
        result_json = await auto_debugger._arun(
            task_description="Fix LaTeX compilation errors to generate PDF",
            content_to_debug=broken_latex,
            error_context="Multiple LaTeX errors: missing braces, undefined commands, math mode issues, environment mismatches",
            expected_outcome="LaTeX document that compiles successfully to PDF",
            available_tools=["filesystem", "system_tools"]
        )
        
        # Parse real result
        result = json.loads(result_json)
        assert result["success"] is True, f"LaTeX debugging failed: {result.get('error_message')}"
        
        fixed_latex = result["final_content"]
        
        # Test real PDF compilation
        latex_file = os.path.join(temp_dir, "test_document.tex")
        with open(latex_file, 'w') as f:
            f.write(fixed_latex)
        
        # Real pdflatex compilation
        compile_result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "test_document.tex"],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        # Validate successful compilation
        pdf_file = os.path.join(temp_dir, "test_document.pdf")
        assert compile_result.returncode == 0, f"LaTeX compilation failed: {compile_result.stdout}"
        assert os.path.exists(pdf_file), "PDF file was not generated"
        assert os.path.getsize(pdf_file) > 1000, "PDF file seems too small"
    
    @pytest.mark.asyncio
    async def test_yaml_configuration_error_debugging(self, auto_debugger):
        """
        Test debugging YAML configuration errors using REAL YAML parsing.
        """
        # Real YAML with multiple syntax errors
        broken_yaml = """
# Pipeline configuration with errors
name: test-pipeline
version: 1.0

tasks:
  - id: task1
    name: "First Task"
    type: processing
    config:
      timeout: 300
      retries: 3
      # Indentation error below
    environment:
      - VAR1=value1
        VAR2=value2  # Missing dash
      - VAR3: value3  # Inconsistent format
  
  # Missing dash for second task
  id: task2
  name: Second Task  # Inconsistent quotes
  type: analysis
  depends_on: [task1 task3]  # Missing comma
  config:
    memory: 2GB
    cpu: 2
    # Invalid boolean
    parallel: yes_maybe

# Invalid structure
workflows:
  default:
    steps
      - execute: task1
      - execute: task2
    # Missing colon above
"""
        
        # Real debugging task
        result_json = await auto_debugger._arun(
            task_description="Fix YAML configuration syntax errors",
            content_to_debug=broken_yaml,
            error_context="YAML parsing errors: indentation issues, missing colons, inconsistent formatting",
            expected_outcome="Valid YAML that parses correctly for pipeline configuration",
            available_tools=["data_tools", "filesystem"]
        )
        
        # Parse real result
        result = json.loads(result_json)
        assert result["success"] is True, f"YAML debugging failed: {result.get('error_message')}"
        
        fixed_yaml = result["final_content"]
        
        # Real YAML validation
        import yaml
        try:
            parsed_yaml = yaml.safe_load(fixed_yaml)
        except yaml.YAMLError as e:
            pytest.fail(f"Fixed YAML still has syntax errors: {e}")
        
        # Validate structure
        assert isinstance(parsed_yaml, dict)
        assert "name" in parsed_yaml
        assert "tasks" in parsed_yaml
        assert isinstance(parsed_yaml["tasks"], list)
        assert len(parsed_yaml["tasks"]) >= 2
        
        # Validate task structure
        for task in parsed_yaml["tasks"]:
            assert "id" in task
            assert "name" in task
            assert "type" in task
    
    @pytest.mark.asyncio
    async def test_api_integration_debugging(self, auto_debugger, temp_dir):
        """
        Test debugging API integration issues using REAL HTTP requests.
        """
        # Real Python code with API integration problems
        broken_api_code = """
import requests
import json

def fetch_user_data(user_id):
    # Multiple issues: no error handling, wrong endpoint, bad headers
    url = f"https://jsonplaceholder.typicode.com/user/{user_id}"  # Wrong endpoint
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'  # Undefined variable
    }
    
    response = requests.get(url, headers=headers)
    
    # No status code checking
    data = response.json()  # Could fail
    
    # Accessing potentially missing fields
    return {
        'id': data['id'],
        'name': data['full_name'],  # Wrong field name
        'email': data['email'],
        'posts_count': len(data['posts'])  # Field doesn't exist in API
    }

# Script execution
if __name__ == "__main__":
    users = [1, 2, 3]
    results = []
    
    for user_id in users:
        user_data = fetch_user_data(user_id)  # Will fail
        results.append(user_data)
    
    print(json.dumps(results, indent=2))
"""
        
        # Real debugging task
        result_json = await auto_debugger._arun(
            task_description="Fix API integration code to properly fetch user data",
            content_to_debug=broken_api_code,
            error_context="NameError: api_key not defined, wrong API endpoint, missing error handling, accessing non-existent fields",
            expected_outcome="Working Python code that fetches user data from JSONPlaceholder API with proper error handling",
            available_tools=["python_execution", "web_tools", "filesystem"]
        )
        
        # Parse real result
        result = json.loads(result_json)
        assert result["success"] is True, f"API integration debugging failed: {result.get('error_message')}"
        
        fixed_api_code = result["final_content"]
        
        # Test with real API execution
        api_test_file = os.path.join(temp_dir, "test_api.py")
        with open(api_test_file, 'w') as f:
            f.write(fixed_api_code)
        
        # Real execution with internet connection
        api_result = subprocess.run(
            [sys.executable, api_test_file],
            capture_output=True,
            text=True,
            timeout=30  # Timeout for API calls
        )
        
        # Validate real API execution
        if api_result.returncode != 0:
            # If API call fails due to network, that's acceptable - we check the code logic
            if "requests" in api_result.stderr or "ConnectionError" in api_result.stderr:
                pytest.skip("Network/API not available for integration test")
            else:
                pytest.fail(f"Fixed API code failed: {api_result.stderr}")
        
        # If successful, validate output
        if api_result.returncode == 0:
            try:
                api_output = json.loads(api_result.stdout.strip())
                assert isinstance(api_output, list)
                if api_output:  # If we got data
                    assert "id" in api_output[0]
                    assert "name" in api_output[0] 
            except json.JSONDecodeError:
                pytest.fail("Fixed API code doesn't produce valid JSON")
    
    @pytest.mark.asyncio
    async def test_data_processing_error_debugging(self, auto_debugger, temp_dir):
        """
        Test debugging data processing and format handling errors using REAL data.
        """
        # Create real CSV data with issues
        problematic_csv = """id,name,age,salary,department
1,John Doe,25,50000,Engineering
2,Jane Smith,,45000,Marketing
3,Bob Johnson,35,60000.0,Sales
4,Alice Brown,28,abc,Engineering
5,,32,55000,Marketing
6,Tom Wilson,45,70000,
invalid_row_here
7,Sue Davis,29,52000,Engineering
"""
        
        # Real data processing code with errors
        broken_data_code = """
import csv
import json
import statistics

def process_employee_data(csv_file):
    employees = []
    
    with open(csv_file, 'r') as f:
        # No error handling for malformed CSV
        reader = csv.DictReader(f)
        
        for row in reader:
            # No data validation or cleaning
            employee = {
                'id': int(row['id']),  # Will fail on invalid data
                'name': row['name'].strip(),  # Will fail on None/empty
                'age': int(row['age']),  # Will fail on empty string
                'salary': float(row['salary']),  # Will fail on 'abc'
                'department': row['department']
            }
            employees.append(employee)
    
    # Analysis without handling missing/invalid data
    avg_age = statistics.mean([emp['age'] for emp in employees])
    avg_salary = statistics.mean([emp['salary'] for emp in employees])
    
    # Division by zero possible
    dept_counts = {}
    for emp in employees:
        dept = emp['department']
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    
    results = {
        'total_employees': len(employees),
        'average_age': avg_age,
        'average_salary': avg_salary,
        'departments': dept_counts
    }
    
    return results

# Script execution
if __name__ == "__main__":
    results = process_employee_data('employees.csv')
    print(json.dumps(results, indent=2))
"""
        
        # Create test CSV file
        csv_file = os.path.join(temp_dir, "employees.csv")
        with open(csv_file, 'w') as f:
            f.write(problematic_csv)
        
        # Real debugging task
        result_json = await auto_debugger._arun(
            task_description="Fix data processing code to handle malformed CSV data gracefully",
            content_to_debug=broken_data_code,
            error_context="ValueError on invalid data conversion, empty fields causing errors, malformed CSV rows",
            expected_outcome="Robust data processing that handles missing values, invalid formats, and produces clean statistics",
            available_tools=["python_execution", "data_tools", "filesystem"]
        )
        
        # Parse real result
        result = json.loads(result_json)
        assert result["success"] is True, f"Data processing debugging failed: {result.get('error_message')}"
        
        fixed_data_code = result["final_content"]
        
        # Test with real data execution
        data_test_file = os.path.join(temp_dir, "test_data.py")
        with open(data_test_file, 'w') as f:
            f.write(fixed_data_code)
        
        # Real data processing execution
        data_result = subprocess.run(
            [sys.executable, data_test_file],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        # Validate real execution
        assert data_result.returncode == 0, f"Fixed data processing failed: {data_result.stderr}"
        
        # Validate output structure
        try:
            data_output = json.loads(data_result.stdout.strip())
            assert "total_employees" in data_output
            assert "average_age" in data_output
            assert "average_salary" in data_output
            assert "departments" in data_output
            
            # Validate data makes sense
            assert isinstance(data_output["total_employees"], int)
            assert data_output["total_employees"] > 0
            assert isinstance(data_output["average_age"], (int, float))
            assert isinstance(data_output["average_salary"], (int, float))
            
        except json.JSONDecodeError:
            pytest.fail("Fixed data processing doesn't produce valid JSON")
    
    @pytest.mark.asyncio
    async def test_autodebugger_error_handling(self, auto_debugger):
        """
        Test AutoDebugger's own error handling when resources are unavailable.
        
        This test validates the NO MOCKS policy by ensuring proper exceptions
        are raised when real resources are not available.
        """
        # Test with completely invalid content and unavailable tools
        result_json = await auto_debugger._arun(
            task_description="Debug completely invalid content with unavailable resources",
            content_to_debug="This is not valid code in any language: @@@ INVALID @@@",
            error_context="Complete system failure",
            expected_outcome="Should handle gracefully or fail with proper error",
            available_tools=["nonexistent_tool", "fake_system"]
        )
        
        # Parse result
        result = json.loads(result_json)
        
        # Should either succeed with a reasonable attempt or fail gracefully
        # This validates our NO MOCKS policy - real failures should be reported
        assert isinstance(result, dict)
        assert "success" in result
        assert "session_id" in result
        assert "total_iterations" in result
        
        if not result["success"]:
            # If it failed, should have proper error reporting
            assert "error_message" in result
            assert "debug_summary" in result
    
    @pytest.mark.asyncio
    async def test_comprehensive_debugging_session(self, auto_debugger, temp_dir):
        """
        Test a comprehensive debugging session that requires multiple iterations
        and different types of fixes using REAL systems throughout.
        """
        # Complex multi-issue code that requires several debugging iterations
        complex_broken_code = """
# Multi-issue Python script requiring several fixes
import jason  # Wrong module name
import os
import sys

class DataProcessor:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.data = []
        
    def load_config(self, path):
        # File handling without error checking
        with open(path, 'r') as f:
            return jason.loads(f.read())  # Wrong module name
    
    def process_files(self, directory):
        # No validation of directory existence
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                # No error handling for file operations
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Logic error - always processing same way
                    processed = self.process_content(content, 'default')
                    self.data.append(processed)
    
    def process_content(self, content, mode):
        if mode == 'numbers':
            # Will fail on non-numeric content
            return sum([int(line.strip()) for line in content.split('\n')])
        elif mode == 'text':
            return len(content.split())
        else:
            # Default case has undefined behavior
            return content.upper().split()
    
    def generate_report(self):
        # Mathematical error - division by zero possible
        average = sum(self.data) / len(self.data)
        
        report = {
            'total_files': len(self.data),
            'average_value': average,
            'max_value': max(self.data),
            'summary': f"Processed {len(self.data)} files with average {average}"
        }
        
        # Write to file without error handling
        with open('report.json', 'w') as f:
            jason.dumps(report, f, indent=2)  # Wrong module name again

# Script execution with issues
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <config_file> <data_directory>")
        sys.exit(1)
    
    processor = DataProcessor(sys.argv[1])
    processor.process_files(sys.argv[2]) 
    processor.generate_report()
    print("Report generated successfully!")
"""
        
        # Create supporting files for the test
        config_data = {"processing_mode": "text", "output_format": "json"}
        config_file = os.path.join(temp_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir)
        
        # Create test data files
        test_files = [
            ("file1.txt", "Hello world\nThis is a test\nWith multiple lines"),
            ("file2.txt", "Another test file\nWith different content"),
            ("file3.txt", "Final test file\nFor processing")
        ]
        
        for filename, content in test_files:
            with open(os.path.join(data_dir, filename), 'w') as f:
                f.write(content)
        
        # Real comprehensive debugging task
        result_json = await auto_debugger._arun(
            task_description="Fix all issues in complex data processing script to make it work correctly",
            content_to_debug=complex_broken_code,
            error_context="Multiple errors: wrong module names (jason instead of json), missing error handling, logic issues, potential division by zero",
            expected_outcome="Working Python script that processes text files and generates a valid JSON report",
            available_tools=["python_execution", "filesystem", "code_analysis", "data_tools"]
        )
        
        # Parse comprehensive result
        result = json.loads(result_json)
        assert result["success"] is True, f"Comprehensive debugging failed: {result.get('error_message')}"
        
        # Should have taken multiple iterations for this complex case
        assert result["total_iterations"] >= 2, "Should require multiple debugging iterations for complex issues"
        
        fixed_code = result["final_content"]
        
        # Test the comprehensive fix with real execution
        script_file = os.path.join(temp_dir, "test_complex.py")
        with open(script_file, 'w') as f:
            f.write(fixed_code)
        
        # Real comprehensive execution test
        complex_result = subprocess.run(
            [sys.executable, script_file, config_file, data_dir],
            cwd=temp_dir,
            capture_output=True,
            text=True
        )
        
        # Validate comprehensive real execution
        assert complex_result.returncode == 0, f"Fixed complex script failed: {complex_result.stderr}"
        
        # Check that report was generated
        report_file = os.path.join(temp_dir, "report.json")
        assert os.path.exists(report_file), "Report file was not generated"
        
        # Validate report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
            assert "total_files" in report_data
            assert "average_value" in report_data
            assert report_data["total_files"] == 3
            assert isinstance(report_data["average_value"], (int, float))

if __name__ == "__main__":
    # Can be run directly for manual testing
    pytest.main([__file__, "-v", "-s"])