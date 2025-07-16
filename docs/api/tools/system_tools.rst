System Tools
============

Tools for file operations and system command execution.

.. automodule:: orchestrator.tools.system_tools
   :members:
   :undoc-members:
   :show-inheritance:

FileSystemTool
--------------

.. autoclass:: orchestrator.tools.system_tools.FileSystemTool
   :members:
   :undoc-members:
   :show-inheritance:

Provides safe file and directory operations.

**Parameters:**

* ``action`` (string, required): Operation to perform ("read", "write", "copy", "delete", "list", "exists")
* ``path`` (string, required): File or directory path
* ``content`` (string, optional): Content to write (for write action)
* ``destination`` (string, optional): Destination path (for copy action)
* ``encoding`` (string, optional): File encoding (default: "utf-8")
* ``create_dirs`` (boolean, optional): Create parent directories if needed (default: False)

**Actions:**

read
~~~~

Read content from a file:

.. code-block:: python

   result = await fs_tool.execute(
       action="read",
       path="/path/to/file.txt",
       encoding="utf-8"
   )

**Returns:**

.. code-block:: python

   {
       "success": True,
       "content": "File content here...",
       "file_size": 1024,
       "encoding": "utf-8",
       "last_modified": "2024-01-15T10:30:00Z"
   }

write
~~~~~

Write content to a file:

.. code-block:: python

   result = await fs_tool.execute(
       action="write",
       path="/path/to/output.txt",
       content="Hello, World!",
       create_dirs=True
   )

**Returns:**

.. code-block:: python

   {
       "success": True,
       "bytes_written": 13,
       "path": "/path/to/output.txt",
       "created_dirs": ["/path/to"]
   }

copy
~~~~

Copy files or directories:

.. code-block:: python

   result = await fs_tool.execute(
       action="copy",
       path="/source/file.txt",
       destination="/dest/file.txt",
       create_dirs=True
   )

list
~~~~

List directory contents:

.. code-block:: python

   result = await fs_tool.execute(
       action="list",
       path="/path/to/directory"
   )

**Returns:**

.. code-block:: python

   {
       "success": True,
       "items": [
           {
               "name": "file.txt",
               "type": "file",
               "size": 1024,
               "modified": "2024-01-15T10:30:00Z"
           },
           {
               "name": "subdir",
               "type": "directory", 
               "items": 5,
               "modified": "2024-01-14T15:20:00Z"
           }
       ],
       "total_items": 2
   }

exists
~~~~~~

Check if a file or directory exists:

.. code-block:: python

   result = await fs_tool.execute(
       action="exists",
       path="/path/to/check"
   )

**Returns:**

.. code-block:: python

   {
       "success": True,
       "exists": True,
       "type": "file",  # or "directory"
       "size": 1024,
       "permissions": "rw-r--r--"
   }

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.system_tools import FileSystemTool
   import asyncio
   
   async def file_operations():
       fs_tool = FileSystemTool()
       
       # Write a file
       write_result = await fs_tool.execute(
           action="write",
           path="./output/report.txt",
           content="Generated report content...",
           create_dirs=True
       )
       
       # Read it back
       read_result = await fs_tool.execute(
           action="read",
           path="./output/report.txt"
       )
       
       # List directory
       list_result = await fs_tool.execute(
           action="list",
           path="./output"
       )
       
       return {
           "write": write_result,
           "read": read_result,
           "list": list_result
       }
   
   # Run operations
   asyncio.run(file_operations())

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: save_results
       action: write_file
       parameters:
         action: "write"
         path: "./output/{{ inputs.filename }}"
         content: "{{ results.previous_step.output }}"
         create_dirs: true

TerminalTool
------------

.. autoclass:: orchestrator.tools.system_tools.TerminalTool
   :members:
   :undoc-members:
   :show-inheritance:

Executes shell commands safely with proper isolation.

**Parameters:**

* ``command`` (string, required): Command to execute
* ``working_dir`` (string, optional): Working directory (default: current directory)
* ``timeout`` (integer, optional): Timeout in seconds (default: 30)
* ``shell`` (boolean, optional): Use shell for execution (default: False)
* ``env`` (object, optional): Environment variables
* ``capture_output`` (boolean, optional): Capture stdout/stderr (default: True)

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.system_tools import TerminalTool
   import asyncio
   
   async def run_commands():
       terminal = TerminalTool()
       
       # Simple command
       result = await terminal.execute(
           command="ls -la",
           working_dir="/tmp",
           timeout=10
       )
       
       # Command with environment variables
       env_result = await terminal.execute(
           command="echo $MY_VAR",
           env={"MY_VAR": "Hello World"},
           timeout=5
       )
       
       return {
           "ls_output": result,
           "env_output": env_result
       }
   
   # Run commands
   asyncio.run(run_commands())

**Returns:**

.. code-block:: python

   {
       "success": True,
       "stdout": "Command output...",
       "stderr": "",
       "return_code": 0,
       "execution_time": 1.23,
       "command": "ls -la",
       "working_dir": "/tmp"
   }

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: run_tests
       action: run_command
       parameters:
         command: "python -m pytest tests/"
         working_dir: "{{ inputs.project_dir }}"
         timeout: 300
         env:
           PYTHONPATH: "{{ inputs.python_path }}"

Security Features
-----------------

Sandboxing
~~~~~~~~~~

System tools run in restricted environments:

* **Path Restrictions**: Access limited to allowed directories
* **Command Filtering**: Dangerous commands are blocked
* **Resource Limits**: CPU, memory, and time limits enforced
* **Network Isolation**: Network access can be disabled

.. code-block:: python

   # Configure security settings
   fs_tool = FileSystemTool(
       allowed_paths=["/app/data", "/tmp"],
       max_file_size=10_000_000,  # 10MB limit
       blocked_extensions=[".exe", ".bat", ".sh"]
   )

Command Restrictions
~~~~~~~~~~~~~~~~~~~~

Terminal tool blocks potentially dangerous commands:

* System modification commands (``sudo``, ``rm -rf``, etc.)
* Network tools (``wget``, ``curl`` to internal networks)
* Process manipulation (``kill``, ``killall``)
* Package installation (``pip install``, ``apt install``)

.. code-block:: python

   # Safe commands are allowed
   result = await terminal.execute(command="echo hello")  # ✓
   result = await terminal.execute(command="ls -la")      # ✓
   result = await terminal.execute(command="python script.py")  # ✓
   
   # Dangerous commands are blocked
   result = await terminal.execute(command="sudo rm -rf /")     # ✗
   result = await terminal.execute(command="wget evil.com")     # ✗

Error Handling
--------------

Permission Errors
~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "permission_denied",
       "message": "Access denied to /restricted/path",
       "path": "/restricted/path"
   }

File Not Found
~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "file_not_found",
       "message": "File not found: /missing/file.txt",
       "path": "/missing/file.txt"
   }

Command Execution Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "command_failed",
       "message": "Command exited with code 1",
       "return_code": 1,
       "stdout": "...",
       "stderr": "Error message..."
   }

Timeout Errors
~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "timeout",
       "message": "Command timed out after 30 seconds",
       "timeout": 30,
       "partial_output": "..."
   }

Configuration
-------------

System tools can be configured for security and performance:

.. code-block:: yaml

   # config/orchestrator.yaml
   system_tools:
     filesystem:
       allowed_paths:
         - "/app/data"
         - "/tmp"
         - "/var/tmp"
       blocked_extensions:
         - ".exe"
         - ".bat"
         - ".sh"
       max_file_size: 100000000  # 100MB
       max_files_per_operation: 1000
     
     terminal:
       timeout: 30
       max_memory: 512000000  # 512MB
       blocked_commands:
         - "sudo"
         - "rm -rf"
         - "dd"
         - "mkfs"
       allowed_shells:
         - "/bin/bash"
         - "/bin/sh"
       working_dir_restrictions: true

Best Practices
--------------

File Operations
~~~~~~~~~~~~~~~

* **Path Validation**: Always validate file paths
* **Size Limits**: Check file sizes before operations
* **Atomic Operations**: Use temporary files for writes
* **Error Handling**: Handle all possible file errors
* **Cleanup**: Clean up temporary files

.. code-block:: python

   async def safe_file_write(content, path):
       fs_tool = FileSystemTool()
       
       # Write to temporary file first
       temp_path = f"{path}.tmp"
       
       try:
           # Write content
           result = await fs_tool.execute(
               action="write",
               path=temp_path,
               content=content
           )
           
           if result["success"]:
               # Move to final location
               copy_result = await fs_tool.execute(
                   action="copy",
                   path=temp_path,
                   destination=path
               )
               
               # Clean up temp file
               await fs_tool.execute(
                   action="delete",
                   path=temp_path
               )
               
               return copy_result
       
       except Exception as e:
           # Clean up on error
           await fs_tool.execute(action="delete", path=temp_path)
           raise

Command Execution
~~~~~~~~~~~~~~~~~

* **Input Sanitization**: Sanitize all command inputs
* **Timeout Settings**: Set appropriate timeouts
* **Output Limits**: Limit output size to prevent memory issues
* **Environment Control**: Control environment variables
* **Error Logging**: Log command failures for debugging

.. code-block:: python

   async def safe_command_execution(command, working_dir=None):
       terminal = TerminalTool()
       
       # Sanitize command
       if any(dangerous in command for dangerous in ["sudo", "rm -rf", ";"]):
           raise ValueError("Command contains dangerous elements")
       
       # Execute with limits
       result = await terminal.execute(
           command=command,
           working_dir=working_dir,
           timeout=30,
           capture_output=True
       )
       
       return result

Examples
--------

File Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def process_files(input_dir, output_dir):
       fs_tool = FileSystemTool()
       
       # List input files
       files = await fs_tool.execute(
           action="list",
           path=input_dir
       )
       
       results = []
       for file_info in files["items"]:
           if file_info["type"] == "file":
               # Read file
               content = await fs_tool.execute(
                   action="read",
                   path=f"{input_dir}/{file_info['name']}"
               )
               
               # Process content
               processed = content["content"].upper()
               
               # Write processed file
               output_path = f"{output_dir}/processed_{file_info['name']}"
               write_result = await fs_tool.execute(
                   action="write",
                   path=output_path,
                   content=processed,
                   create_dirs=True
               )
               
               results.append(write_result)
       
       return results

Build Automation
~~~~~~~~~~~~~~~~

.. code-block:: python

   async def build_project(project_dir):
       terminal = TerminalTool()
       
       # Install dependencies
       install_result = await terminal.execute(
           command="npm install",
           working_dir=project_dir,
           timeout=300
       )
       
       if not install_result["success"]:
           return install_result
       
       # Run tests
       test_result = await terminal.execute(
           command="npm test",
           working_dir=project_dir,
           timeout=600
       )
       
       if not test_result["success"]:
           return test_result
       
       # Build project
       build_result = await terminal.execute(
           command="npm run build",
           working_dir=project_dir,
           timeout=600
       )
       
       return {
           "install": install_result,
           "test": test_result,
           "build": build_result
       }

For more examples, see :doc:`../../tutorials/examples/file_operations` and :doc:`../../tutorials/examples/automation`.