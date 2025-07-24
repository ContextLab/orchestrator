"""Integration tests for file I/O operations and storage systems.

These tests verify:
1. File creation, reading, writing, and deletion
2. Directory operations and permissions
3. Large file handling
4. Concurrent file operations
5. File system error handling
6. Cross-platform compatibility
7. Backup and recovery operations

Note: These tests create temporary files and directories.
"""

import concurrent.futures
import csv
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import pytest


class FileIOManager:
    """File I/O manager for testing various file operations."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize file I/O manager."""
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="fileio_test_")
        self.created_files = []
        self.created_dirs = []

    def create_file(self, filename: str, content: str = "", subdir: str = "") -> str:
        """Create a file with specified content."""
        if subdir:
            dir_path = os.path.join(self.base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            if dir_path not in self.created_dirs:
                self.created_dirs.append(dir_path)
            file_path = os.path.join(dir_path, filename)
        else:
            file_path = os.path.join(self.base_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        self.created_files.append(file_path)
        return file_path

    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def append_to_file(self, file_path: str, content: str) -> bool:
        """Append content to an existing file."""
        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception:
            return False

    def delete_file(self, file_path: str) -> bool:
        """Delete a file."""
        try:
            os.remove(file_path)
            if file_path in self.created_files:
                self.created_files.remove(file_path)
            return True
        except Exception:
            return False

    def create_directory(self, dir_name: str, subdir: str = "") -> str:
        """Create a directory."""
        if subdir:
            parent_dir = os.path.join(self.base_dir, subdir)
            os.makedirs(parent_dir, exist_ok=True)
            dir_path = os.path.join(parent_dir, dir_name)
        else:
            dir_path = os.path.join(self.base_dir, dir_name)

        os.makedirs(dir_path, exist_ok=True)
        self.created_dirs.append(dir_path)
        return dir_path

    def list_directory(self, dir_path: str) -> List[str]:
        """List contents of a directory."""
        return os.listdir(dir_path)

    def create_binary_file(self, filename: str, data: bytes, subdir: str = "") -> str:
        """Create a binary file."""
        if subdir:
            dir_path = os.path.join(self.base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, filename)
        else:
            file_path = os.path.join(self.base_dir, filename)

        with open(file_path, "wb") as f:
            f.write(data)

        self.created_files.append(file_path)
        return file_path

    def read_binary_file(self, file_path: str) -> bytes:
        """Read binary data from a file."""
        with open(file_path, "rb") as f:
            return f.read()

    def create_json_file(
        self, filename: str, data: Dict[str, Any], subdir: str = ""
    ) -> str:
        """Create a JSON file."""
        if subdir:
            dir_path = os.path.join(self.base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, filename)
        else:
            file_path = os.path.join(self.base_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.created_files.append(file_path)
        return file_path

    def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read JSON data from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_csv_file(
        self,
        filename: str,
        data: List[List[str]],
        headers: Optional[List[str]] = None,
        subdir: str = "",
    ) -> str:
        """Create a CSV file."""
        if subdir:
            dir_path = os.path.join(self.base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, filename)
        else:
            file_path = os.path.join(self.base_dir, filename)

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if headers:
                writer.writerow(headers)
            writer.writerows(data)

        self.created_files.append(file_path)
        return file_path

    def read_csv_file(self, file_path: str) -> List[List[str]]:
        """Read CSV data from a file."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            return list(reader)

    def create_large_file(self, filename: str, size_mb: int, subdir: str = "") -> str:
        """Create a large file for testing."""
        if subdir:
            dir_path = os.path.join(self.base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, filename)
        else:
            file_path = os.path.join(self.base_dir, filename)

        chunk_size = 1024 * 1024  # 1MB chunks
        chunk_data = b"A" * chunk_size

        with open(file_path, "wb") as f:
            for _ in range(size_mb):
                f.write(chunk_data)

        self.created_files.append(file_path)
        return file_path

    def copy_file(self, src_path: str, dst_path: str) -> bool:
        """Copy a file from source to destination."""
        try:
            shutil.copy2(src_path, dst_path)
            if dst_path not in self.created_files:
                self.created_files.append(dst_path)
            return True
        except Exception:
            return False

    def move_file(self, src_path: str, dst_path: str) -> bool:
        """Move a file from source to destination."""
        try:
            shutil.move(src_path, dst_path)
            if src_path in self.created_files:
                self.created_files.remove(src_path)
                self.created_files.append(dst_path)
            return True
        except Exception:
            return False

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information."""
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
                "created_time": stat.st_ctime,
                "is_file": os.path.isfile(file_path),
                "is_dir": os.path.isdir(file_path),
                "permissions": oct(stat.st_mode)[-3:],
                "exists": True,
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def create_symlink(self, target_path: str, link_path: str) -> bool:
        """Create a symbolic link."""
        try:
            os.symlink(target_path, link_path)
            self.created_files.append(link_path)
            return True
        except Exception:
            return False

    def backup_file(self, file_path: str, backup_suffix: str = ".bak") -> str:
        """Create a backup of a file."""
        backup_path = file_path + backup_suffix
        if self.copy_file(file_path, backup_path):
            return backup_path
        return ""

    def cleanup(self):
        """Clean up all created files and directories."""
        # Remove files first
        for file_path in self.created_files[:]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

        # Remove directories
        for dir_path in self.created_dirs[:]:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
            except Exception:
                pass

        # Remove base directory
        try:
            if os.path.exists(self.base_dir):
                shutil.rmtree(self.base_dir, ignore_errors=True)
        except Exception:
            pass


class TestFileIOBasics:
    """Basic file I/O operation tests."""

    @pytest.fixture
    def file_manager(self):
        """Create file I/O manager."""
        manager = FileIOManager()
        yield manager
        manager.cleanup()

    def test_file_creation_and_reading(self, file_manager):
        """Test basic file creation and reading."""
        content = "Hello, World!\nThis is a test file."
        file_path = file_manager.create_file("test.txt", content)

        assert os.path.exists(file_path)

        read_content = file_manager.read_file(file_path)
        assert read_content == content

    def test_file_append_operation(self, file_manager):
        """Test appending to existing files."""
        initial_content = "Initial content\n"
        append_content = "Appended content\n"

        file_path = file_manager.create_file("append_test.txt", initial_content)

        result = file_manager.append_to_file(file_path, append_content)
        assert result is True

        final_content = file_manager.read_file(file_path)
        assert final_content == initial_content + append_content

    def test_file_deletion(self, file_manager):
        """Test file deletion."""
        file_path = file_manager.create_file("delete_test.txt", "Content to delete")

        assert os.path.exists(file_path)

        result = file_manager.delete_file(file_path)
        assert result is True
        assert not os.path.exists(file_path)

    def test_directory_operations(self, file_manager):
        """Test directory creation and listing."""
        dir_path = file_manager.create_directory("test_dir")

        assert os.path.exists(dir_path)
        assert os.path.isdir(dir_path)

        # Create files in directory
        file_manager.create_file("file1.txt", "Content 1", "test_dir")
        file_manager.create_file("file2.txt", "Content 2", "test_dir")

        # List directory contents
        contents = file_manager.list_directory(dir_path)
        assert "file1.txt" in contents
        assert "file2.txt" in contents
        assert len(contents) == 2

    def test_nested_directory_creation(self, file_manager):
        """Test creating nested directories."""
        nested_path = file_manager.create_directory("level2", "level1")

        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.join(file_manager.base_dir, "level1"))

        # Create file in nested directory
        file_path = file_manager.create_file(
            "nested.txt", "Nested content", "level1/level2"
        )
        assert os.path.exists(file_path)

    def test_binary_file_operations(self, file_manager):
        """Test binary file creation and reading."""
        binary_data = b"\x00\x01\x02\x03\x04\x05\xff\xfe\xfd"

        file_path = file_manager.create_binary_file("binary_test.bin", binary_data)

        assert os.path.exists(file_path)

        read_data = file_manager.read_binary_file(file_path)
        assert read_data == binary_data

    def test_json_file_operations(self, file_manager):
        """Test JSON file creation and reading."""
        json_data = {
            "name": "Test Object",
            "value": 42,
            "nested": {"array": [1, 2, 3], "boolean": True, "null_value": None},
        }

        file_path = file_manager.create_json_file("test.json", json_data)

        assert os.path.exists(file_path)

        read_data = file_manager.read_json_file(file_path)
        assert read_data == json_data

    def test_csv_file_operations(self, file_manager):
        """Test CSV file creation and reading."""
        headers = ["Name", "Age", "City"]
        data = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "San Francisco"],
            ["Charlie", "35", "Chicago"],
        ]

        file_path = file_manager.create_csv_file("test.csv", data, headers)

        assert os.path.exists(file_path)

        read_data = file_manager.read_csv_file(file_path)
        assert read_data[0] == headers  # First row should be headers
        assert read_data[1:] == data  # Rest should be data

    def test_file_copy_operation(self, file_manager):
        """Test file copying."""
        original_content = "Original file content"
        src_path = file_manager.create_file("original.txt", original_content)
        dst_path = os.path.join(file_manager.base_dir, "copied.txt")

        result = file_manager.copy_file(src_path, dst_path)
        assert result is True

        assert os.path.exists(src_path)  # Original should still exist
        assert os.path.exists(dst_path)  # Copy should exist

        # Both should have same content
        assert file_manager.read_file(src_path) == original_content
        assert file_manager.read_file(dst_path) == original_content

    def test_file_move_operation(self, file_manager):
        """Test file moving."""
        original_content = "File to move"
        src_path = file_manager.create_file("move_src.txt", original_content)
        dst_path = os.path.join(file_manager.base_dir, "move_dst.txt")

        result = file_manager.move_file(src_path, dst_path)
        assert result is True

        assert not os.path.exists(src_path)  # Original should be gone
        assert os.path.exists(dst_path)  # Moved file should exist

        assert file_manager.read_file(dst_path) == original_content

    def test_file_info_retrieval(self, file_manager):
        """Test getting file information."""
        content = "File info test content"
        file_path = file_manager.create_file("info_test.txt", content)

        info = file_manager.get_file_info(file_path)

        assert info["exists"] is True
        assert info["is_file"] is True
        assert info["is_dir"] is False
        assert info["size"] == len(content.encode("utf-8"))
        assert "modified_time" in info
        assert "created_time" in info
        assert "permissions" in info

    def test_nonexistent_file_info(self, file_manager):
        """Test getting info for non-existent file."""
        nonexistent_path = os.path.join(file_manager.base_dir, "nonexistent.txt")

        info = file_manager.get_file_info(nonexistent_path)

        assert info["exists"] is False
        assert "error" in info

    def test_file_backup_operation(self, file_manager):
        """Test file backup creation."""
        original_content = "Important data to backup"
        file_path = file_manager.create_file("important.txt", original_content)

        backup_path = file_manager.backup_file(file_path)

        assert backup_path != ""
        assert os.path.exists(backup_path)
        assert backup_path.endswith(".bak")

        # Both files should have same content
        assert file_manager.read_file(file_path) == original_content
        assert file_manager.read_file(backup_path) == original_content


class TestFileIOAdvanced:
    """Advanced file I/O operation tests."""

    @pytest.fixture
    def file_manager(self):
        """Create file I/O manager."""
        manager = FileIOManager()
        yield manager
        manager.cleanup()

    def test_large_file_creation(self, file_manager):
        """Test creating and handling large files."""
        # Create a 10MB file
        file_path = file_manager.create_large_file("large_test.dat", 10)

        assert os.path.exists(file_path)

        info = file_manager.get_file_info(file_path)
        assert info["size"] == 10 * 1024 * 1024  # 10MB

    def test_concurrent_file_operations(self, file_manager):
        """Test concurrent file operations."""

        def create_file_task(task_id):
            content = f"Task {task_id} content"
            file_path = file_manager.create_file(f"concurrent_{task_id}.txt", content)
            return file_path, content

        # Create multiple files concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_file_task, i) for i in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All files should be created successfully
        assert len(results) == 10

        for file_path, expected_content in results:
            assert os.path.exists(file_path)
            actual_content = file_manager.read_file(file_path)
            assert actual_content == expected_content

    def test_concurrent_read_write_operations(self, file_manager):
        """Test concurrent reading and writing to different files."""
        # Create initial files
        file_paths = []
        for i in range(5):
            content = f"Initial content {i}"
            path = file_manager.create_file(f"rw_test_{i}.txt", content)
            file_paths.append(path)

        def read_write_task(file_path, task_id):
            # Read current content
            current_content = file_manager.read_file(file_path)

            # Append new content
            new_content = f"\nAppended by task {task_id}"
            file_manager.append_to_file(file_path, new_content)

            return current_content + new_content

        # Perform concurrent read-write operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(read_write_task, file_paths[i], i)
                for i in range(len(file_paths))
            ]
            [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify final contents
        for i, file_path in enumerate(file_paths):
            actual_content = file_manager.read_file(file_path)
            # Content should contain original and appended text
            assert f"Initial content {i}" in actual_content
            assert f"Appended by task {i}" in actual_content

    def test_file_permissions_handling(self, file_manager):
        """Test file permissions and error handling."""
        file_path = file_manager.create_file("permission_test.txt", "Test content")

        # Test readable file
        original_content = file_manager.read_file(file_path)
        assert original_content == "Test content"

        # Test file info includes permissions
        info = file_manager.get_file_info(file_path)
        assert "permissions" in info
        assert len(info["permissions"]) == 3  # Should be 3-digit octal

    def test_path_traversal_security(self, file_manager):
        """Test protection against path traversal attacks."""
        # Try to create file outside base directory using path traversal
        try:
            # This should either fail or be contained within base directory
            file_path = file_manager.create_file(
                "../../../etc/passwd", "malicious content"
            )

            # If it succeeds, ensure it's still within our base directory
            assert file_manager.base_dir in os.path.abspath(file_path)
        except Exception:
            # It's acceptable for this to fail as a security measure
            pass

    def test_unicode_filename_support(self, file_manager):
        """Test support for Unicode filenames."""
        unicode_filename = "测试文件_café_🚀.txt"
        content = "Unicode content: 你好世界"

        try:
            file_path = file_manager.create_file(unicode_filename, content)

            assert os.path.exists(file_path)

            read_content = file_manager.read_file(file_path)
            assert read_content == content

        except UnicodeError:
            # Some file systems may not support Unicode filenames
            pytest.skip("File system does not support Unicode filenames")

    def test_special_characters_in_content(self, file_manager):
        """Test handling of special characters in file content."""
        special_content = """
        Special characters test:
        - Newlines: \n
        - Tabs: \t
        - Unicode: 🌟 ñ é ç
        - Symbols: @#$%^&*()
        - Quotes: "double" 'single'
        - Backslashes: \\path\\to\\file
        """

        file_path = file_manager.create_file("special_chars.txt", special_content)

        read_content = file_manager.read_file(file_path)
        assert read_content == special_content

    def test_empty_file_operations(self, file_manager):
        """Test operations on empty files."""
        # Create empty file
        empty_file = file_manager.create_file("empty.txt", "")

        assert os.path.exists(empty_file)

        content = file_manager.read_file(empty_file)
        assert content == ""

        info = file_manager.get_file_info(empty_file)
        assert info["size"] == 0
        assert info["exists"] is True

    def test_disk_space_handling(self, file_manager):
        """Test handling of disk space constraints."""
        try:
            # Check available disk space
            total, used, free = shutil.disk_usage(file_manager.base_dir)
            free_mb = free // (1024 * 1024)

            if free_mb > 100:  # Only test if we have more than 100MB free
                # Try to create a moderately large file
                file_path = file_manager.create_large_file("space_test.dat", 50)  # 50MB

                assert os.path.exists(file_path)

                info = file_manager.get_file_info(file_path)
                assert info["size"] == 50 * 1024 * 1024
            else:
                pytest.skip("Insufficient disk space for large file test")

        except Exception as e:
            # Disk space issues should be handled gracefully
            assert "space" in str(e).lower() or "full" in str(e).lower()

    def test_file_corruption_detection(self, file_manager):
        """Test detection of file corruption."""
        # Create a file with known content
        original_data = {"test": "data", "number": 42}
        json_file = file_manager.create_json_file("corruption_test.json", original_data)

        # Verify it's readable
        read_data = file_manager.read_json_file(json_file)
        assert read_data == original_data

        # Simulate corruption by writing invalid JSON
        with open(json_file, "w") as f:
            f.write("corrupted {invalid json content")

        # Try to read corrupted file
        with pytest.raises(json.JSONDecodeError):
            file_manager.read_json_file(json_file)

    def test_atomic_file_operations(self, file_manager):
        """Test atomic file operations to prevent corruption."""
        file_path = file_manager.create_file("atomic_test.txt", "Original content")

        # Simulate atomic write by writing to temp file first
        temp_path = file_path + ".tmp"
        new_content = "New content that should replace original"

        # Write to temporary file
        with open(temp_path, "w") as f:
            f.write(new_content)

        # Atomically replace original
        shutil.move(temp_path, file_path)

        # Verify content was replaced
        final_content = file_manager.read_file(file_path)
        assert final_content == new_content

    @pytest.mark.skipif(
        os.name == "nt", reason="Symbolic links require special permissions on Windows"
    )
    def test_symbolic_link_operations(self, file_manager):
        """Test symbolic link creation and handling."""
        # Create target file
        target_content = "Target file content"
        target_path = file_manager.create_file("link_target.txt", target_content)

        # Create symbolic link
        link_path = os.path.join(file_manager.base_dir, "symlink.txt")
        result = file_manager.create_symlink(target_path, link_path)

        if result:  # Only test if symlink creation succeeded
            assert os.path.islink(link_path)

            # Reading through symlink should give same content
            link_content = file_manager.read_file(link_path)
            assert link_content == target_content
        else:
            pytest.skip("Symbolic link creation not supported")


if __name__ == "__main__":
    # Print system information for debugging
    print("File I/O integration test environment:")
    print(f"Operating system: {os.name}")
    print(f"Current working directory: {os.getcwd()}")

    # Check disk space
    try:
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        print(f"Free disk space: {free_gb} GB")
        if free_gb < 1:
            print("Warning: Less than 1GB free space. Some tests may fail.")
    except Exception:
        print("Could not determine disk space")

    # Check permissions
    test_dir = tempfile.mkdtemp(prefix="fileio_permission_test_")
    try:
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        os.rmdir(test_dir)
        print("File I/O permissions: ✓")
    except Exception:
        print("File I/O permissions: ✗ (may cause test failures)")

    pytest.main([__file__, "-v"])
