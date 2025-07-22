#!/usr/bin/env python3
"""Clean up old checkpoint files."""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_checkpoints(days_to_keep=7, dry_run=True):
    """Clean up checkpoint files older than specified days."""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("No checkpoints directory found")
        return
    
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    
    total_files = 0
    files_to_delete = []
    total_size = 0
    
    for checkpoint_file in checkpoint_dir.glob("*.json"):
        total_files += 1
        file_stat = checkpoint_file.stat()
        
        if file_stat.st_mtime < cutoff_time:
            files_to_delete.append(checkpoint_file)
            total_size += file_stat.st_size
    
    print(f"Total checkpoint files: {total_files}")
    print(f"Files older than {days_to_keep} days: {len(files_to_delete)}")
    print(f"Total size to free: {total_size / 1024 / 1024:.2f} MB")
    
    if dry_run:
        print("\nDRY RUN - No files deleted")
        if files_to_delete:
            print("\nFiles that would be deleted:")
            for f in sorted(files_to_delete)[:10]:
                age_days = (time.time() - f.stat().st_mtime) / (24 * 60 * 60)
                print(f"  - {f.name} (age: {age_days:.1f} days)")
            if len(files_to_delete) > 10:
                print(f"  ... and {len(files_to_delete) - 10} more files")
    else:
        if files_to_delete:
            confirm = input(f"\nDelete {len(files_to_delete)} files? (yes/no): ")
            if confirm.lower() == 'yes':
                for f in files_to_delete:
                    f.unlink()
                print(f"Deleted {len(files_to_delete)} files")
            else:
                print("Cancelled")
        else:
            print("No files to delete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean up old checkpoint files")
    parser.add_argument("--days", type=int, default=7, help="Keep files newer than this many days")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry run)")
    
    args = parser.parse_args()
    cleanup_checkpoints(days_to_keep=args.days, dry_run=not args.execute)