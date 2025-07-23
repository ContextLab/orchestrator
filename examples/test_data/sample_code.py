"""Sample code for testing code analysis pipelines."""

import time
from typing import List, Dict, Any

def inefficient_fibonacci(n: int) -> int:
    """Calculate fibonacci number recursively (inefficient)."""
    if n <= 1:
        return n
    return inefficient_fibonacci(n - 1) + inefficient_fibonacci(n - 2)

def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a list of records."""
    result = {"total": 0, "items": []}
    
    # Inefficient: multiple passes through data
    for item in data:
        if item.get("active"):
            result["items"].append(item)
    
    for item in data:
        if item.get("value"):
            result["total"] += item["value"]
    
    # Code smell: magic number
    if len(result["items"]) > 100:
        print("Warning: Large dataset")
    
    return result

class DataProcessor:
    def __init__(self):
        # Issue: hardcoded configuration
        self.batch_size = 50
        self.timeout = 30
    
    def process_batch(self, items):
        # Issue: no error handling
        processed = []
        for item in items:
            processed.append(self.transform(item))
        return processed
    
    def transform(self, item):
        # Issue: no validation
        return {
            "id": item["id"],
            "name": item["name"].upper(),
            "timestamp": time.time()
        }

# Issue: global variable
GLOBAL_COUNTER = 0

def increment_counter():
    global GLOBAL_COUNTER
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER