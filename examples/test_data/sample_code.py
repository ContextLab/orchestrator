# Sample Python code for optimization testing

def inefficient_sum(numbers):
    """Calculate sum using inefficient method."""
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total

def find_duplicates_slow(items):
    """Find duplicates using O(n²) algorithm."""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

def nested_loops_example(matrix):
    """Inefficient matrix processing."""
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0:
                row.append(matrix[i][j] * 2)
            else:
                row.append(0)
        result.append(row)
    return result

class BadClassExample:
    """Class with poor design patterns."""
    
    def __init__(self):
        self.data = []
        self.cache = {}
        
    def add_item(self, item):
        # Inefficient: checking duplicates every time
        if item not in self.data:
            self.data.append(item)
        # Not using the cache properly
        self.cache = {}
        
    def get_item_count(self):
        # Recalculating every time
        count = 0
        for item in self.data:
            count += 1
        return count
    
    def find_item(self, target):
        # Linear search instead of using better data structures
        for i, item in enumerate(self.data):
            if item == target:
                return i
        return -1