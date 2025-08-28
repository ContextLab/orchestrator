# Sample Python code for optimization testing
def inefficient_function(data):
    """A deliberately inefficient function for testing optimization."""
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                result.append(data[i] + data[j])
    return result

def another_function():
    """Another simple function."""
    x = 1
    y = 2
    z = x + y
    return z

if __name__ == "__main__":
    test_data = [1, 2, 3, 4, 5]
    result = inefficient_function(test_data)
    print(f"Result: {result}")