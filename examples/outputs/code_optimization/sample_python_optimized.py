def calculate_sum(list_of_numbers):
    try:
        return sum(list_of_numbers)
    except TypeError:
        print("Error: List elements should be numbers")
        return None
    except Exception as e:
        print(f"An error occured: {e}")
        return None

print(calculate_sum([1, 2, 3, 4, 5]))
print(calculate_sum([1, 'a', 3, 4, 5]))