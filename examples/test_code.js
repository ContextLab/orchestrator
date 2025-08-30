function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Test with multiple values
console.log(fibonacci(10));
console.log(fibonacci(20));