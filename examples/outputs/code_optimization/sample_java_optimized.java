import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class OptimizedCode {
    private static final int BATCH_SIZE = 100; // Configurable
    private static final int TIMEOUT = 5000;   // Configurable
    private static final int PROCESS_THRESHOLD = 50; // Named constant for magic number
    private AtomicInteger globalCounter = new AtomicInteger(0);
    private Map<Integer, Integer> memo = new ConcurrentHashMap<>();

    // Efficient Fibonacci Implementation with memoization
    public int efficientFibonacci(int n) {
        if (n <= 1) return n;
        if (!memo.containsKey(n)) {
            memo.put(n, efficientFibonacci(n - 1) + efficientFibonacci(n - 2));
        }
        return memo.get(n);
    }

    // Single Pass Through Data
    public void processData(List<Data> data) {
        for (Data item : data) {
            if (item.getValue() > PROCESS_THRESHOLD) {
                transformItem(item);
            }
        }
    }

    // Efficient String Concatenation
    public String buildReport(List<Data> data) {
        StringBuilder report = new StringBuilder();
        for (Data item : data) {
            report.append(item.toString()).append("\n");
        }
        return report.toString();
    }

    // Efficient Collection Usage
    public List<String> getActiveItemNames(List<Item> items, Set<String> activeNames) {
        List<String> activeItemNames = new ArrayList<>();
        for (Item item : items) {
            if (activeNames.contains(item.getName())) {
                activeItemNames.add(item.getName());
            }
        }
        return activeItemNames;
    }

    // Thread-Safe Global State Modification
    public void incrementCounter() {
        globalCounter.incrementAndGet();
    }

    // Proper Error Handling and Validation
    public void transformItem(Data item) {
        if (item == null) {
            throw new IllegalArgumentException("Data item cannot be null");
        }
        // Transformation logic here
    }

    // Proper Exception Handling
    public void processBatch(List<Data> batch) {
        try {
            // Batch processing logic here
        } catch (Exception e) {
            // Log and handle exception
        }
    }
}

class Data {
    public int getValue() {
        // Implementation here
        return 0;
    }
}

class Item {
    public String getName() {
        // Implementation here
        return "";
    }
}