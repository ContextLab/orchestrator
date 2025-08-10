/**
 * Sample Java code with optimization opportunities
 */

import java.util.*;
import java.util.stream.Collectors;

public class DataProcessor {
    
    // Hardcoded configuration (should be configurable)
    private static final int BATCH_SIZE = 100;
    private static final int TIMEOUT = 5000;
    
    // Global state (not thread-safe)
    private static int globalCounter = 0;
    
    // Inefficient fibonacci implementation
    public static int inefficientFibonacci(int n) {
        if (n <= 1) return n;
        return inefficientFibonacci(n - 1) + inefficientFibonacci(n - 2);
    }
    
    // Multiple passes through data (inefficient)
    public List<Map<String, Object>> processData(List<Map<String, Object>> data) {
        List<Map<String, Object>> activeItems = new ArrayList<>();
        
        // First pass - filter active items
        for (Map<String, Object> item : data) {
            Boolean active = (Boolean) item.get("active");
            if (active != null && active) {
                activeItems.add(item);
            }
        }
        
        // Second pass - calculate totals
        double total = 0;
        for (Map<String, Object> item : data) {
            Object value = item.get("value");
            if (value instanceof Number) {
                total += ((Number) value).doubleValue();
            }
        }
        
        // Third pass - transform items
        List<Map<String, Object>> result = new ArrayList<>();
        for (Map<String, Object> item : activeItems) {
            result.add(transformItem(item));
        }
        
        // Magic number usage
        if (result.size() > 50) {
            System.out.println("Warning: Large dataset");
        }
        
        return result;
    }
    
    // No error handling or validation
    public Map<String, Object> transformItem(Map<String, Object> item) {
        Map<String, Object> transformed = new HashMap<>();
        
        // Potential NullPointerException
        transformed.put("id", item.get("id"));
        transformed.put("name", ((String) item.get("name")).toUpperCase());
        transformed.put("timestamp", System.currentTimeMillis());
        
        return transformed;
    }
    
    // Inefficient string concatenation
    public String buildReport(List<Map<String, Object>> items) {
        String report = "";
        
        for (int i = 0; i < items.size(); i++) {
            Map<String, Object> item = items.get(i);
            report += "Item " + i + ": " + item.get("name") + "\n";
            report += "Status: " + item.get("status") + "\n";
            report += "---\n";
        }
        
        return report;
    }
    
    // Not thread-safe global state modification
    public static synchronized int incrementCounter() {
        globalCounter++;
        return globalCounter;
    }
    
    // Missing proper exception handling
    public void processBatch(List<Map<String, Object>> batch) {
        for (Map<String, Object> item : batch) {
            Map<String, Object> processed = transformItem(item);
            // No null checks or exception handling
            System.out.println("Processed: " + processed.get("name"));
        }
    }
    
    // Inefficient collection usage
    public List<String> getActiveItemNames(List<Map<String, Object>> items) {
        List<String> names = new ArrayList<>();
        
        for (Map<String, Object> item : items) {
            Boolean active = (Boolean) item.get("active");
            if (active != null && active) {
                String name = (String) item.get("name");
                if (name != null && !names.contains(name)) {  // O(n) contains check
                    names.add(name);
                }
            }
        }
        
        return names;
    }
}