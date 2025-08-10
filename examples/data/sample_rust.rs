// Sample Rust code with optimization opportunities

use std::collections::HashMap;
use std::time::Instant;

// Inefficient fibonacci implementation
fn inefficient_fibonacci(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    return inefficient_fibonacci(n - 1) + inefficient_fibonacci(n - 2);
}

// Struct with hardcoded values and poor error handling
pub struct DataProcessor {
    batch_size: usize,  // Hardcoded value
    timeout: u64,       // Hardcoded value
}

impl DataProcessor {
    pub fn new() -> Self {
        DataProcessor {
            batch_size: 100,
            timeout: 5000,
        }
    }
    
    // Multiple passes through data (inefficient)
    pub fn process_data(&self, data: &Vec<HashMap<String, String>>) -> Vec<HashMap<String, String>> {
        let mut result = Vec::new();
        
        // First pass - filter active items
        for item in data {
            if let Some(active) = item.get("active") {
                if active == "true" {
                    result.push(item.clone());
                }
            }
        }
        
        // Second pass - transform items (inefficient)
        let mut transformed = Vec::new();
        for item in &result {
            let mut new_item = HashMap::new();
            // No error handling for missing keys
            new_item.insert("id".to_string(), item["id"].clone());
            new_item.insert("name".to_string(), item["name"].to_uppercase());
            new_item.insert("timestamp".to_string(), format!("{:?}", Instant::now()));
            transformed.push(new_item);
        }
        
        // Magic number usage
        if transformed.len() > 50 {
            println!("Warning: Large dataset");
        }
        
        transformed
    }
    
    // No proper error handling
    pub fn transform_item(&self, item: &HashMap<String, String>) -> HashMap<String, String> {
        let mut transformed = HashMap::new();
        
        // Potential panic if keys don't exist
        transformed.insert("id".to_string(), item["id"].clone());
        transformed.insert("name".to_string(), item["name"].to_uppercase());
        transformed.insert("processed".to_string(), "true".to_string());
        
        transformed
    }
    
    // Inefficient string building
    pub fn build_report(&self, items: &Vec<HashMap<String, String>>) -> String {
        let mut report = String::new();
        
        // Inefficient string concatenation
        for (i, item) in items.iter().enumerate() {
            report.push_str(&format!("Item {}: {}\n", i, item.get("name").unwrap_or(&"Unknown".to_string())));
            report.push_str(&format!("Status: {}\n", item.get("status").unwrap_or(&"N/A".to_string())));
            report.push_str("---\n");
        }
        
        report
    }
}

// Global variable (bad practice in Rust)
static mut GLOBAL_COUNTER: u32 = 0;

pub fn increment_counter() -> u32 {
    unsafe {
        GLOBAL_COUNTER += 1;
        GLOBAL_COUNTER
    }
}

// Function with poor error handling
pub fn process_file(filename: &str) -> String {
    // No proper error handling
    std::fs::read_to_string(filename).unwrap()
}