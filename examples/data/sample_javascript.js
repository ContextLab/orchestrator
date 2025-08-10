/**
 * Sample JavaScript code with optimization opportunities
 */

// Inefficient fibonacci implementation
function inefficientFibonacci(n) {
    if (n <= 1) return n;
    return inefficientFibonacci(n - 1) + inefficientFibonacci(n - 2);
}

// Class with hardcoded values and poor error handling
class DataProcessor {
    constructor() {
        this.batchSize = 100;  // Hardcoded value
        this.timeout = 5000;   // Hardcoded value
    }
    
    processData(data) {
        let result = [];
        let total = 0;
        
        // Inefficient: multiple loops
        for (let item of data) {
            if (item.active) {
                result.push(item);
            }
        }
        
        for (let item of data) {
            if (item.value) {
                total += item.value;  // No error handling for non-numeric values
            }
        }
        
        // Magic number
        if (result.length > 50) {
            console.log("Large dataset warning");
        }
        
        return { items: result, total: total };
    }
    
    transform(item) {
        // No validation
        return {
            id: item.id,
            name: item.name.toUpperCase(),
            timestamp: Date.now()
        };
    }
    
    processBatch(items) {
        // No error handling
        return items.map(item => this.transform(item));
    }
}

// Global variables (bad practice)
let globalCounter = 0;
const GLOBAL_CONFIG = { retries: 3, delay: 1000 };

function incrementGlobalCounter() {
    globalCounter++;
    return globalCounter;
}

// Function with inefficient string concatenation
function buildReport(items) {
    let report = "";
    
    // Inefficient string concatenation in loop
    for (let i = 0; i < items.length; i++) {
        report += `Item ${i}: ${items[i].name}\n`;
        report += `Status: ${items[i].status}\n`;
        report += `---\n`;
    }
    
    return report;
}

// Missing async/await best practices
function fetchData(url) {
    return fetch(url)
        .then(response => response.json())
        .then(data => data.results)
        .catch(error => {
            console.log("Error occurred"); // Poor error logging
            return [];
        });
}