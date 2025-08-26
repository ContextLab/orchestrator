class DataProcessor {
    constructor(config = {}) {
        this.batchSize = config.batchSize || 50;
        this.timeout = config.timeout || 30000;
        this.globalCounter = 0;
    }

    incrementCounter() {
        return ++this.globalCounter;
    }

    inefficientFibonacci(n, memo = {}) {
        if (n <= 1) return n;
        if (memo[n] !== undefined) return memo[n];
        memo[n] = this.inefficientFibonacci(n - 1, memo) + this.inefficientFibonacci(n - 2, memo);
        return memo[n];
    }

    processData(data) {
        const result = [];
        for (let item of data) {
            if (item && typeof item === 'object') {
                if (item.active) {
                    result.push(item);
                }
                if (typeof item.value === 'number') {
                    item.processedValue = item.value * 2;
                }
            }
        }
        return result;
    }

    transform(item) {
        if (!item || typeof item !== 'object') {
            throw new Error('Invalid item: must be an object');
        }
        const name = typeof item.name === 'string' ? item.name : '';
        const id = item.id !== undefined ? item.id : null;
        return {
            id: id,
            name: name.toUpperCase(),
            timestamp: Date.now()
        };
    }
}