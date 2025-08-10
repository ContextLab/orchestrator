const fetch = require('node-fetch');

class DataProcessor {
  constructor(batchSize = 100, timeout = 1000) {
    this.batchSize = batchSize;
    this.timeout = timeout;
  }

  efficientFibonacci(n, memo = {}) {
    if (n <= 0) return 0;
    else if (n == 1) return 1;
    else if (!memo[n]) {
      memo[n] = this.efficientFibonacci(n - 1, memo) + this.efficientFibonacci(n - 2, memo);
    }
    return memo[n];
  }

  async fetchData(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`An error has occurred: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error(`Fetch Error: ${error}`);
      throw error;
    }
  }

  transform(data) {
    if (!Array.isArray(data)) throw new Error('Invalid data. Expected an array.');
    return data.map(item => Number(item));
  }

  processData(data) {
    if (!data || !data.length) return { total: 0, average: 0 };

    const total = data.reduce((sum, number) => {
      if (isNaN(number)) throw new Error('Invalid data. Expected a numeric value.');
      return sum + number;
    }, 0);

    const average = total / data.length;

    return { total, average };
  }

  async processBatch(batch, callback) {
    try {
      const transformedData = this.transform(batch);
      const result = this.processData(transformedData);
      callback(null, result);
    } catch (error) {
      console.error(`Process Error: ${error}`);
      callback(error);
    }
  }

  async processAll(data, callback) {
    const batches = this.createBatches(data, this.batchSize);
    for (const batch of batches) {
      setTimeout(() => {
        this.processBatch(batch, callback);
      }, this.timeout);
    }
  }

  createBatches(data, batchSize) {
    let result = [];
    for (let i = 0; i < data.length; i += batchSize) {
      result.push(data.slice(i, i + batchSize));
    }
    return result;
  }
}