#!/usr/bin/env python3
"""Generate realistic sample dataset for modular analysis pipeline."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_sample_data(num_rows=1000):
    """Generate realistic sales/metrics data with patterns."""
    
    # Generate date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_rows)]
    
    # Product categories
    categories = ['Electronics', 'Home & Garden', 'Clothing', 'Sports', 'Books', 'Toys']
    category_weights = [0.35, 0.25, 0.20, 0.10, 0.07, 0.03]  # Electronics dominates
    
    # Regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate base sales with trend and seasonality
    trend = np.linspace(100, 150, num_rows)  # Upward trend
    seasonality = 20 * np.sin(np.arange(num_rows) * 2 * np.pi / 365)  # Annual seasonality
    weekly_pattern = 10 * np.sin(np.arange(num_rows) * 2 * np.pi / 7)  # Weekly pattern
    noise = np.random.normal(0, 10, num_rows)
    
    base_sales = trend + seasonality + weekly_pattern + noise
    base_sales = np.maximum(base_sales, 10)  # Ensure positive values
    
    # Generate data
    data = []
    sample_comments = [
        "Great product, highly recommend!",
        "Good value for money",
        "Disappointed with the quality",
        "Excellent service and fast delivery",
        "Product as described, satisfied",
        "Not worth the price",
        "Amazing! Will buy again",
        "Average product, nothing special",
        "Exceeded my expectations",
        "Poor customer service experience",
        "Love it! Perfect for my needs",
        "Broke after a week of use",
        "Best purchase this year",
        "Okay product but overpriced",
        "Fantastic quality and design",
        "Would not recommend to others",
        "Solid product, works as expected",
        "Shipping was delayed but product is good",
        "Five stars! Absolutely perfect",
        "Mediocre at best",
    ]
    
    for i in range(num_rows):
        date = dates[i]
        
        # Add day-of-week effects
        day_multiplier = 1.2 if date.weekday() in [4, 5, 6] else 1.0  # Higher on Fri-Sun
        
        # Add month effects (higher in Nov-Dec for holidays)
        month_multiplier = 1.3 if date.month in [11, 12] else 1.0
        
        # Category selection
        category = np.random.choice(categories, p=category_weights)
        
        # Category-specific price ranges
        if category == 'Electronics':
            base_price = np.random.uniform(50, 500)
        elif category == 'Home & Garden':
            base_price = np.random.uniform(20, 200)
        elif category == 'Clothing':
            base_price = np.random.uniform(15, 150)
        elif category == 'Sports':
            base_price = np.random.uniform(25, 250)
        elif category == 'Books':
            base_price = np.random.uniform(10, 50)
        else:  # Toys
            base_price = np.random.uniform(10, 100)
        
        # Calculate units sold
        units = int(base_sales[i] * day_multiplier * month_multiplier * np.random.uniform(0.8, 1.2))
        units = max(1, units)
        
        # Calculate revenue
        revenue = units * base_price * np.random.uniform(0.95, 1.05)  # Small price variations
        
        # Customer satisfaction (correlated with comments)
        satisfaction = np.random.uniform(1, 5)
        
        # Select comment based on satisfaction
        if satisfaction > 4:
            comment = random.choice([c for c in sample_comments if 'great' in c.lower() or 'excellent' in c.lower() or 'love' in c.lower() or 'best' in c.lower() or 'amazing' in c.lower()])
        elif satisfaction < 2:
            comment = random.choice([c for c in sample_comments if 'poor' in c.lower() or 'disappointed' in c.lower() or 'not' in c.lower() or 'broke' in c.lower()])
        else:
            comment = random.choice([c for c in sample_comments if 'good' in c.lower() or 'average' in c.lower() or 'okay' in c.lower() or 'satisfied' in c.lower()])
        
        # Add row
        data.append({
            'timestamp': date.strftime('%Y-%m-%d'),
            'date': date.strftime('%Y-%m-%d'),
            'sales': units,
            'revenue': round(revenue, 2),
            'product_category': category,
            'region': random.choice(regions),
            'average_price': round(base_price, 2),
            'units_sold': units,
            'customer_satisfaction': round(satisfaction, 1),
            'comments': comment,
            'conversion_rate': round(np.random.uniform(0.01, 0.05), 3),
            'return_rate': round(np.random.uniform(0.01, 0.10), 3),
            'inventory_level': int(np.random.uniform(100, 1000)),
            'marketing_spend': round(np.random.uniform(100, 1000), 2),
            'discount_percentage': round(np.random.uniform(0, 0.3), 2),
            'shipping_time_days': int(np.random.uniform(1, 7)),
        })
    
    return pd.DataFrame(data)

def main():
    """Generate and save sample data."""
    
    # Create output directory
    output_dir = 'examples/outputs/modular_analysis/input'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    print("Generating sample dataset...")
    df = generate_sample_data(1000)
    
    # Save as CSV
    output_path = os.path.join(output_dir, 'dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"- Rows: {len(df)}")
    print(f"- Columns: {len(df.columns)}")
    print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"- Total revenue: ${df['revenue'].sum():,.2f}")
    print(f"- Average satisfaction: {df['customer_satisfaction'].mean():.2f}")
    print(f"- Product categories: {', '.join(df['product_category'].unique())}")
    print(f"- Regions: {', '.join(df['region'].unique())}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

if __name__ == '__main__':
    main()