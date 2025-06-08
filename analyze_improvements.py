#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

def load_data(file_path):
    """Load test cases from JSON file and convert to DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract input parameters and expected output
    rows = []
    for case in data:
        row = {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'expected_output': case['expected_output']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def calculate_reimbursement_original(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Original implementation with average error of $271.92
    """
    # Initialize reimbursement
    reimbursement = 0.0
    
    # Base per diem calculation
    base_per_diem = 95.0
    
    # Apply per diem based on trip duration
    if trip_duration_days <= 3:
        per_diem = base_per_diem * trip_duration_days
    elif 4 <= trip_duration_days <= 7:
        per_diem = base_per_diem * trip_duration_days * 0.98
    else:
        per_diem = base_per_diem * trip_duration_days * 0.85
    
    reimbursement += per_diem
    
    # Mileage calculation
    mileage_rate_tier1 = 0.58  # First 100 miles
    mileage_rate_tier2 = 0.50  # Next 200 miles
    mileage_rate_tier3 = 0.40  # Next 300 miles
    mileage_rate_tier4 = 0.30  # Beyond 600 miles
    
    if miles_traveled <= 100:
        mileage_reimbursement = miles_traveled * mileage_rate_tier1
    elif miles_traveled <= 300:
        mileage_reimbursement = 100 * mileage_rate_tier1 + (miles_traveled - 100) * mileage_rate_tier2
    elif miles_traveled <= 600:
        mileage_reimbursement = 100 * mileage_rate_tier1 + 200 * mileage_rate_tier2 + (miles_traveled - 300) * mileage_rate_tier3
    else:
        mileage_reimbursement = 100 * mileage_rate_tier1 + 200 * mileage_rate_tier2 + 300 * mileage_rate_tier3 + (miles_traveled - 600) * mileage_rate_tier4
    
    if miles_traveled > 800:
        mileage_reimbursement = min(mileage_reimbursement, 350)
    
    reimbursement += mileage_reimbursement
    
    # Receipt processing
    if total_receipts_amount < 20:
        receipt_adjustment = total_receipts_amount * 0.5
    elif total_receipts_amount < 100:
        receipt_adjustment = 10 + (total_receipts_amount - 20) * 0.6
    elif total_receipts_amount < 600:
        receipt_adjustment = 58 + (total_receipts_amount - 100) * 0.4
    elif total_receipts_amount < 1000:
        receipt_adjustment = 258 + (total_receipts_amount - 600) * 0.3
    else:
        receipt_adjustment = 378 + (total_receipts_amount - 1000) * 0.15
    
    if total_receipts_amount > 2000:
        receipt_adjustment = min(receipt_adjustment, 500)
    
    if trip_duration_days >= 8:
        receipt_adjustment *= 0.7
    
    reimbursement += receipt_adjustment
    
    # Efficiency bonus
    miles_per_day = miles_traveled / trip_duration_days if trip_duration_days > 0 else 0
    
    if 180 <= miles_per_day <= 220:
        efficiency_bonus = 25.0 * min(trip_duration_days, 5)
        reimbursement += efficiency_bonus
    
    # Date-related factors (simplified for this analysis)
    reimbursement += np.random.uniform(-50, 50)  # Approximate the date effects
    
    # Caps and floors
    if trip_duration_days >= 10:
        reimbursement = min(reimbursement, 1900)
    
    if miles_traveled > 800 and total_receipts_amount > 1500:
        reimbursement = min(reimbursement, 1800)
    
    return round(reimbursement, 2)

def calculate_reimbursement_improved(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Improved implementation with average error of $266.87
    """
    # Initialize reimbursement
    reimbursement = 0.0
    
    # Base per diem calculation
    base_per_diem = 95.0
    
    # Apply per diem based on trip duration
    if trip_duration_days <= 3:
        per_diem = base_per_diem * trip_duration_days
    elif 4 <= trip_duration_days <= 7:
        per_diem = base_per_diem * trip_duration_days * 0.98
    else:
        per_diem = base_per_diem * trip_duration_days * 0.85
    
    reimbursement += per_diem
    
    # Mileage calculation
    mileage_rate_tier1 = 0.58  # First 100 miles
    mileage_rate_tier2 = 0.50  # Next 200 miles
    mileage_rate_tier3 = 0.40  # Next 300 miles
    mileage_rate_tier4 = 0.30  # Beyond 600 miles
    
    if miles_traveled <= 100:
        mileage_reimbursement = miles_traveled * mileage_rate_tier1
    elif miles_traveled <= 300:
        mileage_reimbursement = 100 * mileage_rate_tier1 + (miles_traveled - 100) * mileage_rate_tier2
    elif miles_traveled <= 600:
        mileage_reimbursement = 100 * mileage_rate_tier1 + 200 * mileage_rate_tier2 + (miles_traveled - 300) * mileage_rate_tier3
    else:
        mileage_reimbursement = 100 * mileage_rate_tier1 + 200 * mileage_rate_tier2 + 300 * mileage_rate_tier3 + (miles_traveled - 600) * mileage_rate_tier4
    
    # Special case for high mileage in 7-9 day trips
    if 7 <= trip_duration_days <= 9 and miles_traveled > 1000:
        mileage_reimbursement = 600  # Higher value for these specific cases
    # Regular cap for very high mileage in other cases
    elif miles_traveled > 800:
        mileage_reimbursement = min(mileage_reimbursement, 350)
    
    reimbursement += mileage_reimbursement
    
    # Receipt processing
    if total_receipts_amount < 20:
        receipt_adjustment = total_receipts_amount * 0.5
    elif total_receipts_amount < 100:
        receipt_adjustment = 10 + (total_receipts_amount - 20) * 0.6
    elif total_receipts_amount < 600:
        receipt_adjustment = 58 + (total_receipts_amount - 100) * 0.4
    elif total_receipts_amount < 1000:
        receipt_adjustment = 258 + (total_receipts_amount - 600) * 0.3
    else:
        receipt_adjustment = 378 + (total_receipts_amount - 1000) * 0.15
    
    # Special case for high receipts in 7-9 day trips
    if 7 <= trip_duration_days <= 9 and total_receipts_amount > 1000:
        receipt_adjustment = 500 + (total_receipts_amount - 1000) * 0.3
    # Regular cap for extremely high receipts in other cases
    elif total_receipts_amount > 2000:
        receipt_adjustment = min(receipt_adjustment, 500)
    
    # For long trips (8+ days), reduce receipt contribution
    # But not for the special 7-9 day high-receipt cases
    if trip_duration_days >= 8 and not (7 <= trip_duration_days <= 9 and total_receipts_amount > 1000):
        receipt_adjustment *= 0.7
    
    reimbursement += receipt_adjustment
    
    # Efficiency bonus
    miles_per_day = miles_traveled / trip_duration_days if trip_duration_days > 0 else 0
    
    if 180 <= miles_per_day <= 220:
        efficiency_bonus = 25.0 * min(trip_duration_days, 5)
        reimbursement += efficiency_bonus
    
    # Special case for 7-9 day trips with high mileage and high receipts
    if 7 <= trip_duration_days <= 9 and miles_traveled > 900 and total_receipts_amount > 900:
        high_combo_bonus = 800
        reimbursement += high_combo_bonus
    
    # Date-related factors (simplified for this analysis)
    reimbursement += np.random.uniform(-50, 50)  # Approximate the date effects
    
    # Caps and floors
    if trip_duration_days >= 10:
        reimbursement = min(reimbursement, 1900)
    
    if miles_traveled > 800 and total_receipts_amount > 1500 and not (7 <= trip_duration_days <= 9):
        reimbursement = min(reimbursement, 1800)
    
    return round(reimbursement, 2)

def analyze_improvements(df):
    """Analyze the impact of our improvements on different categories of trips."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Add calculated reimbursements
    df['original_calc'] = df.apply(lambda row: calculate_reimbursement_original(
        row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']), axis=1)
    
    df['improved_calc'] = df.apply(lambda row: calculate_reimbursement_improved(
        row['trip_duration_days'], row['miles_traveled'], row['total_receipts_amount']), axis=1)
    
    # Calculate errors
    df['original_error'] = abs(df['expected_output'] - df['original_calc'])
    df['improved_error'] = abs(df['expected_output'] - df['improved_calc'])
    df['error_change'] = df['original_error'] - df['improved_error']
    
    # Overall error metrics
    original_mae = mean_absolute_error(df['expected_output'], df['original_calc'])
    improved_mae = mean_absolute_error(df['expected_output'], df['improved_calc'])
    
    print(f"Original MAE: ${original_mae:.2f}")
    print(f"Improved MAE: ${improved_mae:.2f}")
    print(f"Improvement: ${original_mae - improved_mae:.2f} ({(original_mae - improved_mae) / original_mae * 100:.2f}%)")
    
    # Analyze specific categories
    print("\n=== Analysis by Trip Categories ===")
    
    # Define categories
    df['category'] = 'Other'
    
    # 7-9 day trips with high mileage and high receipts
    mask_7_9_high = (
        (df['trip_duration_days'] >= 7) & 
        (df['trip_duration_days'] <= 9) & 
        (df['miles_traveled'] > 900) & 
        (df['total_receipts_amount'] > 900)
    )
    df.loc[mask_7_9_high, 'category'] = '7-9 days, high miles, high receipts'
    
    # 7-9 day trips with high mileage
    mask_7_9_high_miles = (
        (df['trip_duration_days'] >= 7) & 
        (df['trip_duration_days'] <= 9) & 
        (df['miles_traveled'] > 900) & 
        (df['total_receipts_amount'] <= 900)
    )
    df.loc[mask_7_9_high_miles, 'category'] = '7-9 days, high miles, low receipts'
    
    # 7-9 day trips with high receipts
    mask_7_9_high_receipts = (
        (df['trip_duration_days'] >= 7) & 
        (df['trip_duration_days'] <= 9) & 
        (df['miles_traveled'] <= 900) & 
        (df['total_receipts_amount'] > 900)
    )
    df.loc[mask_7_9_high_receipts, 'category'] = '7-9 days, low miles, high receipts'
    
    # Long trips (10+ days)
    mask_long = (df['trip_duration_days'] >= 10)
    df.loc[mask_long, 'category'] = '10+ days'
    
    # Analyze error by category
    category_analysis = df.groupby('category').agg({
        'original_error': 'mean',
        'improved_error': 'mean',
        'error_change': 'mean',
        'trip_duration_days': 'count'
    }).rename(columns={'trip_duration_days': 'count'})
    
    print(category_analysis)
    
    # Plot error improvement by category
    plt.figure(figsize=(12, 6))
    sns.barplot(x='category', y='error_change', data=df)
    plt.title('Error Improvement by Trip Category')
    plt.xlabel('Trip Category')
    plt.ylabel('Error Reduction ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('error_improvement_by_category.png')
    plt.close()
    
    # Analyze the top improved cases
    print("\n=== Top 10 Most Improved Cases ===")
    top_improved = df.sort_values('error_change', ascending=False).head(10)
    for _, row in top_improved.iterrows():
        print(f"Case with {row['trip_duration_days']} days, {row['miles_traveled']:.0f} miles, ${row['total_receipts_amount']:.2f} receipts:")
        print(f"  Expected: ${row['expected_output']:.2f}")
        print(f"  Original: ${row['original_calc']:.2f} (Error: ${row['original_error']:.2f})")
        print(f"  Improved: ${row['improved_calc']:.2f} (Error: ${row['improved_error']:.2f})")
        print(f"  Improvement: ${row['error_change']:.2f}")
        print()
    
    # Analyze the cases that got worse
    print("\n=== Top 10 Cases That Got Worse ===")
    got_worse = df[df['error_change'] < 0].sort_values('error_change', ascending=True).head(10)
    for _, row in got_worse.iterrows():
        print(f"Case with {row['trip_duration_days']} days, {row['miles_traveled']:.0f} miles, ${row['total_receipts_amount']:.2f} receipts:")
        print(f"  Expected: ${row['expected_output']:.2f}")
        print(f"  Original: ${row['original_calc']:.2f} (Error: ${row['original_error']:.2f})")
        print(f"  Improved: ${row['improved_calc']:.2f} (Error: ${row['improved_error']:.2f})")
        print(f"  Worsened by: ${-row['error_change']:.2f}")
        print()
    
    # Analyze error distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['original_error'], bins=50, alpha=0.5, label='Original')
    plt.hist(df['improved_error'], bins=50, alpha=0.5, label='Improved')
    plt.title('Error Distribution')
    plt.xlabel('Absolute Error ($)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('error_distribution.png')
    plt.close()
    
    # Analyze error by trip duration
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='trip_duration_days', y='improved_error', data=df)
    plt.title('Error by Trip Duration')
    plt.xlabel('Trip Duration (days)')
    plt.ylabel('Absolute Error ($)')
    plt.savefig('error_by_duration.png')
    plt.close()
    
    # Analyze error by mileage and receipt amount
    plt.figure(figsize=(10, 8))
    plt.scatter(df['miles_traveled'], df['total_receipts_amount'], 
                c=df['improved_error'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Absolute Error ($)')
    plt.title('Error by Mileage and Receipt Amount')
    plt.xlabel('Miles Traveled')
    plt.ylabel('Receipt Amount ($)')
    plt.savefig('error_by_miles_receipts.png')
    plt.close()
    
    # Return the dataframe with calculations for further analysis
    return df

def main():
    # Load data
    print("Loading data...")
    df = load_data('public_cases.json')
    print(f"Loaded {len(df)} test cases")
    
    # Analyze improvements
    print("\nAnalyzing improvements...")
    df_with_calcs = analyze_improvements(df)
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()