#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

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

def check_for_hidden_date_patterns(df):
    """
    Check if there might be hidden date-related patterns in the data.
    
    Since we don't have actual dates, we'll look for patterns that might
    suggest date-based calculations are happening.
    """
    print("\n=== Checking for Hidden Date Patterns ===")
    
    # 1. Check if case index (order in the file) affects reimbursement
    df['case_index'] = range(len(df))
    
    # Plot reimbursement by case index
    plt.figure(figsize=(12, 6))
    plt.scatter(df['case_index'], df['expected_output'], alpha=0.5)
    plt.title('Reimbursement by Case Index')
    plt.xlabel('Case Index (Order in File)')
    plt.ylabel('Reimbursement Amount ($)')
    plt.savefig('reimbursement_by_case_index.png')
    plt.close()
    
    # Check correlation
    correlation = df['case_index'].corr(df['expected_output'])
    print(f"Correlation between case index and reimbursement: {correlation:.4f}")
    
    # 2. Check if there are cyclical patterns in the data
    # We'll use the case index as a proxy for time and look for cycles
    
    # Group by every 30 cases (simulating monthly cycles)
    df['month_proxy'] = df['case_index'] // 30
    monthly_avg = df.groupby('month_proxy')['expected_output'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_avg.plot(kind='bar')
    plt.title('Average Reimbursement by "Month" (Every 30 Cases)')
    plt.xlabel('Month Proxy (Every 30 Cases)')
    plt.ylabel('Average Reimbursement ($)')
    plt.savefig('reimbursement_by_month_proxy.png')
    plt.close()
    
    # Group by every 7 cases (simulating weekly cycles)
    df['day_of_week_proxy'] = df['case_index'] % 7
    daily_avg = df.groupby('day_of_week_proxy')['expected_output'].mean()
    
    plt.figure(figsize=(12, 6))
    daily_avg.plot(kind='bar')
    plt.title('Average Reimbursement by "Day of Week" (Case Index % 7)')
    plt.xlabel('Day of Week Proxy (Case Index % 7)')
    plt.ylabel('Average Reimbursement ($)')
    plt.savefig('reimbursement_by_day_proxy.png')
    plt.close()
    
    print(f"Average reimbursement by 'day of week':")
    for day, avg in daily_avg.items():
        print(f"  Day {day}: ${avg:.2f}")
    
    # 3. Check for lunar cycle patterns (every 29.5 days)
    df['moon_phase_proxy'] = df['case_index'] % 30
    lunar_avg = df.groupby('moon_phase_proxy')['expected_output'].mean()
    
    plt.figure(figsize=(15, 6))
    lunar_avg.plot(kind='line')
    plt.title('Average Reimbursement by "Moon Phase" (Case Index % 30)')
    plt.xlabel('Moon Phase Proxy (Case Index % 30)')
    plt.ylabel('Average Reimbursement ($)')
    plt.savefig('reimbursement_by_moon_proxy.png')
    plt.close()
    
    # 4. Check for quarter-end effects
    df['quarter_proxy'] = (df['case_index'] % 90) // 30
    df['is_quarter_end'] = (df['case_index'] % 90) >= 60
    
    quarter_avg = df.groupby('quarter_proxy')['expected_output'].mean()
    quarter_end_avg = df.groupby('is_quarter_end')['expected_output'].mean()
    
    print(f"\nAverage reimbursement by 'quarter month':")
    for month, avg in quarter_avg.items():
        print(f"  Month {month+1} of quarter: ${avg:.2f}")
    
    print(f"\nQuarter-end effect:")
    for is_end, avg in quarter_end_avg.items():
        print(f"  {'Last month of quarter' if is_end else 'First two months of quarter'}: ${avg:.2f}")
    
    # 5. Check for holiday effects (assuming holidays might be encoded in some way)
    # We'll use case index % 365 to simulate day of year
    df['day_of_year_proxy'] = df['case_index'] % 365
    
    # Common US holidays (approximate days of year)
    holidays = {
        "New Year's Day": 1,
        "MLK Day": 15,
        "Presidents Day": 46,
        "Memorial Day": 150,
        "Independence Day": 185,
        "Labor Day": 246,
        "Columbus Day": 285,
        "Veterans Day": 315,
        "Thanksgiving": 330,
        "Christmas": 359
    }
    
    # Check if being near a holiday affects reimbursement
    holiday_effects = []
    for holiday, day in holidays.items():
        # Consider cases within 3 days of the holiday
        holiday_cases = df[(df['day_of_year_proxy'] >= day - 3) & (df['day_of_year_proxy'] <= day + 3)]
        non_holiday_cases = df[~((df['day_of_year_proxy'] >= day - 3) & (df['day_of_year_proxy'] <= day + 3))]
        
        if len(holiday_cases) > 0:
            holiday_avg = holiday_cases['expected_output'].mean()
            non_holiday_avg = non_holiday_cases['expected_output'].mean()
            difference = holiday_avg - non_holiday_avg
            holiday_effects.append((holiday, holiday_avg, non_holiday_avg, difference))
    
    if holiday_effects:
        print("\nPossible holiday effects:")
        for holiday, h_avg, nh_avg, diff in holiday_effects:
            print(f"  {holiday}: ${h_avg:.2f} vs ${nh_avg:.2f} (Diff: ${diff:.2f})")
    
    # 6. Check for specific "magic numbers" in the reimbursement amounts
    # Marcus mentioned $847 as a "lucky number"
    magic_numbers = [847]
    for number in magic_numbers:
        close_cases = df[(df['expected_output'] > number - 5) & (df['expected_output'] < number + 5)]
        if len(close_cases) > 0:
            print(f"\nCases with reimbursement close to ${number}:")
            print(f"  Found {len(close_cases)} cases")
            
            # Check if these cases have anything in common
            avg_duration = close_cases['trip_duration_days'].mean()
            avg_miles = close_cases['miles_traveled'].mean()
            avg_receipts = close_cases['total_receipts_amount'].mean()
            
            print(f"  Average duration: {avg_duration:.2f} days")
            print(f"  Average miles: {avg_miles:.2f}")
            print(f"  Average receipts: ${avg_receipts:.2f}")

def main():
    # Load data
    print("Loading data...")
    df = load_data('public_cases.json')
    print(f"Loaded {len(df)} test cases")
    
    # Check for hidden date patterns
    check_for_hidden_date_patterns(df)
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()