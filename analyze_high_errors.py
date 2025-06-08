#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load test cases from JSON file and convert to DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract input parameters and expected output
    rows = []
    for i, case in enumerate(data):
        row = {
            'case_id': i,
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'expected_output': case['expected_output']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def analyze_high_error_cases(df):
    """Analyze the high-error cases in relation to miles traveled distribution."""
    # High-error case IDs
    high_error_cases = [881, 750, 684, 996, 144]
    
    # Create a boxplot of miles traveled
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['miles_traveled'])
    plt.title('Distribution of Miles Traveled')
    plt.xlabel('Miles Traveled')
    plt.savefig('miles_traveled_boxplot.png')
    plt.close()
    
    # Calculate percentiles for miles traveled
    miles_percentiles = np.percentile(df['miles_traveled'], [25, 50, 75, 90, 95])
    print(f"Miles Traveled Percentiles:")
    print(f"25th percentile: {miles_percentiles[0]:.2f}")
    print(f"50th percentile (median): {miles_percentiles[1]:.2f}")
    print(f"75th percentile: {miles_percentiles[2]:.2f}")
    print(f"90th percentile: {miles_percentiles[3]:.2f}")
    print(f"95th percentile: {miles_percentiles[4]:.2f}")
    
    # Check where the high-error cases fall in the distribution
    print("\nHigh-Error Cases Analysis:")
    for case_id in high_error_cases:
        case = df.loc[case_id]
        miles = case['miles_traveled']
        percentile = np.sum(df['miles_traveled'] <= miles) / len(df) * 100
        
        print(f"Case {case_id}: {case['trip_duration_days']} days, {miles:.0f} miles, ${case['total_receipts_amount']:.2f} receipts")
        print(f"  Miles traveled is in the {percentile:.1f}th percentile")
        
        if miles > miles_percentiles[2]:
            print(f"  This case is in the TOP 25% of miles traveled")
        elif miles > miles_percentiles[1]:
            print(f"  This case is in the TOP 50% of miles traveled")
        else:
            print(f"  This case is in the BOTTOM 50% of miles traveled")
    
    # Create a histogram of miles traveled with high-error cases marked
    plt.figure(figsize=(12, 6))
    sns.histplot(df['miles_traveled'], bins=30, kde=True)
    
    # Add vertical lines for the high-error cases
    for case_id in high_error_cases:
        miles = df.loc[case_id, 'miles_traveled']
        plt.axvline(x=miles, color='r', linestyle='--', alpha=0.7)
        plt.text(miles, 0, f"Case {case_id}", rotation=90, verticalalignment='bottom')
    
    # Add vertical lines for percentiles
    plt.axvline(x=miles_percentiles[2], color='g', linestyle='-', label='75th percentile')
    plt.axvline(x=miles_percentiles[3], color='b', linestyle='-', label='90th percentile')
    
    plt.title('Distribution of Miles Traveled with High-Error Cases Marked')
    plt.xlabel('Miles Traveled')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('miles_traveled_histogram.png')
    plt.close()
    
    # Create a scatter plot of miles traveled vs. trip duration with high-error cases highlighted
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    plt.scatter(df['trip_duration_days'], df['miles_traveled'], alpha=0.3, label='All cases')
    
    # Highlight high-error cases
    for case_id in high_error_cases:
        case = df.loc[case_id]
        plt.scatter(case['trip_duration_days'], case['miles_traveled'], color='r', s=100, 
                   label=f'Case {case_id}' if case_id == high_error_cases[0] else "")
        plt.text(case['trip_duration_days'], case['miles_traveled'], f"{case_id}", fontsize=9)
    
    plt.title('Miles Traveled vs. Trip Duration with High-Error Cases Highlighted')
    plt.xlabel('Trip Duration (days)')
    plt.ylabel('Miles Traveled')
    plt.axhline(y=miles_percentiles[2], color='g', linestyle='--', label='75th percentile')
    plt.legend()
    plt.savefig('miles_vs_duration.png')
    plt.close()
    
    # Create a 3D scatter plot of miles traveled, trip duration, and receipt amount
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    ax.scatter(df['trip_duration_days'], df['miles_traveled'], df['total_receipts_amount'], 
              alpha=0.3, label='All cases')
    
    # Highlight high-error cases
    for case_id in high_error_cases:
        case = df.loc[case_id]
        ax.scatter(case['trip_duration_days'], case['miles_traveled'], case['total_receipts_amount'], 
                  color='r', s=100, label=f'Case {case_id}' if case_id == high_error_cases[0] else "")
        ax.text(case['trip_duration_days'], case['miles_traveled'], case['total_receipts_amount'], 
               f"{case_id}", fontsize=9)
    
    ax.set_title('3D Plot of Trip Duration, Miles Traveled, and Receipt Amount')
    ax.set_xlabel('Trip Duration (days)')
    ax.set_ylabel('Miles Traveled')
    ax.set_zlabel('Receipt Amount ($)')
    plt.legend()
    plt.savefig('3d_plot.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data('public_cases.json')
    print(f"Loaded {len(df)} test cases")
    
    # Analyze high-error cases
    print("\nAnalyzing high-error cases...")
    analyze_high_error_cases(df)
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()