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

def analyze_specific_cases(df):
    """Analyze specific cases mentioned in the evaluation output."""
    # Define the specific cases from the evaluation output
    specific_cases = [
        {'id': 881, 'days': 9, 'miles': 1139, 'receipts': 1973.31, 'expected': 1759.33, 'got': 3316.75},
        {'id': 750, 'days': 9, 'miles': 1079, 'receipts': 1981.94, 'expected': 1763.16, 'got': 3136.75},
        {'id': 684, 'days': 8, 'miles': 795, 'receipts': 1645.99, 'expected': 644.69, 'got': 2010.09},
        {'id': 996, 'days': 1, 'miles': 1082, 'receipts': 1809.49, 'expected': 446.94, 'got': 1800.00},
        {'id': 144, 'days': 9, 'miles': 1096, 'receipts': 1690.22, 'expected': 1894.85, 'got': 3166.75}
    ]
    
    # Calculate percentiles for miles traveled
    miles_percentiles = np.percentile(df['miles_traveled'], [25, 50, 75, 90, 95])
    print(f"Miles Traveled Percentiles:")
    print(f"25th percentile: {miles_percentiles[0]:.2f}")
    print(f"50th percentile (median): {miles_percentiles[1]:.2f}")
    print(f"75th percentile: {miles_percentiles[2]:.2f}")
    print(f"90th percentile: {miles_percentiles[3]:.2f}")
    print(f"95th percentile: {miles_percentiles[4]:.2f}")
    
    # Check where the specific cases fall in the distribution
    print("\nSpecific Cases Analysis:")
    for case in specific_cases:
        miles = case['miles']
        percentile = np.sum(df['miles_traveled'] <= miles) / len(df) * 100
        
        print(f"Case {case['id']}: {case['days']} days, {miles} miles, ${case['receipts']:.2f} receipts")
        print(f"  Expected: ${case['expected']:.2f}, Got: ${case['got']:.2f}, Error: ${abs(case['expected'] - case['got']):.2f}")
        print(f"  Miles traveled is in the {percentile:.1f}th percentile")
        
        if miles > miles_percentiles[2]:
            print(f"  This case is in the TOP 25% of miles traveled")
        elif miles > miles_percentiles[1]:
            print(f"  This case is in the TOP 50% of miles traveled")
        else:
            print(f"  This case is in the BOTTOM 50% of miles traveled")
    
    # Create a histogram of miles traveled with specific cases marked
    plt.figure(figsize=(12, 6))
    sns.histplot(df['miles_traveled'], bins=30, kde=True)
    
    # Add vertical lines for the specific cases
    for case in specific_cases:
        miles = case['miles']
        plt.axvline(x=miles, color='r', linestyle='--', alpha=0.7)
        plt.text(miles, 0, f"Case {case['id']}", rotation=90, verticalalignment='bottom')
    
    # Add vertical lines for percentiles
    plt.axvline(x=miles_percentiles[2], color='g', linestyle='-', label='75th percentile')
    plt.axvline(x=miles_percentiles[3], color='b', linestyle='-', label='90th percentile')
    
    plt.title('Distribution of Miles Traveled with Specific Cases Marked')
    plt.xlabel('Miles Traveled')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('miles_traveled_histogram_specific.png')
    plt.close()
    
    # Create a scatter plot of miles traveled vs. trip duration with specific cases highlighted
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    plt.scatter(df['trip_duration_days'], df['miles_traveled'], alpha=0.3, label='All cases')
    
    # Highlight specific cases
    for i, case in enumerate(specific_cases):
        plt.scatter(case['days'], case['miles'], color='r', s=100, 
                   label='Specific cases' if i == 0 else "")
        plt.text(case['days'], case['miles'], f"{case['id']}", fontsize=9)
    
    plt.title('Miles Traveled vs. Trip Duration with Specific Cases Highlighted')
    plt.xlabel('Trip Duration (days)')
    plt.ylabel('Miles Traveled')
    plt.axhline(y=miles_percentiles[2], color='g', linestyle='--', label='75th percentile')
    plt.legend()
    plt.savefig('miles_vs_duration_specific.png')
    plt.close()
    
    # Create a 3D scatter plot of miles traveled, trip duration, and receipt amount
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points
    ax.scatter(df['trip_duration_days'], df['miles_traveled'], df['total_receipts_amount'], 
              alpha=0.3, label='All cases')
    
    # Highlight specific cases
    for i, case in enumerate(specific_cases):
        ax.scatter(case['days'], case['miles'], case['receipts'], 
                  color='r', s=100, label='Specific cases' if i == 0 else "")
        ax.text(case['days'], case['miles'], case['receipts'], 
               f"{case['id']}", fontsize=9)
    
    ax.set_title('3D Plot of Trip Duration, Miles Traveled, and Receipt Amount')
    ax.set_xlabel('Trip Duration (days)')
    ax.set_ylabel('Miles Traveled')
    ax.set_zlabel('Receipt Amount ($)')
    plt.legend()
    plt.savefig('3d_plot_specific.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data('public_cases.json')
    print(f"Loaded {len(df)} test cases")
    
    # Analyze specific cases
    print("\nAnalyzing specific cases...")
    analyze_specific_cases(df)
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()