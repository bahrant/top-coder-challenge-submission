#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

def engineer_features(df):
    """Create additional features that might help explain the patterns."""
    # Basic features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    df['reimbursement_per_day'] = df['expected_output'] / df['trip_duration_days']
    
    # Interaction features (Kevin emphasized these)
    df['miles_x_days'] = df['miles_traveled'] * df['trip_duration_days']
    df['receipts_x_days'] = df['total_receipts_amount'] * df['trip_duration_days']
    df['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    
    # Categorical features
    df['trip_category'] = pd.cut(
        df['trip_duration_days'], 
        bins=[0, 3, 7, float('inf')], 
        labels=['short', 'medium', 'long']
    )
    
    # Simulate user history/profile effects
    df['user_id'] = df['case_id'] % 50  # Simulate 50 different users
    
    # Simulate department effects
    departments = ['sales', 'marketing', 'operations', 'finance', 'hr']
    df['department'] = [departments[i % len(departments)] for i in df['case_id']]
    
    # Simulate location/city effects
    cities = ['new_york', 'chicago', 'los_angeles', 'dallas', 'miami', 
              'seattle', 'denver', 'boston', 'atlanta', 'phoenix']
    df['city'] = [cities[i % len(cities)] for i in df['case_id']]
    
    # Simulate submission timing effects
    days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    df['submission_day'] = [days_of_week[i % len(days_of_week)] for i in df['case_id']]
    
    # Simulate lunar cycle effects
    df['moon_phase'] = df['case_id'] % 30
    df['is_new_moon'] = (df['moon_phase'] < 3).astype(int)
    df['is_full_moon'] = ((df['moon_phase'] > 13) & (df['moon_phase'] < 17)).astype(int)
    
    # Simulate quarter effects
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    df['quarter'] = [quarters[i % len(quarters)] for i in df['case_id']]
    df['is_end_of_quarter'] = ((df['case_id'] % 90) >= 60).astype(int)
    
    return df

def analyze_daily_spending_effects(df):
    """Analyze if daily spending rates affect reimbursement differently than total receipts."""
    print("\n=== Daily Spending vs. Total Receipts Effects ===")
    
    # Create bins for daily spending
    df['daily_spending_bin'] = pd.cut(
        df['receipts_per_day'],
        bins=[0, 50, 75, 100, 150, float('inf')],
        labels=['0-50', '50-75', '75-100', '100-150', '150+']
    )
    
    # Analyze by trip duration category
    for trip_cat in ['short', 'medium', 'long']:
        subset = df[df['trip_category'] == trip_cat]
        
        print(f"\nFor {trip_cat} trips (duration: {subset['trip_duration_days'].min()}-{subset['trip_duration_days'].max()} days):")
        
        # Daily spending analysis
        daily_avg = subset.groupby('daily_spending_bin')['reimbursement_per_day'].mean()
        
        print("  Reimbursement by daily spending:")
        for bin_name, avg in daily_avg.items():
            print(f"    ${bin_name}/day: ${avg:.2f} per day")
        
        # Check if Kevin's optimal spending ranges are reflected in the data
        if trip_cat == 'short':
            optimal = '0-50, 50-75'
        elif trip_cat == 'medium':
            optimal = '75-100, 100-150'
        else:
            optimal = '0-50, 50-75, 75-100'
        
        print(f"  Kevin's optimal spending range for {trip_cat} trips: ${optimal}/day")

def analyze_submission_timing_effects(df):
    """Analyze if submission timing affects reimbursement."""
    print("\n=== Submission Timing Effects ===")
    
    # Day of week effects
    day_avg = df.groupby('submission_day')['reimbursement_per_day'].mean()
    
    print("Average reimbursement by submission day:")
    for day, avg in day_avg.items():
        print(f"  {day.capitalize()}: ${avg:.2f} per day")
    
    # Check if Tuesday is better than Monday (as Kevin claimed)
    monday_avg = day_avg['monday']
    tuesday_avg = day_avg['tuesday']
    friday_avg = day_avg['friday']
    
    print(f"\nTuesday vs. Monday: ${tuesday_avg - monday_avg:.2f} per day ({(tuesday_avg - monday_avg) / monday_avg * 100:.2f}%)")
    print(f"Tuesday vs. Friday: ${tuesday_avg - friday_avg:.2f} per day ({(tuesday_avg - friday_avg) / friday_avg * 100:.2f}%)")
    
    # Lunar cycle effects
    new_moon_avg = df[df['is_new_moon'] == 1]['reimbursement_per_day'].mean()
    full_moon_avg = df[df['is_full_moon'] == 1]['reimbursement_per_day'].mean()
    
    print("\nLunar cycle effects:")
    print(f"  New moon: ${new_moon_avg:.2f} per day")
    print(f"  Full moon: ${full_moon_avg:.2f} per day")
    print(f"  New moon vs. Full moon: ${new_moon_avg - full_moon_avg:.2f} per day ({(new_moon_avg - full_moon_avg) / full_moon_avg * 100:.2f}%)")

def identify_calculation_paths(df):
    """Try to identify different calculation paths using clustering."""
    print("\n=== Multiple Calculation Paths Analysis ===")
    
    # Select features for clustering
    features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 
                'miles_per_day', 'receipts_per_day']
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Perform k-means clustering with 6 clusters (as Kevin suggested)
    kmeans = KMeans(n_clusters=6, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Analyze clusters
    cluster_stats = df.groupby('cluster').agg({
        'trip_duration_days': 'mean',
        'miles_traveled': 'mean',
        'total_receipts_amount': 'mean',
        'miles_per_day': 'mean',
        'receipts_per_day': 'mean',
        'expected_output': 'mean',
        'reimbursement_per_day': 'mean',
        'case_id': 'count'
    }).rename(columns={'case_id': 'count'})
    
    print(f"\nK-means clustering with 6 clusters (as Kevin suggested):")
    for cluster, stats in cluster_stats.iterrows():
        print(f"  Cluster {cluster} ({stats['count']} cases):")
        print(f"    Avg duration: {stats['trip_duration_days']:.1f} days")
        print(f"    Avg miles: {stats['miles_traveled']:.1f}")
        print(f"    Avg receipts: ${stats['total_receipts_amount']:.2f}")
        print(f"    Avg miles/day: {stats['miles_per_day']:.1f}")
        print(f"    Avg spending/day: ${stats['receipts_per_day']:.2f}")
        print(f"    Avg reimbursement: ${stats['expected_output']:.2f}")
        print(f"    Avg reimbursement/day: ${stats['reimbursement_per_day']:.2f}")

def main():
    # Load data
    print("Loading data...")
    df = load_data('public_cases.json')
    print(f"Loaded {len(df)} test cases")
    
    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(df)
    
    # Analyze patterns we might have missed
    analyze_daily_spending_effects(df)
    analyze_submission_timing_effects(df)
    identify_calculation_paths(df)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()