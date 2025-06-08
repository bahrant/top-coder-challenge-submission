#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

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

def engineer_features(df):
    """Create additional features that might help explain the patterns."""
    # Basic features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days']
    
    # Categorical features based on interview insights
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)
    df['is_4_to_6_day_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    df['is_long_trip'] = (df['trip_duration_days'] >= 8).astype(int)
    
    # Efficiency categories
    df['is_efficient'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    df['is_very_efficient'] = (df['miles_per_day'] > 220).astype(int)
    df['is_inefficient'] = (df['miles_per_day'] < 100).astype(int)
    
    # Receipt categories
    df['has_small_receipts'] = (df['total_receipts_amount'] < 50).astype(int)
    df['has_medium_receipts'] = ((df['total_receipts_amount'] >= 50) & (df['total_receipts_amount'] < 600)).astype(int)
    df['has_sweet_spot_receipts'] = ((df['total_receipts_amount'] >= 600) & (df['total_receipts_amount'] < 800)).astype(int)
    df['has_large_receipts'] = (df['total_receipts_amount'] >= 800).astype(int)
    
    # Special combinations
    df['is_sweet_spot_combo'] = ((df['trip_duration_days'] == 5) & 
                                (df['miles_per_day'] >= 180) & 
                                (df['miles_per_day'] <= 220) & 
                                (df['total_receipts_amount'] < 100)).astype(int)
    
    df['is_vacation_penalty'] = ((df['trip_duration_days'] >= 8) & 
                                (df['total_receipts_amount'] > 1000)).astype(int)
    
    # Rounding quirk
    df['has_rounding_quirk'] = (df['total_receipts_amount'].apply(
        lambda x: int(round((x % 1) * 100)) in [49, 99])).astype(int)
    
    # Interaction terms
    df['miles_x_days'] = df['miles_traveled'] * df['trip_duration_days']
    df['receipts_x_days'] = df['total_receipts_amount'] * df['trip_duration_days']
    df['miles_x_receipts'] = df['miles_traveled'] * df['total_receipts_amount']
    
    # Nonlinear transformations
    df['log_miles'] = np.log1p(df['miles_traveled'])
    df['log_receipts'] = np.log1p(df['total_receipts_amount'])
    df['sqrt_miles'] = np.sqrt(df['miles_traveled'])
    df['sqrt_receipts'] = np.sqrt(df['total_receipts_amount'])
    
    # Tiered mileage features
    df['miles_tier1'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_tier2'] = df['miles_traveled'].apply(lambda x: max(0, min(x - 100, 200)))
    df['miles_tier3'] = df['miles_traveled'].apply(lambda x: max(0, min(x - 300, 300)))
    df['miles_tier4'] = df['miles_traveled'].apply(lambda x: max(0, x - 600))
    
    # Receipt cents features (to capture rounding quirks)
    df['receipt_cents'] = df['total_receipts_amount'].apply(lambda x: int(round((x % 1) * 100)))
    
    return df

def train_models(df):
    """Train various regression models and evaluate their performance."""
    # Prepare features and target
    feature_cols = [col for col in df.columns if col != 'expected_output']
    X = df[feature_cols]
    y = df['expected_output']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'model': model, 'mae': mae, 'r2': r2}
        print(f"{name}: MAE = ${mae:.2f}, R² = {r2:.4f}")
    
    # Return the best model
    best_model_name = min(results, key=lambda k: results[k]['mae'])
    return results[best_model_name]['model'], feature_cols

def analyze_feature_importance(model, feature_names):
    """Analyze feature importance from the best model."""
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Importance:")
        for i in range(min(20, len(feature_names))):
            idx = indices[i]
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(min(20, len(feature_names))), 
                [importances[i] for i in indices[:20]], 
                align='center')
        plt.xticks(range(min(20, len(feature_names))), 
                  [feature_names[i] for i in indices[:20]], 
                  rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_
        indices = np.argsort(np.abs(coefficients))[::-1]
        
        print("\nFeature Coefficients:")
        for i in range(min(20, len(feature_names))):
            idx = indices[i]
            print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")
        
        # Plot coefficients
        plt.figure(figsize=(12, 8))
        plt.title('Feature Coefficients')
        plt.bar(range(min(20, len(feature_names))), 
                [coefficients[i] for i in indices[:20]], 
                align='center')
        plt.xticks(range(min(20, len(feature_names))), 
                  [feature_names[i] for i in indices[:20]], 
                  rotation=90)
        plt.tight_layout()
        plt.savefig('feature_coefficients.png')
        plt.close()

def analyze_patterns(df):
    """Analyze specific patterns mentioned in the interviews."""
    # 1. Analyze 5-day trip bonus
    five_day_trips = df[df['trip_duration_days'] == 5]
    other_trips = df[(df['trip_duration_days'] == 4) | (df['trip_duration_days'] == 6)]
    
    per_day_5 = five_day_trips['expected_output'] / five_day_trips['trip_duration_days']
    per_day_other = other_trips['expected_output'] / other_trips['trip_duration_days']
    
    print(f"\n5-day trips average per day: ${per_day_5.mean():.2f}")
    print(f"4/6-day trips average per day: ${per_day_other.mean():.2f}")
    print(f"Difference: ${per_day_5.mean() - per_day_other.mean():.2f} per day")
    
    # 2. Analyze mileage tiers
    mileage_groups = [
        (0, 100),
        (100, 300),
        (300, 600),
        (600, float('inf'))
    ]
    
    print("\nMileage Tier Analysis:")
    for low, high in mileage_groups:
        group = df[(df['miles_traveled'] >= low) & (df['miles_traveled'] < high)]
        if len(group) > 0:
            avg_per_mile = (group['expected_output'] - group['trip_duration_days'] * 100) / group['miles_traveled']
            print(f"Miles {low}-{high}: ${avg_per_mile.mean():.4f} per mile (n={len(group)})")
    
    # 3. Analyze receipt processing
    receipt_groups = [
        (0, 50),
        (50, 100),
        (100, 600),
        (600, 800),
        (800, float('inf'))
    ]
    
    print("\nReceipt Amount Analysis:")
    for low, high in receipt_groups:
        group = df[(df['total_receipts_amount'] >= low) & (df['total_receipts_amount'] < high)]
        if len(group) > 0:
            # Estimate receipt contribution by subtracting estimated per diem and mileage
            est_per_diem = group['trip_duration_days'] * 100
            est_mileage = group['miles_tier1'] * 0.58 + group['miles_tier2'] * 0.52 + \
                         group['miles_tier3'] * 0.47 + group['miles_tier4'] * 0.40
            est_receipt_contrib = group['expected_output'] - est_per_diem - est_mileage
            avg_receipt_factor = est_receipt_contrib / group['total_receipts_amount']
            print(f"Receipts ${low}-${high}: Factor ~{avg_receipt_factor.mean():.2f} (n={len(group)})")
    
    # 4. Analyze rounding quirk
    quirk_cases = df[df['has_rounding_quirk'] == 1]
    non_quirk_cases = df[df['has_rounding_quirk'] == 0]
    
    # Normalize by trip duration to compare fairly
    quirk_per_day = quirk_cases['expected_output'] / quirk_cases['trip_duration_days']
    non_quirk_per_day = non_quirk_cases['expected_output'] / non_quirk_cases['trip_duration_days']
    
    print(f"\nRounding Quirk Analysis:")
    print(f"Cases with .49/.99 cents: ${quirk_per_day.mean():.2f} per day (n={len(quirk_cases)})")
    print(f"Cases without .49/.99 cents: ${non_quirk_per_day.mean():.2f} per day (n={len(non_quirk_cases)})")
    print(f"Difference: ${quirk_per_day.mean() - non_quirk_per_day.mean():.2f} per day")
    
    # 5. Analyze efficiency bonus
    efficient_cases = df[df['is_efficient'] == 1]
    inefficient_cases = df[df['is_inefficient'] == 1]
    
    # Normalize by trip duration to compare fairly
    efficient_per_day = efficient_cases['expected_output'] / efficient_cases['trip_duration_days']
    inefficient_per_day = inefficient_cases['expected_output'] / inefficient_cases['trip_duration_days']
    
    print(f"\nEfficiency Analysis:")
    print(f"Efficient trips (180-220 miles/day): ${efficient_per_day.mean():.2f} per day (n={len(efficient_cases)})")
    print(f"Inefficient trips (<100 miles/day): ${inefficient_per_day.mean():.2f} per day (n={len(inefficient_cases)})")
    print(f"Difference: ${efficient_per_day.mean() - inefficient_per_day.mean():.2f} per day")

def visualize_patterns(df):
    """Create visualizations to help identify patterns."""
    # Set up the plotting style
    sns.set(style="whitegrid")
    
    # 1. Reimbursement by trip duration
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='trip_duration_days', y='expected_output', data=df)
    plt.title('Reimbursement by Trip Duration')
    plt.xlabel('Trip Duration (days)')
    plt.ylabel('Reimbursement Amount ($)')
    plt.savefig('reimbursement_by_duration.png')
    plt.close()
    
    # 2. Reimbursement per day by trip duration
    df['reimbursement_per_day'] = df['expected_output'] / df['trip_duration_days']
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='trip_duration_days', y='reimbursement_per_day', data=df)
    plt.title('Reimbursement per Day by Trip Duration')
    plt.xlabel('Trip Duration (days)')
    plt.ylabel('Reimbursement per Day ($)')
    plt.savefig('reimbursement_per_day.png')
    plt.close()
    
    # 3. Reimbursement vs Miles Traveled
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='miles_traveled', y='expected_output', hue='trip_duration_days', data=df)
    plt.title('Reimbursement vs Miles Traveled')
    plt.xlabel('Miles Traveled')
    plt.ylabel('Reimbursement Amount ($)')
    plt.savefig('reimbursement_vs_miles.png')
    plt.close()
    
    # 4. Reimbursement vs Receipt Amount
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_receipts_amount', y='expected_output', hue='trip_duration_days', data=df)
    plt.title('Reimbursement vs Receipt Amount')
    plt.xlabel('Receipt Amount ($)')
    plt.ylabel('Reimbursement Amount ($)')
    plt.savefig('reimbursement_vs_receipts.png')
    plt.close()
    
    # 5. Reimbursement vs Efficiency (miles per day)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='miles_per_day', y='reimbursement_per_day', hue='trip_duration_days', data=df)
    plt.title('Reimbursement per Day vs Efficiency')
    plt.xlabel('Miles per Day')
    plt.ylabel('Reimbursement per Day ($)')
    plt.savefig('reimbursement_vs_efficiency.png')
    plt.close()
    
    # 6. Correlation heatmap
    plt.figure(figsize=(14, 12))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    df = load_data('public_cases.json')
    print(f"Loaded {len(df)} test cases")
    
    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(df)
    
    # Analyze patterns
    print("\nAnalyzing patterns...")
    analyze_patterns(df)
    
    # Visualize patterns
    print("\nCreating visualizations...")
    visualize_patterns(df)
    
    # Train models
    print("\nTraining models...")
    best_model, feature_names = train_models(df)
    
    # Analyze feature importance
    analyze_feature_importance(best_model, feature_names)
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()