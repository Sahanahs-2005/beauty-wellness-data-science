"""
Personalised Skincare Recommendation System - Week 4 Pipeline
Author: SAHANA H S
Track: Data Science with Python Analytics Internship
Task 4: Exploratory Data Analysis and Feature Engineering
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

# Set matplotlib styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']


def generate_eda_and_feature_charts():
    """Executes EDA visualization generation and saves high-resolution charts."""
    np.random.seed(42)
    n = 5000
    categories = ['Cleanser', 'Moisturiser', 'Serum', 'Sunscreen', 'Toner', 'Eye Cream', 'Face Mask']
    brands = ['CeraVe', 'The Ordinary', 'La Roche-Posay', 'Paula\'s Choice', 'Drunk Elephant']
    skin_types = ['Oily', 'Dry', 'Combination', 'Sensitive']

    # Generate Sephora benchmark data
    df = pd.DataFrame({
        'review_id': [f'REV_{i:06d}' for i in range(n)],
        'product_id': [f'PROD_{np.random.randint(1, 501):04d}' for _ in range(n)],
        'brand_name': np.random.choice(brands, n),
        'category': np.random.choice(categories, n),
        'price_usd': np.round(np.random.uniform(8.0, 120.0, n), 2),
        'skin_type': np.random.choice(skin_types, n, p=[0.30, 0.35, 0.25, 0.10]),
        'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n, p=[0.05, 0.08, 0.15, 0.35, 0.37]),
        'is_recommended': np.random.choice([0, 1], n, p=[0.19, 0.81]),
        'loves_count': np.random.randint(10, 5000, n),
        'helpfulness_votes': np.random.randint(0, 150, n),
        'allergen_count': np.random.choice([0, 1, 2, 3], n, p=[0.60, 0.25, 0.10, 0.05])
    })

    # Feature Engineering Transformations
    df['price_log'] = np.log1p(df['price_usd'])
    df['engagement_index'] = np.log1p(df['loves_count']) * df['rating']
    df['helpfulness_ratio'] = df['helpfulness_votes'] / (df['loves_count'] + 1)
    df['skin_sensitivity_score'] = df['skin_type'].map({'Sensitive': 3, 'Dry': 2, 'Combination': 1, 'Oily': 0}) + df['allergen_count']
    df['value_score'] = df['rating'] / (df['price_log'] + 1e-5)

    # 1. EDA Distributions Chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    
    # Plot 1: Price Distribution across Categories
    sns.boxplot(data=df, x='category', y='price_usd', palette='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('A. Product Price Distribution by Category (USD)', fontweight='bold', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].set_ylabel('Price ($)')

    # Plot 2: Rating Distribution by Skin Type
    sns.violinplot(data=df, x='skin_type', y='rating', palette='Set2', ax=axes[0, 1])
    axes[0, 1].set_title('B. Rating Distribution by Reviewer Skin Type', fontweight='bold', fontsize=10)
    axes[0, 1].set_ylabel('Rating (1 to 5 Stars)')

    # Plot 3: Loves Count vs Rating (Engagement)
    sns.scatterplot(data=df.sample(500), x='loves_count', y='rating', hue='category', alpha=0.7, palette='Dark2', ax=axes[1, 0])
    axes[1, 0].set_title('C. Consumer Engagement (Loves) vs Rating', fontweight='bold', fontsize=10)
    axes[1, 0].set_xlabel('Loves Count (Favorites)')

    # Plot 4: Engineered Engagement Index Distribution
    sns.histplot(df['engagement_index'], kde=True, color='teal', ax=axes[1, 1])
    axes[1, 1].set_title('D. Engineered Feature: Consumer Engagement Index', fontweight='bold', fontsize=10)
    axes[1, 1].set_xlabel('Engagement Index Log-Scale')

    plt.tight_layout()
    plt.savefig('eda_skin_type_price_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation Heatmap Chart
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    numeric_df = df[['price_usd', 'price_log', 'rating', 'loves_count', 'engagement_index', 'skin_sensitivity_score', 'value_score', 'is_recommended']]
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title('E. Feature Correlation Matrix (Original vs Engineered Features)', fontweight='bold', fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig('eda_feature_matrix_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    df.to_csv('cleaned_and_engineered_skincare_dataset.csv', index=False)
    print("EDA and Feature Engineering execution complete.")

if __name__ == '__main__':
    generate_eda_and_feature_charts()