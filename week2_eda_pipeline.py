"""
Personalised Skincare Recommendation System - Week 2 Pipeline
Author: SAHANA H S
Track: Data Science with Python Analytics Internship
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def run_eda_pipeline(csv_file_path=None, chart_path='eda_summary_charts.png'):
    print("=== Step 1: Loading Dataset & Generating Statistical Summary ===")
    
    # Load dataset if available, otherwise generate Sephora-aligned synthetic benchmark
    if csv_file_path and os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        np.random.seed(42)
        n = 5000
        categories = ['Cleanser', 'Moisturiser', 'Serum', 'Sunscreen', 'Toner', 'Eye Cream', 'Face Mask']
        brands = ['CeraVe', 'The Ordinary', 'La Roche-Posay', 'Paula\'s Choice', 'Drunk Elephant']
        
        data = {
            'review_id': [f'REV_{i:06d}' for i in range(n)],
            'product_id': [f'PROD_{np.random.randint(1, 501):04d}' for _ in range(n)],
            'brand_name': np.random.choice(brands, n),
            'category': np.random.choice(categories, n),
            'price_usd': np.round(np.random.uniform(8.0, 120.0, n), 2),
            'ingredients': np.random.choice([
                'water, glycerin, hyaluronic acid, niacinamide, ceramides',
                'salicylic acid, glycolic acid, aqua, zinc pca, aloe vera',
                'retinol, squalane, tocopherol, jojoba oil, rosehip oil',
                'zinc oxide, titanium dioxide, avobenzone, glycerin'
            ], n),
            'skin_type': np.random.choice(['Oily', 'Dry', 'Combination', 'Sensitive', np.nan], n, p=[0.25, 0.25, 0.30, 0.10, 0.10]),
            'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n, p=[0.05, 0.08, 0.15, 0.35, 0.37]),
            'is_recommended': np.random.choice([0, 1], n, p=[0.2, 0.8]),
            'loves_count': np.random.randint(10, 5000, n),
            'helpfulness_votes': np.random.randint(0, 150, n)
        }
        df = pd.DataFrame(data)

    print(f"Total Dataset Records: {len(df):,}")
    print(f"Unique Products: {df['product_id'].nunique():,}\n")

    # 1. Data Quality Profile (Missingness)
    missing_summary = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Pct': (df.isnull().sum() / len(df)) * 100
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    print("Data Quality Profile (Missingness):")
    print(missing_summary)
    
    # 2. Descriptive Statistics Summary
    print("\nDescriptive Statistics Summary:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    desc_stats = df[numeric_cols].describe().T
    desc_stats['median'] = df[numeric_cols].median()
    desc_stats['IQR'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    print(desc_stats[['mean', 'std', 'median', 'IQR', 'min', 'max']])

    # 3. Hypothesis Tests
    pearson_r, p_val_r = stats.pearsonr(df['price_usd'], df['rating'])
    print(f"\nH2 Test (Price vs Rating Correlation): Pearson r = {pearson_r:.4f}, p-value = {p_val_r:.4f}")

    df_clean_st = df.dropna(subset=['skin_type'])
    st_groups = [group['rating'].values for _, group in df_clean_st.groupby('skin_type')]
    f_stat, p_val_anova = stats.f_oneway(*st_groups)
    print(f"H1 Test (Rating variance across Skin Types): ANOVA F = {f_stat:.4f}, p-value = {p_val_anova:.4f}\n")

    # 4. Recommender Baseline (H5)
    print("=== Step 2: Building TF-IDF Similarity Recommender Baseline (H5) ===")
    prod_df = df.drop_duplicates('product_id').copy()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(prod_df['ingredients'].fillna(''))
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print(f"Ingredient Vector Space Shape: {tfidf_matrix.shape}")
    mean_sim = np.mean(cosine_sim_matrix)
    print(f"Mean Product-Product Ingredient Similarity: {mean_sim:.4f}")

    # 5. Save Visualization Chart Locally
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['price_usd'], kde=True, color='teal')
    plt.title('Price Distribution (USD)')
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df_clean_st, x='skin_type', palette='Set2')
    plt.title('Review Count by Skin Type')
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved EDA chart summary to {chart_path}")


if __name__ == '__main__':
    run_eda_pipeline()