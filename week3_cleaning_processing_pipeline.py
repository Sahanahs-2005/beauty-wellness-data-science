"""
Personalised Skincare Recommendation System - Week 3 Data Cleaning & Preprocessing Pipeline
Author: SAHANA H S
Track: Data Science with Python Analytics Internship

Objective:
Develop an end-to-end data cleaning, imputation, normalization, and feature preprocessing pipeline
for beauty & wellness datasets, explicitly linking hypotheses (H1-H5) to dataset features.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SkincarePreprocessingPipeline:
    def __init__(self, raw_csv_path=None):
        self.raw_csv_path = raw_csv_path
        self.df_raw = None
        self.df_cleaned = None
        self.tfidf_matrix = None
        self.cosine_sim_matrix = None
        
    def load_data(self):
        """Step 1: Raw Data Ingestion & Synthetic Benchmark Generation"""
        print("=== Step 1: Raw Multi-Source Data Ingestion ===")
        if self.raw_csv_path and os.path.exists(self.raw_csv_path):
            self.df_raw = pd.read_csv(self.raw_csv_path)
            print(f"Loaded raw dataset from {self.raw_csv_path}: {len(self.df_raw):,} records")
        else:
            print("Generating Kaggle Sephora-aligned benchmark dataset (5,000 records)...")
            np.random.seed(42)
            n = 5000
            categories = ['Cleanser', 'Moisturiser', 'Serum', 'Sunscreen', 'Toner', 'Eye Cream', 'Face Mask', 
                          'cleansers', 'moisturizers', 'serums', 'sun care', 'toners', 'eye treatment', 'masks']
            brands = ['CeraVe', 'The Ordinary', 'La Roche-Posay', 'Paula\'s Choice', 'Drunk Elephant', 'Kiehl\'s']
            
            # Simulate raw data with typical quality flaws (missing values, typos, outliers)
            data = {
                'review_id': [f'REV_{i:06d}' for i in range(n)],
                'product_id': [f'PROD_{np.random.randint(1, 501):04d}' for _ in range(n)],
                'brand_name': np.random.choice(brands, n),
                'category_raw': np.random.choice(categories, n),
                'price_usd': np.round(np.random.exponential(scale=35.0, size=n) + 8.0, 2), # Highly right-skewed
                'ingredients_raw': np.random.choice([
                    'Water, Glycerin, Hyaluronic Acid, Niacinamide, Vitamin B3, Ceramides',
                    'Salicylic Acid, Glycolic Acid, Aqua, Zinc PCA, Aloe Vera Leaf Juice',
                    'Retinol, Squalane, Tocopherol, Vitamin E, Jojoba Oil, Rosehip Oil',
                    'Zinc Oxide, Titanium Dioxide, Avobenzone, Glycerin, Aqua'
                ], n),
                'skin_type_raw': np.random.choice(['Oily', 'Dry', 'Combination', 'Sensitive', 'oily_skin', 'dry_skin', np.nan], n, p=[0.20, 0.20, 0.25, 0.10, 0.03, 0.03, 0.19]),
                'rating': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n, p=[0.05, 0.08, 0.15, 0.35, 0.37]),
                'is_recommended': np.random.choice([0, 1], n, p=[0.19, 0.81]),
                'loves_count': np.random.randint(10, 5000, n),
                'helpfulness_votes': np.random.randint(0, 150, n)
            }
            # Inject duplicates to test deduplication
            df = pd.DataFrame(data)
            duplicates = df.iloc[:50].copy()
            self.df_raw = pd.concat([df, duplicates], ignore_index=True)
            print(f"Raw Data Loaded: {len(self.df_raw):,} records (including 50 duplicate entries)")

    def deduplicate_and_filter_spam(self):
        """Step 2: Deduplication and Bot Review Filtering"""
        print("\n=== Step 2: Deduplication & Spam Removal ===")
        initial_count = len(self.df_raw)
        
        # Deduplicate exact review matches
        self.df_cleaned = self.df_raw.drop_duplicates(subset=['review_id']).copy()
        dedup_count = initial_count - len(self.df_cleaned)
        print(f"Removed {dedup_count} duplicate review entries.")
        
        # Filter suspicious bot reviews (e.g., negative prices or invalid ratings)
        self.df_cleaned = self.df_cleaned[(self.df_cleaned['price_usd'] > 0) & 
                                         (self.df_cleaned['rating'] >= 1.0) & 
                                         (self.df_cleaned['rating'] <= 5.0)].copy()
        print(f"Cleaned dataset count after spam/invalid filter: {len(self.df_cleaned):,} records")

    def normalize_categories_and_inci(self):
        """Step 3: Categorical Standardization & INCI Chemical Synonym Mapping"""
        print("\n=== Step 3: String & INCI Ingredient Normalization ===")
        
        # Category Mapping (14 raw variations -> 7 canonical categories)
        category_map = {
            'Cleanser': 'Cleanser', 'cleansers': 'Cleanser',
            'Moisturiser': 'Moisturiser', 'moisturizers': 'Moisturiser',
            'Serum': 'Serum', 'serums': 'Serum',
            'Sunscreen': 'Sunscreen', 'sun care': 'Sunscreen',
            'Toner': 'Toner', 'toners': 'Toner',
            'Eye Cream': 'Eye Cream', 'eye treatment': 'Eye Cream',
            'Face Mask': 'Face Mask', 'masks': 'Face Mask'
        }
        self.df_cleaned['category'] = self.df_cleaned['category_raw'].map(category_map).fillna('Other')
        print(f"Standardized Categories: {self.df_cleaned['category'].unique()}")
        
        # INCI Chemical Synonym Harmonization (Addressing Task 2 Feedback for H3 & H5)
        def harmonize_inci(text):
            if pd.isna(text): return ''
            text = text.lower()
            text = re.sub(r'\b(aqua|water|purified water)\b', 'water', text)
            text = re.sub(r'\b(vitamin b3|niacinamide)\b', 'niacinamide', text)
            text = re.sub(r'\b(vitamin e|tocopherol)\b', 'tocopherol', text)
            text = re.sub(r'\b(hyaluronic acid|sodium hyaluronate)\b', 'hyaluronic_acid', text)
            text = re.sub(r'\b(salicylic acid|bha)\b', 'salicylic_acid', text)
            text = re.sub(r'\b(glycolic acid|aha)\b', 'glycolic_acid', text)
            return text
            
        self.df_cleaned['ingredients_clean'] = self.df_cleaned['ingredients_raw'].apply(harmonize_inci)
        print("INCI Chemical synonyms harmonized (e.g., 'Aqua' -> 'water', 'Vitamin B3' -> 'niacinamide').")

    def impute_missing_profiles(self):
        """Step 4: KNN Missing Value Imputation (Addressing Task 2 Feedback for H1)"""
        print("\n=== Step 4: Missing Metadata Imputation (KNN Profile Repair) ===")
        
        # Standardize raw skin_type strings
        skin_map = {
            'Oily': 'Oily', 'oily_skin': 'Oily',
            'Dry': 'Dry', 'dry_skin': 'Dry',
            'Combination': 'Combination',
            'Sensitive': 'Sensitive'
        }
        self.df_cleaned['skin_type_clean'] = self.df_cleaned['skin_type_raw'].map(skin_map)
        missing_count = self.df_cleaned['skin_type_clean'].isnull().sum()
        missing_pct = (missing_count / len(self.df_cleaned)) * 100
        print(f"Raw Missing Skin Type Metadata: {missing_count} records ({missing_pct:.2f}%)")
        
        # Perform KNN Imputation using numerical proxies
        le_cat = LabelEncoder()
        cat_encoded = le_cat.fit_transform(self.df_cleaned['category'])
        
        # Prepare numeric matrix for KNN: [price_usd, loves_count, helpfulness_votes, rating, category_encoded]
        features_for_knn = self.df_cleaned[['price_usd', 'loves_count', 'helpfulness_votes', 'rating']].copy()
        features_for_knn['cat_encoded'] = cat_encoded
        
        # Temporary numeric encoding for skin_type to run KNN
        skin_labels = ['Combination', 'Dry', 'Oily', 'Sensitive']
        skin_num_map = {name: i for i, name in enumerate(skin_labels)}
        features_for_knn['skin_num'] = self.df_cleaned['skin_type_clean'].map(skin_num_map)
        
        imputer = KNNImputer(n_neighbors=5)
        imputed_matrix = imputer.fit_transform(features_for_knn)
        
        # Map back imputed numbers to skin types
        imputed_skin_nums = np.round(imputed_matrix[:, -1]).astype(int)
        imputed_skin_nums = np.clip(imputed_skin_nums, 0, 3)
        self.df_cleaned['skin_type_imputed'] = [skin_labels[idx] for idx in imputed_skin_nums]
        
        print("Successfully repaired missing skin types using 5-Nearest Neighbors Imputation.")
        print(f"Post-Imputation Skin Type Breakdown:\n{self.df_cleaned['skin_type_imputed'].value_counts()}")

    def handle_outliers_and_scaling(self):
        """Step 5: Outlier Detection & Skewness Transformation (Addressing Task 2 Feedback for H2)"""
        print("\n=== Step 5: Outlier Handling & Feature Scaling ===")
        
        # IQR Outlier Detection for Price
        Q1 = self.df_cleaned['price_usd'].quantile(0.25)
        Q3 = self.df_cleaned['price_usd'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = (self.df_cleaned['price_usd'] > upper_bound).sum()
        print(f"IQR Price Bounds: Q1=${Q1:.2f}, Q3=${Q3:.2f}, Upper Bound=${upper_bound:.2f}")
        print(f"Identified {outliers_count} high-price outliers.")
        
        # Log Transformation for Skewed Price Distribution
        self.df_cleaned['price_log'] = np.log1p(self.df_cleaned['price_usd'])
        orig_skew = self.df_cleaned['price_usd'].skew()
        log_skew = self.df_cleaned['price_log'].skew()
        print(f"Price Skewness Reduction: Original Skew = {orig_skew:.4f} -> Log Transformed Skew = {log_skew:.4f}")
        
        # Robust Scaling for Loves Count
        scaler = RobustScaler()
        self.df_cleaned['loves_count_scaled'] = scaler.fit_transform(self.df_cleaned[['loves_count']])
        print("Loves Count scaled using RobustScaler (resistant to engagement spikes).")

    def build_recommender_baseline(self):
        """Step 6: INCI Vectorization & Baseline Similarity Matrix (Addressing Task 2 Feedback for H5)"""
        print("\n=== Step 6: TF-IDF INCI Vectorization & Baseline Model (H5) ===")
        
        prod_df = self.df_cleaned.drop_duplicates('product_id').copy()
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        self.tfidf_matrix = vectorizer.fit_transform(prod_df['ingredients_clean'])
        self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"TF-IDF Feature Space Dimensions: {self.tfidf_matrix.shape} (Products x Ingredients)")
        mean_sim = np.mean(self.cosine_sim_matrix)
        print(f"Mean Product-Product Ingredient Similarity: {mean_sim:.4f}")

    def save_outputs(self, output_csv='cleaned_skincare_dataset.csv', chart_path='eda_summary_charts.png'):
        """Step 7: Data Integrity Verification & Output Export"""
        print("\n=== Step 7: Saving Cleaned Dataset & Charts ===")
        self.df_cleaned.to_csv(output_csv, index=False)
        print(f"Cleaned dataset saved to {output_csv}")
        
        # Plot Summary Visualization
        plt.figure(figsize=(14, 5), dpi=300)
        
        plt.subplot(1, 3, 1)
        sns.histplot(self.df_raw['price_usd'], kde=True, color='red', alpha=0.5, label='Raw Price')
        sns.histplot(self.df_cleaned['price_usd'], kde=True, color='teal', label='Cleaned Price')
        plt.title('Raw vs Cleaned Price Distribution ($)')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        sns.countplot(data=self.df_cleaned, x='skin_type_imputed', palette='Set2')
        plt.title('Imputed Skin Type Distribution (KNN)')
        
        plt.subplot(1, 3, 3)
        sns.boxplot(data=self.df_cleaned, x='skin_type_imputed', y='price_log', palette='Set3')
        plt.title('Log-Price across Imputed Skin Types')
        
        plt.tight_layout()
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        print(f"Summary visualization saved to {chart_path}")

    def execute_full_pipeline(self):
        self.load_data()
        self.deduplicate_and_filter_spam()
        self.normalize_categories_and_inci()
        self.impute_missing_profiles()
        self.handle_outliers_and_scaling()
        self.build_recommender_baseline()
        self.save_outputs()


if __name__ == '__main__':
    pipeline = SkincarePreprocessingPipeline()
    pipeline.execute_full_pipeline()
