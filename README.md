# Beauty & Wellness Data Science: Week 1

Data Science with Python Analytics internship: orientation and problem definition in the beauty and wellness industry.

## Problem Definition

Online skincare shoppers choose from thousands of products, and many buy items that do not suit their skin type. This leads to low satisfaction, wasted spending and brand switching. Ratings and reviews help, but customers still struggle to find products that work for people like them.

This project investigates whether customer attributes (such as skin type and skin tone) and product attributes (such as category, price, brand and ingredients) can predict whether a customer will recommend a product. It also tests whether a similarity-based recommender gives more relevant suggestions than a popularity-only baseline, measured by precision@k.

**Why it matters:** Better recommendations help customers choose with confidence. They also help brands and retailers improve satisfaction, loyalty and product development.

### Research Questions
- Does average rating differ by skin type within the same product category?
- Is price linked to customer satisfaction?
- Which attributes best predict whether a product is recommended?
- Can a similarity-based recommender beat a popularity baseline?

### Planned Approach
Source a public skincare review dataset, clean and explore it, build a baseline and a similarity-based recommender, then evaluate with accuracy, F1 and precision@k.

## Files
- `eda_demo.py`: creates a demo chart from synthetic data
- `build_doc.py`: generates the Week 1 Word report
- `chart.png`: illustrative chart (synthetic data)
- `requirements.txt`: Python packages needed

## How to run
```bash
pip install -r requirements.txt
python eda_demo.py
python build_doc.py
```

## Note
The chart uses synthetic data for illustration only.


## Week 2: Data Sourcing & EDA Pipeline
- `week2_eda_pipeline.py`: Loads review data, computes descriptive statistics, conducts ANOVA & Pearson correlation testing, and builds a TF-IDF cosine similarity baseline for skincare recommendation.

