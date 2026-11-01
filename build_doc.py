import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

NAME = "Sahana"  # change if needed
doc = Document()

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)


def heading(text, level=1):
    doc.add_heading(text, level=level)


def para(text):
    doc.add_paragraph(text)


def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def numbered(items):
    for it in items:
        doc.add_paragraph(it, style="List Number")


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        cell.paragraphs[0].add_run(h).bold = True
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = v
    doc.add_paragraph()


# ---------- Title ----------
title = doc.add_heading("Personalised Skincare Recommendation in Beauty & Wellness", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph("Week 1: Data Science Orientation & Problem Definition")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
by = doc.add_paragraph(f"Prepared by: {NAME}")
by.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ---------- 1 ----------
heading("1. Introduction and Business Context")
para("The beauty and wellness industry spans skincare, haircare, cosmetics, fragrance, "
     "personal care and wellness products and services. Shopping has moved heavily online, "
     "where customers choose from thousands of products and rely on ratings, reviews, "
     "influencers and social media. This abundance of choice makes it hard for customers to "
     "find products that suit their skin and goals, and hard for brands to understand what "
     "drives satisfaction and loyalty. Data science can turn customer reviews, purchase "
     "behaviour and product attributes into insight for better decisions.")

# ---------- 2 ----------
heading("2. Industry Trends (Publicly Available Research)")
para("The following trends are widely discussed in public industry reports and articles. "
     "Each should be supported with a cited figure or statement from the sources in Section 12.")
bullets([
    "Personalisation: growing demand for products tailored to skin type, concerns and goals, "
    "including AI-based skin analysis tools.",
    "Social media influence: platforms such as Instagram and YouTube shape product discovery, "
    "and viral trends can shift demand quickly.",
    "Ingredient transparency and 'clean beauty': customers increasingly check ingredient lists "
    "and avoid ingredients they consider harmful.",
    "Wellness convergence: skincare is increasingly linked with sleep, nutrition, stress "
    "management and overall health.",
    "Direct-to-consumer and online retail growth: brands collect first-party customer data and "
    "compete on experience.",
    "User-generated content: ratings and reviews strongly influence purchase decisions.",
    "Sustainability: eco-friendly packaging and ethical sourcing affect brand perception.",
])
para("[Add 2 to 3 verified statistics here, each with its source, e.g. market size or growth "
     "figures from a public industry report.]")

# ---------- 3 ----------
heading("3. Problem Statement")
para("Customers in the beauty and wellness market often purchase products that do not suit "
     "their skin type, concerns or budget, which results in low ratings, returns and brand "
     "switching. This project investigates whether customer attributes (such as skin type and "
     "concerns) and product attributes (such as category, price, brand and ingredients) can "
     "predict which products a customer will rate highly and recommend. The goal is to support "
     "more relevant, personalised recommendations for brands and retailers.")

# ---------- 4 ----------
heading("4. Relevance to the Beauty & Wellness Sector")
bullets([
    "For customers: less trial and error, lower spending on unsuitable products, better "
    "outcomes and safety.",
    "For brands: higher satisfaction, repeat purchase and loyalty; fewer returns; insight into "
    "which product features matter to which segments.",
    "For retailers and marketers: better targeting, cross-selling and inventory planning.",
    "For the industry: a data-driven way to combine marketing insight with analytic rigour.",
])

# ---------- 5 ----------
heading("5. Objectives")
numbered([
    "Define a clear, measurable business problem in the beauty and wellness domain.",
    "Identify the customer, product and review variables that may influence satisfaction.",
    "Explore patterns in ratings and recommendations across skin types, categories and price.",
    "Design a recommendation and prediction approach that can be tested in later weeks.",
    "Document the rationale for every methodological choice.",
])

# ---------- 6 ----------
heading("6. Research Questions and Hypotheses")
table(
    ["#", "Research Question", "Hypothesis"],
    [
        ["RQ1", "Does average rating differ by skin type within the same product category?",
         "H1: Ratings differ significantly across skin types for the same category."],
        ["RQ2", "Is price associated with customer satisfaction?",
         "H2: Higher price is not by itself associated with higher ratings."],
        ["RQ3", "Which product and customer attributes best predict whether a product is recommended?",
         "H3: Skin-type match and product category are stronger predictors than price."],
        ["RQ4", "Do review texts reveal reasons for dissatisfaction that ratings alone miss?",
         "H4: Negative reviews cluster around themes such as irritation, texture and "
         "ineffectiveness."],
        ["RQ5", "Can a recommender using similar customers outperform a popularity baseline?",
         "H5: A similarity-based recommender gives more relevant recommendations than "
         "recommending only the most popular products."],
    ],
)

# ---------- 7 ----------
heading("7. Variables and Possible Data")
table(
    ["Variable", "Type", "Role", "Possible Source"],
    [
        ["rating (1 to 5)", "Numeric", "Outcome", "Public review dataset"],
        ["is_recommended", "Binary", "Target (classification)", "Public review dataset"],
        ["skin_type", "Categorical", "Customer feature", "Reviewer profile"],
        ["skin_tone / hair colour / eye colour", "Categorical", "Customer features", "Reviewer profile"],
        ["product category", "Categorical", "Product feature", "Product catalogue"],
        ["brand", "Categorical", "Product feature", "Product catalogue"],
        ["price", "Numeric", "Product feature", "Product catalogue"],
        ["ingredients", "Text", "Product feature", "Product catalogue"],
        ["review text", "Text", "Sentiment and themes", "Public review dataset"],
        ["helpfulness votes / loves count", "Numeric", "Engagement signal", "Public review dataset"],
    ],
)
para("Candidate public data: skincare product and review datasets on Kaggle (for example, "
     "search for 'Sephora Products and Skincare Reviews'), open product databases such as "
     "Open Beauty Facts, and public social-media trend data. Exact columns should be verified "
     "once the dataset is downloaded.")

# ---------- 8 ----------
heading("8. Step-by-Step Approach")
numbered([
    "Business understanding: confirm the problem, stakeholders and success measures.",
    "Data collection: download the public review and product datasets.",
    "Data cleaning: handle missing values, duplicates, inconsistent categories and outliers.",
    "Exploratory data analysis: distributions, correlations and group comparisons.",
    "Feature engineering: encode categories, scale numeric values and extract text features.",
    "Modelling: baseline recommender (popularity), similarity-based recommender, and a "
    "classifier to predict recommendation.",
    "Evaluation: accuracy, precision, recall, F1 and ROC-AUC for classification; "
    "precision@k for recommendations.",
    "Interpretation: identify drivers of satisfaction and translate them into business actions.",
    "Communication: visual dashboard and report for non-technical stakeholders.",
])

# ---------- 9 ----------
heading("9. Analysis Plan and Tools")
heading("9.1 Initial exploratory ideas", 2)
bullets([
    "Average rating by skin type and category (see illustrative chart below).",
    "Rating distribution and share of recommended products.",
    "Price versus rating scatter plot and correlation.",
    "Most common words in positive versus negative reviews.",
    "Top brands and categories by volume and average rating.",
])
heading("9.2 Potential tools", 2)
table(
    ["Purpose", "Tool"],
    [
        ["Data handling", "Python, pandas, NumPy"],
        ["Visualisation", "Matplotlib, Seaborn"],
        ["Machine learning", "scikit-learn"],
        ["Text analysis", "NLTK / TextBlob or VADER for sentiment"],
        ["Recommenders", "Cosine similarity, matrix factorisation"],
        ["Environment", "VS Code, Jupyter notebooks, Git"],
    ],
)

if os.path.exists("chart.png"):
    doc.add_picture("chart.png", width=Inches(5.8))
    cap = doc.add_paragraph("Figure 1: Illustrative chart generated on synthetic data to "
                            "demonstrate the planned analysis. Not a real finding.")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ---------- 10 ----------
heading("10. Rationale for the Approach")
bullets([
    "Starting with EDA exposes data quality issues and tests the hypotheses before modelling.",
    "A popularity baseline provides a fair benchmark for any more advanced recommender.",
    "Similarity-based methods are interpretable and work well when user history is limited.",
    "Combining ratings with review text captures both what customers think and why.",
    "Tree-based and logistic models give both performance and explainability.",
])

# ---------- 11 ----------
heading("11. Risks, Limitations and Ethics")
bullets([
    "Reviews may be biased toward extreme opinions or influenced by free samples.",
    "Missing profile data may limit personalisation.",
    "Skin reactions vary, so recommendations are not medical advice.",
    "Customer data must be anonymised and used in line with privacy rules and dataset licences.",
    "Models may under-serve groups that are under-represented in the data.",
])

# ---------- 12 ----------
heading("12. References")
para("[Replace with the sources you actually read. Suggested starting points:]")
bullets([
    "Industry reports on the beauty and personal care market (McKinsey 'State of Beauty', "
    "Statista, Grand View Research).",
    "Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for "
    "recommender systems. IEEE Computer, 42(8).",
    "Kaggle: public skincare product and review datasets.",
    "Open Beauty Facts: open product database.",
])

out = "Week1_Beauty_Wellness_Problem_Definition.docx"
doc.save(out)
print(f"Saved {out}")