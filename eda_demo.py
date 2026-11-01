import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 2000

df = pd.DataFrame({
    "skin_type": rng.choice(["Dry", "Oily", "Combination", "Normal", "Sensitive"], n),
    "category": rng.choice(["Moisturizer", "Serum", "Cleanser", "Sunscreen", "Mask"], n),
    "price_usd": rng.gamma(3, 15, n).round(2),
    "rating": rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.07, 0.13, 0.30, 0.45]),
})
df["is_recommended"] = (df["rating"] >= 4).astype(int)

print(df.head())
print(df.describe(include="all"))

pivot = df.pivot_table(index="skin_type", columns="category",
                       values="rating", aggfunc="mean")

ax = pivot.plot(kind="bar", figsize=(9, 5))
ax.set_title("Illustrative: average rating by skin type and category (synthetic data)")
ax.set_ylabel("Average rating")
ax.set_xlabel("Skin type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart.png", dpi=150)
print("Saved chart.png")