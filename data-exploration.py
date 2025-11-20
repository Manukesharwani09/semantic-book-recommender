# data-exploration.py
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# Load Dataset
# ----------------------------------------
CSV_FILE = "books.csv"  # Make sure this file exists in your folder

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"❌ CSV file not found: {CSV_FILE}")

books = pd.read_csv(CSV_FILE)
print(f"Loaded dataset: {CSV_FILE}")
print("Shape:", books.shape)
print("Columns:", books.columns.tolist())


# ----------------------------------------
# OPTIONAL: Missing value heatmap (disabled)
# Commented out to prevent Tkinter popup freezing
# ----------------------------------------
# ax = plt.axes()
# sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)
# plt.xlabel("Columns")
# plt.ylabel("Missing values")
# plt.savefig("missing_values_heatmap.png")  # Saves instead of showing
# plt.close()


# ----------------------------------------
# Create new numeric features
# ----------------------------------------
books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2024 - books["published_year"]


# ----------------------------------------
# OPTIONAL: Correlation heatmap (disabled)
# ----------------------------------------
# columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]
# correlation_matrix = books[columns_of_interest].corr(method="spearman")
# sns.set_theme(style="white")
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation heatmap")
# plt.savefig("correlation_heatmap.png")
# plt.close()


# ----------------------------------------
# Filter out books missing important fields
# ----------------------------------------
book_missing = books[
    ~(books["description"].isna()) &
    ~(books["num_pages"].isna()) &
    ~(books["average_rating"].isna()) &
    ~(books["published_year"].isna())
].copy()


# ----------------------------------------
# Keep only books with decent description length
# ----------------------------------------
book_missing["words_in_description"] = (
    book_missing["description"].astype(str).str.split().str.len()
)

book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25].copy()


# ----------------------------------------
# Combine title + subtitle
# ----------------------------------------
book_missing_25_words["title_and_subtitle"] = np.where(
    book_missing_25_words["subtitle"].isna(),
    book_missing_25_words["title"],
    book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1)
)


# ----------------------------------------
# Create text input for embeddings
# ----------------------------------------
book_missing_25_words["tagged_description"] = (
    book_missing_25_words[["isbn13", "description"]]
    .astype(str)
    .agg(" ".join, axis=1)
)


# ----------------------------------------
# Save cleaned dataset
# ----------------------------------------
OUTPUT_FILE = "books_cleaned.csv"

book_missing_25_words.drop(
    ["subtitle", "missing_description", "age_of_book", "words_in_description"],
    axis=1
).to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved cleaned dataset as: {OUTPUT_FILE}")
print("Final shape:", book_missing_25_words.shape)
