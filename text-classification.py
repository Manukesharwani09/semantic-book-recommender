# text-classification.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# Load Cleaned Dataset
# ----------------------------------------
INPUT_FILE = "books_cleaned.csv"

books = pd.read_csv(INPUT_FILE)
print(f"Loaded dataset: {INPUT_FILE}")
print("Shape:", books.shape)


# ----------------------------------------
# STEP 1 — CREATE simple_categories column from categories
# ----------------------------------------
category_mapping = {
    'Fiction': "Fiction",
    'Juvenile Fiction': "Children's Fiction",
    'Biography & Autobiography': "Nonfiction",
    'History': "Nonfiction",
    'Literary Criticism': "Nonfiction",
    'Philosophy': "Nonfiction",
    'Religion': "Nonfiction",
    'Comics & Graphic Novels': "Fiction",
    'Drama': "Fiction",
    'Juvenile Nonfiction': "Children's Nonfiction",
    'Science': "Nonfiction",
    'Poetry': "Fiction"
}

# Map categories
books["simple_categories"] = books["categories"].map(category_mapping)

print("Unique simple_categories after mapping:")
print(books["simple_categories"].value_counts(dropna=False))


# ----------------------------------------
# Device selection (MPS if available)
# ----------------------------------------
if torch.backends.mps.is_available():
    device = 0
    print("Using Apple MPS acceleration")
else:
    device = -1
    print("Using CPU")


# ----------------------------------------
# Load Zero-Shot Classification Pipeline
# ----------------------------------------
pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

fiction_categories = ["Fiction", "Nonfiction"]


# ----------------------------------------
# Helper: clean text
# ----------------------------------------
def clean_text(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return "No meaningful description available."
    return text.strip()


# ----------------------------------------
# Helper: classify
# ----------------------------------------
def generate_predictions(sequence, categories):
    sequence = clean_text(sequence)
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])
    return predictions["labels"][max_index]


# ----------------------------------------
# STEP 2 — Predict categories for missing rows
# ----------------------------------------
missing_rows = books[books["simple_categories"].isna()]
print("Missing categories to classify:", len(missing_rows))

isbns = []
predicted_cats = []

for i in tqdm(range(len(missing_rows))):
    description = missing_rows.iloc[i]["description"]
    isbn = missing_rows.iloc[i]["isbn13"]

    predicted = generate_predictions(description, fiction_categories)

    isbns.append(isbn)
    predicted_cats.append(predicted)

pred_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})


# ----------------------------------------
# Merge predictions
# ----------------------------------------
books = pd.merge(books, pred_df, on="isbn13", how="left")

books["simple_categories"] = np.where(
    books["simple_categories"].isna(),
    books["predicted_categories"],
    books["simple_categories"]
)

books = books.drop(columns=["predicted_categories"])

# ----------------------------------------
# Save output
# ----------------------------------------
OUTPUT_FILE = "books_with_categories.csv"
books.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved dataset as: {OUTPUT_FILE}")
print("Final Shape:", books.shape)
