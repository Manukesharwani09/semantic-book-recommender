# sentiment-analysis.py (FAST VERSION)

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# Load Dataset
# ----------------------------------------
INPUT_FILE = "books_with_categories.csv"
books = pd.read_csv(INPUT_FILE)
print(f"Loaded dataset: {INPUT_FILE}")

# ----------------------------------------
# Device selection
# ----------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ----------------------------------------
# Load model + tokenizer ONCE
# ----------------------------------------
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# ----------------------------------------
# Helper: Clean + split into sentences
# ----------------------------------------
def preprocess_sentences(text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return ["no meaningful description available"]
    parts = text.replace("\n", " ").split(".")
    return [s.strip() for s in parts if len(s.strip()) > 3]

# ----------------------------------------
# Helper: batched forward pass
# ----------------------------------------
def classify_batched(sentences, batch_size=32):
    scores = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=1)

        scores.extend(probs.cpu().numpy())

    return np.array(scores)  # shape: (num_sentences, 7 emotions)


# ----------------------------------------
# MAIN LOOP (very fast)
# ----------------------------------------
emotion_scores = {label: [] for label in emotion_labels}
isbn_list = []

print("\nProcessing emotion scores (batched)...\n")

for i in tqdm(range(len(books))):
    isbn_list.append(books.loc[i, "isbn13"])

    # process sentences
    sentences = preprocess_sentences(books.loc[i, "description"])

    # batched inference
    probs = classify_batched(sentences)        # shape (num_sentences, 7)
    max_emotions = probs.max(axis=0)           # best score for each label

    # save
    for idx, label in enumerate(emotion_labels):
        emotion_scores[label].append(float(max_emotions[idx]))

# ----------------------------------------
# Save output
# ----------------------------------------
emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn_list

books = pd.merge(books, emotions_df, on="isbn13")
books.to_csv("books_with_emotions.csv", index=False)

print("\n✅ FAST sentiment analysis complete!")
print("Saved: books_with_emotions.csv")
