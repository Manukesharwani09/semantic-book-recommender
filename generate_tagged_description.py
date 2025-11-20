import pandas as pd

books = pd.read_csv("books_cleaned.csv")

books["tagged"] = books["isbn13"].astype(str) + " " + books["description"].astype(str)

with open("tagged_description.txt", "w", encoding="utf-8") as f:
    for line in books["tagged"]:
        f.write(line + "\n")

print("✅ tagged_description.txt generated with UTF-8 encoding!")
