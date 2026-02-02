import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("❌ OPENAI_API_KEY not found. Please set it in .env file.")

# Load dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Updated embeddings init
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Persistent vector database (avoids rebuilding on every startup)
CHROMA_DIR = "./chroma_db"

if os.path.exists(CHROMA_DIR):
    print("📚 Loading existing vector database...")
    db_books = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    print("🔨 Building vector database (first run)...")
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DIR)
    print("✅ Vector database saved to disk.")


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # pandas 2.x safe sorting
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }

    if tone in tone_map:
        book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs


def recommend_books(query, category, tone):
    # Validate input
    if not query or len(query.strip()) < 3:
        gr.Warning("Please enter at least 3 characters to search.")
        return []
    
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
    except Exception as e:
        gr.Error(f"Search failed: {str(e)}")
        return []
    
    if recommendations.empty:
        gr.Info("No books found matching your criteria. Try different filters.")
        return []
    
    results = []

    for _, row in recommendations.iterrows():
        # Handle NaN descriptions safely
        description = row["description"] if pd.notna(row["description"]) else "No description available."
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Handle NaN authors safely
        authors = row["authors"] if pd.notna(row["authors"]) else "Unknown Author"
        authors_split = authors.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors

        # Enhanced caption with rating and year
        rating = row.get("average_rating", 0)
        stars = "⭐" * int(round(rating)) if pd.notna(rating) else ""
        year = int(row["published_year"]) if pd.notna(row.get("published_year")) else "N/A"
        
        caption = f"{row['title']} by {authors_str} ({year}) {stars}\n{truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Example queries for users
EXAMPLE_QUERIES = [
    "A story about forgiveness and redemption",
    "An adventure in a magical fantasy world",
    "A thrilling mystery with unexpected twists",
    "A heartwarming tale of friendship",
    "A journey of self-discovery and growth",
]

with gr.Blocks(title="Book Recommender") as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")
    gr.Markdown("Find your next favorite book using AI-powered semantic search!")

    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(
                label="Describe the kind of book you're looking for:",
                placeholder="e.g., A story about forgiveness",
                lines=2
            )
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories, 
                label="Category:", 
                value="All"
            )
            tone_dropdown = gr.Dropdown(
                choices=tones, 
                label="Emotional tone:", 
                value="All"
            )
    
    with gr.Row():
        submit_button = gr.Button("🔍 Find Recommendations", variant="primary", scale=2)
        clear_button = gr.Button("🗑️ Clear", scale=1)
    
    gr.Markdown("### 💡 Example searches:")
    with gr.Row():
        for example in EXAMPLE_QUERIES[:3]:
            gr.Button(example, size="sm").click(
                fn=lambda x=example: x,
                outputs=user_query
            )

    gr.Markdown("## 📖 Recommendations")
    output = gr.Gallery(
        label="Recommended books", 
        columns=4, 
        rows=4,
        height="auto",
        object_fit="contain"
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )
    
    # Allow Enter key to submit
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )
    
    clear_button.click(
        fn=lambda: ("", "All", "All", []),
        outputs=[user_query, category_dropdown, tone_dropdown, output]
    )

if __name__ == "__main__":
    # Cloud-ready launch configuration
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        theme=gr.themes.Glass()
    )