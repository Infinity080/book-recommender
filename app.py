import streamlit as st
from recommend import BookRecommender
import kagglehub
import os
import pandas as pd

st.set_page_config(page_title="Book Recommender", layout="centered")

st.title("Book Recommender")
st.write("Enter up to 3 books you like and rate them.")

data_dir = kagglehub.dataset_download("zygmunt/goodbooks-10k")

recommender = BookRecommender(
    books_path=os.path.join(data_dir, 'books.csv'),
    book_tags_path=os.path.join(data_dir, 'book_tags.csv'),
    tags_path=os.path.join(data_dir, 'tags.csv')
)

liked_books = {}

st.divider()
for i in range(1, 4):
    col1, col2 = st.columns([3, 1])
    with col1:
        title = st.text_input(f"Book {i}", key=f"title_{i}", placeholder="book title")
    with col2:
        rating = st.slider("Rating", 1, 5, 5, key=f"rating_{i}", label_visibility="collapsed") - 3
    if title:
        matches = recommender.books[recommender.books['title'].str.lower() == title.lower()]
        if matches.empty:
            st.warning(f"Book {i} not found: \"{title}\"")
        else:
            matched_title = matches.iloc[0]['title']
            st.success(f"Book {i} matched: \"{matched_title}\"")
            liked_books[matched_title] = rating

st.divider()

if st.button("Get Recommendations"):
    if not liked_books:
        st.warning("Enter at least one book.")
    else:
        sim_vector = None
        seen = set()
        for title, weight in liked_books.items():
            matches = recommender.books[recommender.books['title'].str.lower() == title.lower()]
            if matches.empty:
                continue
            idx = matches.index[0]
            vec = recommender.similarity[idx] * weight
            sim_vector = vec if sim_vector is None else sim_vector + vec
            seen.add(title.lower())

        if sim_vector is None:
            st.warning("Books not found.")
        else:
            scores = sorted(enumerate(sim_vector), key=lambda x: x[1], reverse=True)
            st.subheader("Recommended Books")
            count = 0
            for i, _ in scores:
                book = recommender.books.iloc[i]
                if book['title'].lower() in seen:
                    continue

                col1, col2 = st.columns([1, 4])
                with col1:
                    if pd.notna(book.get('image_url', '')) and book['image_url'].startswith('http'):
                        st.image(book['image_url'], width=100)
                with col2:
                    st.markdown(f"**{book['title']}**")
                    st.markdown(f"*{book['authors']}*")
                count += 1
                if count == 5:
                    break
