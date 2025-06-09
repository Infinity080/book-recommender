import sys
import types
import torch
if isinstance(torch.classes, types.ModuleType):
    sys.modules["torch.classes"] = types.SimpleNamespace()
import streamlit as st
from src.BookRecommender import BookRecommender
from src.MovieRecommender import MovieRecommender
import kagglehub
import os
import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Media Recommender", layout="centered")
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        .stButton > button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("# Media Recommender")
st.markdown("### Choose media type")

if "mode" not in st.session_state:
    st.session_state.mode = None

col1, col2, _, _ = st.columns(4)
if col1.button("Books"):
    st.session_state.mode = "Books"
if col2.button("Movies"):
    st.session_state.mode = "Movies"

mode = st.session_state.mode


if mode == "Books":
    st.title("Book Recommender")
    st.write("Get recommendations based on your previously read books or based on a description you provide.")

    data_dir = kagglehub.dataset_download("zygmunt/goodbooks-10k")

    recommender = BookRecommender(
        books_path=os.path.join(data_dir, 'books.csv'),
        book_tags_path=os.path.join(data_dir, 'book_tags.csv'),
        tags_path=os.path.join(data_dir, 'tags.csv')
    )

    liked_books = {}
    st.subheader("Rate Books You Read")

    st.divider()
    for i in range(1, 4):
        col1, col2 = st.columns([3, 1])
        with col1:
            title = st.text_input(
                f"Book {i}", key=f"title_{i}", placeholder="book title")
        with col2:
            rating = st.slider(
                "Rating", 1, 5, 5, key=f"rating_{i}", label_visibility="collapsed") - 3

        if title:
            all_titles = recommender.books['title'].tolist()
            all_titles_lower = [t.lower() for t in all_titles]
            match = process.extractOne(
                title.lower(), all_titles_lower, scorer=fuzz.ratio, score_cutoff=70)

            if match is None:
                st.warning(f"Book {i} not found: \"{title}\"")
            else:
                matched_lower = match[0]
                matched_row = recommender.books[recommender.books['title'].str.lower(
                ) == matched_lower].iloc[0]
                matched_title = matched_row['title']
                author = matched_row.get('authors', 'Unknown author')

                if match[1] >= 90:
                    st.success(
                        f"Book {i} found: \"{matched_title}\" by {author}")
                    liked_books[matched_title] = rating
                else:
                    st.warning(
                        f"Book {i}: \"{matched_title}\" (similarity: {match[1]:.1f}%)")
                    confirm = st.checkbox(
                        f"Did you mean \"{matched_title}\" by {author}?",
                        key=f"confirm_book_{i}"
                    )
                    if confirm:
                        liked_books[matched_title] = rating

    st.divider()
    query = st.text_area("Or describe what you like (optional)",
                         placeholder="book description")

    button_col1, button_col2 = st.columns(2)

    show_liked_recommendations = button_col1.button(
        "Get Recommendations from Books")
    show_text_recommendations = button_col2.button(
        "Get Recommendations from Description")

    if show_liked_recommendations:
        if not liked_books:
            st.warning("Enter at least one book.")
        else:
            sim_vector = None
            seen = set()
            for title, weight in liked_books.items():
                matches = recommender.books[recommender.books['title'].str.lower(
                ) == title.lower()]
                if matches.empty:
                    continue
                idx = matches.index[0]
                vec = recommender.embeddings[idx] * weight
                sim_vector = vec if sim_vector is None else sim_vector + vec
                seen.add(title.lower())

            if sim_vector is None:
                st.warning("Books not found.")
            else:
                scores = cosine_similarity(
                    [sim_vector], recommender.embeddings).flatten()
                top_indices = scores.argsort()[::-1]
                st.subheader("Recommended Books")
                count = 0
                for i in top_indices:
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

    if show_text_recommendations:
        if not query.strip():
            st.warning("Please enter a description.")
        else:
            results = recommender.recommend_from_text(query)
            st.subheader("Recommended Books")
            for book in results:
                col1, col2 = st.columns([1, 4])
                with col1:
                    if pd.notna(book.get('image_url', '')) and book['image_url'].startswith('http'):
                        st.image(book['image_url'], width=100)
                with col2:
                    st.markdown(f"**{book['title']}**")
                    st.markdown(f"*{book['authors']}*")


elif mode == "Movies":
    st.title("Movie Recommender")
    st.write(
        "Get recommendations based on your previously watched movies or based on a description you provide.")

    data_dir = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    movie_recommender = MovieRecommender(
        os.path.join(data_dir, "tmdb_5000_movies.csv"))

    liked_movies = {}
    st.subheader("Rate Movies You Watched")

    st.divider()
    for i in range(1, 4):
        col1, col2 = st.columns([3, 1])
        with col1:
            title = st.text_input(
                f"Movie {i}", key=f"movie_title_{i}", placeholder="movie title")
        with col2:
            rating = st.slider(
                "Rating", 1, 5, 5, key=f"movie_rating_{i}", label_visibility="collapsed") - 3

        if title:
            all_titles = movie_recommender.movies['title'].tolist()
            all_titles_lower = [t.lower() for t in all_titles]
            match = process.extractOne(
                title.lower(), all_titles_lower, scorer=fuzz.ratio, score_cutoff=70)

            if match is None:
                st.warning(f"Movie {i} not found: \"{title}\"")
            else:
                matched_title, score, _ = match
                if score >= 90:
                    st.success(
                        f"Movie {i} matched: \"{matched_title}\" (score: {score:.1f})")
                    liked_movies[matched_title] = rating
                else:
                    st.warning(
                        f"Movie {i}: \"{matched_title}\" (similarity: {score:.1f}%)")
                    confirm = st.checkbox(
                        f"Did you mean {matched_title}?", key=f"confirm_movie_{matched_title}_{i}")
                    if confirm:
                        liked_movies[matched_title] = rating

    st.divider()
    query = st.text_area("Or describe what you like (optional)",
                         placeholder="movie description")

    col1, col2 = st.columns(2)
    show_liked_recommendations = col1.button("Get Recommendations from Movies")
    show_text_recommendations = col2.button(
        "Get Recommendations from Description")

    if show_liked_recommendations:
        if not liked_movies:
            st.warning("Enter at least one movie.")
        else:
            sim_vector = None
            seen = set()
            for title, weight in liked_movies.items():
                matches = movie_recommender.movies[movie_recommender.movies['title'].str.lower(
                ) == title.lower()]
                if matches.empty:
                    continue
                idx = matches.index[0]
                vec = movie_recommender.embeddings[idx] * weight
                sim_vector = vec if sim_vector is None else sim_vector + vec
                seen.add(title.lower())

            if sim_vector is None:
                st.warning("Movies not found.")
            else:
                scores = cosine_similarity(
                    [sim_vector], movie_recommender.embeddings).flatten()
                top_indices = scores.argsort()[::-1]
                st.subheader("Recommended Movies")
                count = 0
                for i in top_indices:
                    movie = movie_recommender.movies.iloc[i]
                    if movie['title'].lower() in seen:
                        continue
                    st.markdown(
                        f"**{movie['title']}**  \n*{movie.get('release_date', 'Unknown')}*")
                    st.markdown(movie.get('overview', ''))
                    count += 1
                    if count == 5:
                        break

    if show_text_recommendations:
        if not query.strip():
            st.warning("Please enter a description.")
        else:
            results = movie_recommender.recommend_from_description(query)
            st.subheader("Recommended Movies")
            for movie in results:
                st.markdown(
                    f"**{movie['title']}**  \n*{movie.get('release_date', 'Unknown')}*")
                st.markdown(movie.get('overview', ''))
