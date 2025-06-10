import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast
import torch
import os
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_model(model_dir, device):
    if os.path.exists(model_dir):
        model = SentenceTransformer(model_dir)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(model_dir)
    return model.to(device)

class MovieRecommender:
    def __init__(self, movie_path, embeddings_path="movie_embeddings.npy", features_path="movie_features.npy", cache=True):
        self.movie_path = movie_path
        self.embeddings_path = embeddings_path
        self.features_path = features_path
        self.cache = cache
        self.movies = pd.read_csv(movie_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        self._prepare_data()

    def _prepare_data(self):
        genres_text = []
        keywords_text = []

        for i in range(len(self.movies)):
            try:
                genres = ast.literal_eval(self.movies.at[i, 'genres'])
                genres_text.append(' '.join([d['name'] for d in genres]))
            except:
                genres_text.append('')
            try:
                keywords = ast.literal_eval(self.movies.at[i, 'keywords'])
                keywords_text.append(' '.join([d['name'] for d in keywords]))
            except:
                keywords_text.append('')

        self.movies['genres_text'] = genres_text
        self.movies['keywords_text'] = keywords_text

        if self.cache and os.path.exists(self.features_path):
            self.features = np.load(self.features_path, allow_pickle=True)
            self.movies['features'] = self.features
        else:
            self.features = (
                self.movies['genres_text'].fillna('') + ' ' +
                self.movies['keywords_text'].fillna('') + ' ' +
                self.movies['tagline'].fillna('') + ' ' +
                self.movies['overview'].fillna('')
            )
            self.movies['features'] = self.features
            if self.cache:
                np.save(self.features_path, self.features.to_numpy())

        if self.cache and os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
        else:
            self._vectorize()
            if self.cache:
                np.save(self.embeddings_path, self.embeddings)

    def _vectorize(self):
        device = 'cpu'
        model_dir = "models/all-MiniLM-L6-v2"

        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.getcwd()

        self.model = load_model(model_dir, device)

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
        else:
            self.embeddings = self.model.encode(
                self.features.tolist(),
                show_progress_bar=True
            )
            np.save(self.embeddings_path, self.embeddings)
    
    def recommend_from_description(self, query, n=5):
        query_embedding = self.model.encode([query])
        sim_scores = cosine_similarity(
            query_embedding, self.embeddings).flatten()
        top_indices = sim_scores.argsort()[::-1]
        results = []
        seen = set()
        for i in top_indices:
            title = self.movies.iloc[i]['title']
            if title.lower() not in seen:
                results.append(self.movies.iloc[i])
                seen.add(title.lower())
            if len(results) >= n:
                break
        return results

    def recommend(self, title, n=5):
        matches = self.movies[self.movies['title'].str.lower()
                              == title.lower()]
        if matches.empty:
            print("No match found.")
            return

        idx = matches.index[0]
        query_vec = self.embeddings[idx].reshape(1, -1)
        sim_scores = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = sim_scores.argsort()[::-1]

        seen = {title.lower()}
        for i in top_indices:
            candidate = self.movies.iloc[i]['title']
            if candidate.lower() not in seen:
                print("-", candidate)
                seen.add(candidate.lower())
            if len(seen) - 1 >= n:
                break

    def recommend_multiple(self, liked_movies: dict, n=5):
        sim_vector = None
        found_titles = []

        for title, weight in liked_movies.items():
            matches = self.movies[self.movies['title'].str.lower()
                                  == title.lower()]
            if matches.empty:
                continue
            idx = matches.index[0]
            vec = self.embeddings[idx] * weight
            sim_vector = vec if sim_vector is None else sim_vector + vec
            found_titles.append(title.lower())

        if sim_vector is None:
            print("No movies found.")
            return

        sim_scores = cosine_similarity([sim_vector], self.embeddings).flatten()
        top_indices = sim_scores.argsort()[::-1]
        seen = set(found_titles)
        recommendations = []

        for i in top_indices:
            candidate = self.movies.iloc[i]['title']
            if candidate.lower() not in seen:
                recommendations.append(candidate)
            if len(recommendations) >= n:
                break

        print("Similar movies:", ", ".join(found_titles))
        for title in recommendations:
            print("-", title)
