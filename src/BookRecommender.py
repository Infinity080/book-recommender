import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

class BookRecommender:
    def __init__(self, books_path, model, book_tags_path, tags_path,
                 embeddings_path="book_embeddings.npy",
                 features_path="book_features.npy",
                 cache=True):
        self.embeddings_path = embeddings_path
        self.features_path = features_path
        self.cache = cache
        self.model = model

        self.books = pd.read_csv(books_path)
        self.book_tags = pd.read_csv(book_tags_path)
        self.tags = pd.read_csv(tags_path)
        self._prepare_data()

    def _prepare_data(self):
        tag_dict = {i: j for i, j in zip(self.tags.tag_id, self.tags.tag_name)}
        mapped_tags = [tag_dict.get(tag_id, '')
                       for tag_id in self.book_tags.tag_id]
        self.book_tags['tag'] = mapped_tags

        tag_lists = {}
        for book_id, tag in zip(self.book_tags.goodreads_book_id, self.book_tags.tag):
            tag_lists.setdefault(book_id, []).append(tag)

        tag_data = pd.DataFrame({
            'goodreads_book_id': list(tag_lists.keys()),
            'tag': [' '.join(tag_lists[bid]) for bid in tag_lists]
        })

        self.books = self.books.merge(
            tag_data, left_on='book_id', right_on='goodreads_book_id', how='left')
        self.books['tag'] = self.books['tag'].fillna('')
        self.books['features'] = (
            self.books['tag'].fillna('') + ' ' +
            self.books['title'].fillna('') + ' ' +
            self.books['authors'].fillna('')
        )

        if self.cache and os.path.exists(self.features_path):
            self.books['features'] = np.load(
                self.features_path, allow_pickle=True)
        elif self.cache:
            np.save(self.features_path, self.books['features'].values)

        self._vectorize()

    def _vectorize(self):
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.getcwd()

        if self.cache and os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
        else:
            self.embeddings = self.model.encode(
                self.books["features"].tolist(),
                show_progress_bar=True
            )
            if self.cache:
                np.save(self.embeddings_path, self.embeddings)

    def recommend_from_text(self, query, n=5):
        query_embedding = self.model.encode([query])
        sim_scores = cosine_similarity(
            query_embedding, self.embeddings).flatten()
        top_indices = sim_scores.argsort()[::-1]
        results = []
        seen = set()
        for i in top_indices:
            title = self.books.iloc[i]['title']
            if title.lower() not in seen:
                results.append(self.books.iloc[i])
                seen.add(title.lower())
            if len(results) >= n:
                break
        return results

    def recommend(self, title, n=5):
        matches = self.books[self.books['title'].str.lower() == title.lower()]
        if matches.empty:
            print("No match found.")
            return

        idx = matches.index[0]
        query_vec = self.embeddings[idx].reshape(1, -1)
        sim_scores = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = sim_scores.argsort()[::-1]

        seen = {title.lower()}
        for i in top_indices:
            candidate = self.books.iloc[i]['title']
            if candidate.lower() not in seen:
                print("-", candidate)
                seen.add(candidate.lower())
            if len(seen) - 1 >= n:
                break

    def recommend_multiple(self, liked_books: dict, n=5):
        sim_vector = None
        found_titles = []

        for title, weight in liked_books.items():
            matches = self.books[self.books['title'].str.lower()
                                 == title.lower()]
            if matches.empty:
                continue
            idx = matches.index[0]
            vec = self.embeddings[idx] * weight
            sim_vector = vec if sim_vector is None else sim_vector + vec
            found_titles.append(title.lower())

        if sim_vector is None:
            print("No books found.")
            return

        sim_scores = cosine_similarity([sim_vector], self.embeddings).flatten()
        top_indices = sim_scores.argsort()[::-1]
        seen = set(found_titles)
        recommendations = []

        for i in top_indices:
            candidate = self.books.iloc[i]['title']
            if candidate.lower() not in seen:
                recommendations.append(candidate)
            if len(recommendations) >= n:
                break

        print("Similar books:", ", ".join(found_titles))
        for title in recommendations:
            print("-", title)
