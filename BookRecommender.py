import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

class BookRecommender:
    def __init__(self, books_path, book_tags_path, tags_path):
        self.books = pd.read_csv(books_path)
        self.book_tags = pd.read_csv(book_tags_path)
        self.tags = pd.read_csv(tags_path)
        self._prepare_data()

    def _prepare_data(self):
        tag_dict = {i: j for i, j in zip(self.tags.tag_id, self.tags.tag_name)}
        mapped_tags = [tag_dict.get(tag_id, '') for tag_id in self.book_tags.tag_id]
        self.book_tags['tag'] = mapped_tags

        tag_lists = {}
        for book_id, tag in zip(self.book_tags.goodreads_book_id, self.book_tags.tag):
            tag_lists.setdefault(book_id, []).append(tag)

        tag_data = pd.DataFrame({
            'goodreads_book_id': list(tag_lists.keys()),
            'tag': [' '.join(tag_lists[bid]) for bid in tag_lists]
        })

        self.books = self.books.merge(tag_data, left_on='book_id', right_on='goodreads_book_id', how='left')
        self.books['tag'] = self.books['tag'].fillna('')
        self.books['features'] = (
            self.books['tag'].fillna('') + ' ' +
            self.books['title'].fillna('') + ' ' +
            self.books['authors'].fillna('')
        )

        self._vectorize()

    def _vectorize(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.embeddings = self.model.encode(
            self.books["features"].fillna("").tolist(),
            show_progress_bar=True
        )

    def recommend_from_text(self, query, n=5):
        query_embedding = self.model.encode([query])
        sim_scores = cosine_similarity(query_embedding, self.embeddings).flatten()
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
            matches = self.books[self.books['title'].str.lower() == title.lower()]
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
