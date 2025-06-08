import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

class MovieRecommender:
    def __init__(self, movie_path):
        self.movies = pd.read_csv(movie_path)
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

        features = []
        for i in range(len(self.movies)):
            genre = self.movies.at[i, 'genres_text']
            keyword = self.movies.at[i, 'keywords_text']
            tag = self.movies.at[i, 'tagline']
            overview = self.movies.at[i, 'overview']
            features.append(f"{genre} {keyword} {tag} {overview}")

        self.movies['features'] = features

        self._vectorize()

    def _vectorize(self):
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['features'])
        self.similarity = cosine_similarity(self.tfidf_matrix)

    def recommend(self, title, n=5):
        matches = self.movies[self.movies['title'].str.lower() == title.lower()]
        if matches.empty:
            return

        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        for i, _ in sim_scores:
            print(self.movies.iloc[i]['title'])

    def recommend_multiple(self, liked_movies: dict, n=5):
        sim_vector = None
        found_titles = []

        for title, weight in liked_movies.items():
            matches = self.movies[self.movies['title'].str.lower() == title.lower()]
            if matches.empty:
                continue
            idx = matches.index[0]
            vec = self.similarity[idx] * weight
            sim_vector = vec if sim_vector is None else sim_vector + vec
            found_titles.append(title)

        if sim_vector is None:
            print("No movies found.")
            return

        sim_scores = list(enumerate(sim_vector))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        seen = set([b.lower() for b in liked_movies.keys()])
        recommendations = []

        for i, _ in sim_scores:
            candidate = self.movies.iloc[i]['title']
            if candidate.lower() not in seen:
                recommendations.append(candidate)
            if len(recommendations) >= n:
                break

        print("Similar movies:", ", ".join(found_titles))
        for title in recommendations:
            print("-", title)
