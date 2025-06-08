import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    def __init__(self, books_path, book_tags_path, tags_path):
        self.books = pd.read_csv(books_path)
        self.book_tags = pd.read_csv(book_tags_path)
        self.tags = pd.read_csv(tags_path)
        self._prepare_data()

    def _prepare_data(self):
        tag_dict = {}
        for i,j in zip(self.tags.tag_id, self.tags.tag_name):
            tag_dict[i] = j

        mapped_tags = [tag_dict.get(tag_id, '') for tag_id in self.book_tags.tag_id]
        self.book_tags['tag'] = mapped_tags

        tag_lists = {}
        for book_id, group in zip(self.book_tags.goodreads_book_id, self.book_tags.tag):
            tag_lists.setdefault(book_id, []).append(group)

        goodreads_ids = list(tag_lists.keys())
        tag_data = pd.DataFrame({'goodreads_book_id': goodreads_ids, 'tag': [' '.join(tag_lists[book_id]) for book_id in goodreads_ids]})

        self.books = self.books.merge(tag_data, left_on='book_id', right_on='goodreads_book_id', how='left')
        self.books['tag'] = self.books['tag'].fillna('')

        self._vectorize()

    def _vectorize(self):
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.books['tag'])
        self.similarity = cosine_similarity(self.tfidf_matrix)

    def recommend(self, title, n=5):
        matches = self.books[self.books['title'].str.lower() == title.lower()]
        if matches.empty:
            return
            
        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        for i, _ in sim_scores:
            print(self.books.iloc[i]['title'])
