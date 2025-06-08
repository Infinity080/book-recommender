import streamlit
from recommend import BookRecommender
import kagglehub
import os

data_dir = kagglehub.dataset_download("zygmunt/goodbooks-10k")

recommender = BookRecommender(
    books_path=os.path.join(data_dir, 'books.csv'),
    book_tags_path=os.path.join(data_dir, 'book_tags.csv'),
    tags_path=os.path.join(data_dir, 'tags.csv')
)

streamlit.title("Book Recommender")

title = streamlit.text_input("Enter a title")
if streamlit.button("Recommend"):
    streamlit.write("Similar books:")
    matches = recommender.books[recommender.books['title'].str.lower() == title.lower()]
    if matches.empty:
        streamlit.warning("Book not found.")
    else:
        idx = matches.index[0]
        sims = recommender.similarity[idx]
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[1:6]
        for i in top_indices:
            streamlit.write(f"- {recommender.books.iloc[i]['title']}")
