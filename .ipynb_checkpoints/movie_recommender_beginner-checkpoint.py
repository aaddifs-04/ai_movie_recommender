import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(movies_path="tmdb_5000_movies.csv", credits_path="tmdb_5000_credits.csv"):
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    credits.rename(columns={"movie_id": "id"}, inplace=True)
    credits.drop(columns=["title"], inplace=True)
    movies = movies.merge(credits, on="id")

    return movies


def select_features(movies):
    movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    return movies


def clean_data(movies):
    movies.dropna(inplace=True)
    movies.drop_duplicates(inplace=True)

    return movies


def parse_names(text):
    names = []
    for item in ast.literal_eval(text):
        names.append(item["name"])
    return names


def parse_top_cast(text, top_n=3):
    names = []
    for i, item in enumerate(ast.literal_eval(text)):
        if i >= top_n:
            break
        names.append(item["name"])
    return names


def parse_director(text):
    for item in ast.literal_eval(text):
        if item.get("job") == "Director":
            return [item["name"]]
    return []


def build_tags(movies):
    movies = movies.copy()

    movies["genres"] = movies["genres"].apply(parse_names)
    movies["keywords"] = movies["keywords"].apply(parse_names)
    movies["cast"] = movies["cast"].apply(parse_top_cast)
    movies["director"] = movies["crew"].apply(parse_director)
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    for col in ["genres", "keywords", "cast", "director"]:
        movies[col] = movies[col].apply(lambda items: [i.replace(" ", "") for i in items])

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["director"]
    )
    movies["tags"] = movies["tags"].apply(lambda tags: " ".join(tags).lower())

    return movies[["id", "title", "tags"]]


def compute_similarity(movies, max_features=5000):
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    vectors = vectorizer.fit_transform(movies["tags"]).toarray()

    similarity = cosine_similarity(vectors)
    return similarity


def recommend(movie, movies, similarity, top_n=5):
    matches = movies[movies["title"].str.lower() == movie.lower()]
    if matches.empty:
        print(f"'{movie}' not found in the dataset.")
        return []

    index = matches.index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in scores[1:top_n + 1]:
        recommendations.append(movies.iloc[i].title)

    print(f"\nTop {top_n} movies similar to '{movie}':")
    for i, title in enumerate(recommendations, start=1):
        print(f"{i}. {title}")

    return recommendations


if __name__ == "__main__":
    raw = load_data("tmdb_5000_movies.csv", "tmdb_5000_credits.csv")
    raw = select_features(raw)
    raw = clean_data(raw)

    movies = build_tags(raw)
    movies.reset_index(drop=True, inplace=True)

    similarity = compute_similarity(movies)
    print(f"Similarity matrix shape: {similarity.shape}")

    while True:
        user_input = input("\nEnter a movie name (or type 'quit' to exit): ")

        if user_input.lower() == "quit":
            break

        recommend(user_input, movies, similarity)