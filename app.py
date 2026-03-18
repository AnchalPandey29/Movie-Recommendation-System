import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
    }
    .movie-card {
        padding: 15px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1f1f2e, #2c2c3e);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    df["Summary"] = df["Summary"].fillna("").str.lower().str.strip()
    df["Film Name"] = df["Film Name"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

    df = df.dropna(subset=["Summary"]).reset_index(drop=True)

    return df

df = load_data()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tf_matrix = tfidf.fit_transform(data["Summary"])

    knn = NearestNeighbors(
        n_neighbors=6,
        metric="cosine",
        algorithm="brute"
    )
    knn.fit(tf_matrix)

    return tfidf, tf_matrix, knn

tfidf, tf_matrix, knn = load_model(df)

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie_name):
    movie_name = movie_name.lower().strip()
    movie_list = df['Film Name'].str.lower().str.strip().values

    if movie_name not in movie_list:
        return None

    index = df[df['Film Name'].str.lower().str.strip() == movie_name].index[0]
    distances, indices = knn.kneighbors(tf_matrix[index])

    results = []
    for i in range(1, len(indices[0])):
        movie_index = indices[0][i]

        results.append({
            "name": df.iloc[movie_index]["Film Name"],
            "rating": df.iloc[movie_index]["Ratings"],
            "year": df.iloc[movie_index]["Year"]
        })

    return results

# ---------------- HEADER ----------------
st.title("🎬 AI Movie Recommender System")
st.markdown("Discover movies similar to your favorites using NLP + Machine Learning")

# ---------------- SEARCH UI ----------------
col1, col2 = st.columns([3,1])

with col1:
    movie_input = st.text_input("🔍 Enter a movie name")

with col2:
    st.write("")
    st.write("")
    search_btn = st.button("Recommend 🎯")

# ---------------- RESULTS ----------------
if search_btn and movie_input:

    results = recommend(movie_input)

    if results is None:
        st.error("❌ Movie not found. Try another one.")
    else:
        st.success(f"✨ Recommendations for **{movie_input.title()}**")

        cols = st.columns(3)

        for i, movie in enumerate(results):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="movie-card">
                        <h4>🎬 {movie['name']}</h4>
                        <p>⭐ Rating: {movie['rating']}</p>
                        <p>📅 Year: {movie['year']}</p>
                    </div>
                """, unsafe_allow_html=True)

# ---------------- ANALYTICS ----------------
st.markdown("---")
st.subheader("📊 Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("### Ratings Distribution")
    df["Ratings"] = pd.to_numeric(df["Ratings"], errors='coerce')
    st.bar_chart(df["Ratings"].value_counts().sort_index())

with col2:
    st.write("### Top 10 Movies")
    top_movies = df.sort_values(by="Ratings", ascending=False).head(10)
    st.dataframe(top_movies[["Film Name", "Ratings", "Year"]])

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | NLP Recommendation Engine")
