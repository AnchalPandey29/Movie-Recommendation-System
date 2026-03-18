import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ---------------- OMDB FREE API ----------------
def fetch_movie_data(movie_name):
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey=thewdb"
    data = requests.get(url).json()

    if data["Response"] == "True":
        return {
            "poster": data["Poster"],
            "plot": data["Plot"],
            "genre": data["Genre"]
        }
    else:
        return {
            "poster": None,
            "plot": "No description available",
            "genre": "N/A"
        }

# ---------------- CSS ----------------
st.markdown("""
<style>

body {
    background-color: #0f172a;
}

/* Card Design */
.movie-card {
    background: #1e293b;
    border-radius: 15px;
    padding: 15px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.movie-card:hover {
    transform: translateY(-5px);
}

/* Text Styling */
.title {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
}

.meta {
    color: #cbd5e1;
    font-size: 14px;
}

/* Input */
.stTextInput input {
    background-color: #1e293b !important;
    color: white !important;
}

/* Button */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    height: 45px;
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

    knn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    knn.fit(tf_matrix)

    return tf_matrix, knn

tf_matrix, knn = load_model(df)

# ---------------- RECOMMEND ----------------
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
st.title("🎬 AI Movie Recommender")
st.markdown("Discover movies with **AI-powered recommendations**")

# ---------------- SEARCH ----------------
col1, col2 = st.columns([4,1])

with col1:
    movie_input = st.text_input("🔍 Enter movie name")

with col2:
    st.write("")
    st.write("")
    search = st.button("Recommend")

# ---------------- RESULTS ----------------
if search and movie_input:
    results = recommend(movie_input)

    if results is None:
        st.error("Movie not found")
    else:
        st.success(f"Recommendations for {movie_input}")

        cols = st.columns(3)

        for i, movie in enumerate(results):
            data = fetch_movie_data(movie["name"])

            with cols[i % 3]:

                if data["poster"] and data["poster"] != "N/A":
                    st.image(data["poster"])

                st.markdown(f"""
                <div class="movie-card">
                    <div class="title">{movie['name']}</div>
                    <div class="meta">⭐ {movie['rating']} | 📅 {movie['year']}</div>
                    <div class="meta">🎭 {data['genre']}</div>
                    <div class="meta">{data['plot'][:120]}...</div>
                </div>
                """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("✨ Built with Streamlit | Free Movie API")
