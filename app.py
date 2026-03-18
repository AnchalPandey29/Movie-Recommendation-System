import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Movie AI", layout="wide")

WATCHLIST_FILE = "watchlist.json"

# ---------------- WATCHLIST ----------------
def load_watchlist():
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_watchlist(data):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(data, f)

watchlist = load_watchlist()

# ---------------- API ----------------
def fetch_movie(name):
    url = f"http://www.omdbapi.com/?t={name}&apikey=thewdb"
    data = requests.get(url).json()
    return data if data.get("Response") == "True" else None

def get_youtube_embed(movie):
    query = movie.replace(" ", "+") + "+trailer"
    return f"https://www.youtube.com/embed?listType=search&list={query}"

# ---------------- STYLE ----------------
st.markdown("""
<style>
.block-container { padding: 1.5rem 3rem; }
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
    margin-bottom: 30px;
}
.card {
    background: #111827;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
}
.title { font-size: 18px; font-weight: 600; }
.meta { color: #9ca3af; font-size: 14px; }
.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 10px;
}
.stButton button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    # 🔥 SAME CLEANING AS NOTEBOOK
    df["Film Name"] = df["Film Name"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
    df["Film Name"] = df["Film Name"].str.lower().str.strip()

    df["Summary"] = df["Summary"].fillna("").str.lower().str.strip()

    df = df.dropna(subset=["Summary"]).reset_index(drop=True)

    return df

df = load_data()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model(data):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(data["Summary"])

    knn = NearestNeighbors(n_neighbors=6, metric="cosine")
    knn.fit(matrix)

    return matrix, knn

matrix, knn = load_model(df)

# ---------------- RECOMMEND ----------------
def clean_input(text):
    return "".join([c for c in text.lower() if c.isalpha() or c.isspace()]).strip()

def recommend(movie):
    movie = clean_input(movie)

    matches = df[df["Film Name"].str.contains(movie)]

    if matches.empty:
        return None

    idx = matches.index[0]

    _, indices = knn.kneighbors(matrix[idx])

    results = []
    for i in indices[0][1:]:
        results.append(df.iloc[i]["Film Name"].title())

    return results

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Recommendation System</h1>
    <p>AI-powered movie suggestions with trailers & watchlist</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SEARCH ----------------
search_input = st.text_input("🔍 Search your favorite movie")

search = ""
if search_input:
    search = clean_input(search_input)

# 🔍 Suggestions
if search:
    suggestions = df[df["Film Name"].str.contains(search)]["Film Name"].head(5)

    for s in suggestions:
        if st.button(s.title()):
            search = s

# ---------------- RESULTS ----------------
if search:
    recs = recommend(search)

    if recs is None:
        st.error("Movie not found in dataset")
    else:
        st.markdown("## 🎯 Recommendations")

        for i, movie in enumerate(recs):
            data = fetch_movie(movie)

            if data:
                col1, col2 = st.columns([1,2])

                with col1:
                    if data.get("Poster") != "N/A":
                        st.image(data["Poster"])

                with col2:
                    st.markdown(f"""
                    <div class="card">
                        <div class="title">{movie}</div>
                        <div class="meta">⭐ {data.get("imdbRating","N/A")} | 🎭 {data.get("Genre","")}</div>
                        <div class="meta">📅 {data.get("Year","")} | ⏱ {data.get("Runtime","")}</div>
                        <p>{data.get("Plot","")}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 🎬 Trailer
                    st.markdown("#### 🎥 Trailer")
                    st.components.v1.iframe(get_youtube_embed(movie), height=300)

                    # ❤️ Watchlist (unique keys fix)
                    if movie not in watchlist:
                        if st.button(f"❤️ Add - {movie}", key=f"add_{i}"):
                            watchlist.append(movie)
                            save_watchlist(watchlist)
                            st.success("Added!")
                    else:
                        if st.button(f"❌ Remove - {movie}", key=f"remove_{i}"):
                            watchlist.remove(movie)
                            save_watchlist(watchlist)
                            st.warning("Removed")

# ---------------- WATCHLIST ----------------
st.markdown("## ❤️ Your Watchlist")

if watchlist:
    cols = st.columns(4)

    for i, movie in enumerate(watchlist):
        data = fetch_movie(movie)

        with cols[i % 4]:
            if data and data.get("Poster") != "N/A":
                st.image(data["Poster"])

            st.markdown(f"""
            <div class="card">
                <div class="title">{movie}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No movies in watchlist yet")
