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
    return data if data["Response"] == "True" else None

def get_youtube_embed(movie):
    query = movie.replace(" ", "+") + "+trailer"
    return f"https://www.youtube.com/embed?listType=search&list={query}"

# ---------------- STYLE ----------------
st.markdown("""
<style>

.block-container {
    padding: 1.5rem 3rem;
}

/* HERO */
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
    margin-bottom: 30px;
}

/* CARD */
.card {
    background: #111827;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-5px);
}

/* TITLE */
.title {
    font-size: 18px;
    font-weight: 600;
}

/* META */
.meta {
    color: #9ca3af;
    font-size: 14px;
}

/* SEARCH */
.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 10px;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 10px;
}

/* SECTION */
.section {
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df["Summary"] = df["Summary"].fillna("").str.lower()
    df["Film Name"] = df["Film Name"].str.strip()
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
def recommend(movie):
    movie = movie.lower().strip()

    if movie not in df["Film Name"].str.lower().values:
        return None

    idx = df[df["Film Name"].str.lower() == movie].index[0]
    _, indices = knn.kneighbors(matrix[idx])

    return [df.iloc[i]["Film Name"] for i in indices[0][1:]]

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Recommendation System</h1>
    <p>AI-powered movie suggestions with trailers, details & watchlist</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SEARCH ----------------
search = st.text_input("🔍 Search your favorite movie")

# 🔍 Suggestions
if search:
    suggestions = df[df["Film Name"].str.lower().str.contains(search.lower())]["Film Name"].head(5)

    for s in suggestions:
        if st.button(s):
            search = s

# ---------------- RESULTS ----------------
if search:
    recs = recommend(search)

    if recs is None:
        st.error("Movie not found in dataset")
    else:
        st.markdown("## 🎯 Recommendations")

        for movie in recs:
            data = fetch_movie(movie)

            if data:
                col1, col2 = st.columns([1,2])

                with col1:
                    if data["Poster"] != "N/A":
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

                    # ❤️ Watchlist
                    if movie not in watchlist:
                        if st.button(f"❤️ Add to Watchlist - {movie}"):
                            watchlist.append(movie)
                            save_watchlist(watchlist)
                            st.success("Added to Watchlist")
                    else:
                        if st.button(f"❌ Remove - {movie}"):
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
            if data and data["Poster"] != "N/A":
                st.image(data["Poster"])

            st.markdown(f"""
            <div class="card">
                <div class="title">{movie}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No movies in watchlist yet")
