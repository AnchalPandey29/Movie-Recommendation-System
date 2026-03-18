import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Movie AI", layout="wide")

WATCHLIST_FILE = "watchlist.json"
PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Image"

# ---------------- SESSION PAGE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = ""

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

# ✅ FIXED TRAILER (real embed)
def get_trailer(movie):
    query = movie.replace(" ", "+") + "+trailer"
    return f"https://www.youtube.com/results?search_query={query}"

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
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    text-align: center;
}

.title { font-size: 18px; font-weight: 600; }

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

    df["Film Name"] = df["Film Name"].str.replace(r"[^a-zA-Z\s]", "", regex=True)
    df["Film Name"] = df["Film Name"].str.lower().str.strip()

    df["Summary"] = df["Summary"].fillna("").str.lower().str.strip()

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
    return "".join([c for c in text.lower() if c.isalpha() or c.isspace()])

def recommend(movie):
    movie = clean_input(movie)

    matches = df[df["Film Name"].str.contains(movie)]

    if matches.empty:
        return None

    idx = matches.index[0]

    _, indices = knn.kneighbors(matrix[idx])

    return [df.iloc[i]["Film Name"].title() for i in indices[0][1:]]

# ---------------- HOME PAGE ----------------
if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
        <h1>🎬 Movie Recommendation System</h1>
        <p>Discover movies with AI</p>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input("🔍 Search movie")

    if search:
        recs = recommend(search)

        if recs:
            st.markdown("## 🎯 Recommendations")

            cols = st.columns(5)

            for i, movie in enumerate(recs):
                data = fetch_movie(movie)

                poster = PLACEHOLDER
                if data and data.get("Poster") != "N/A":
                    poster = data["Poster"]

                with cols[i % 5]:
                    st.image(poster)

                    st.markdown(f"""
                    <div class="card">
                        <div class="title">{movie}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 🔥 OPEN DETAILS PAGE
                    if st.button("View", key=f"view_{i}"):
                        st.session_state.selected_movie = movie
                        st.session_state.page = "details"

# ---------------- DETAILS PAGE ----------------
if st.session_state.page == "details":

    movie = st.session_state.selected_movie
    data = fetch_movie(movie)

    st.button("⬅ Back", on_click=lambda: st.session_state.update({"page": "home"}))

    if data:
        col1, col2 = st.columns([1,2])

        poster = PLACEHOLDER
        if data.get("Poster") != "N/A":
            poster = data["Poster"]

        with col1:
            st.image(poster)

        with col2:
            st.title(movie)
            st.write(f"⭐ {data.get('imdbRating','N/A')}")
            st.write(f"🎭 {data.get('Genre','')}")
            st.write(f"📅 {data.get('Year','')}")
            st.write(f"⏱ {data.get('Runtime','')}")
            st.write(data.get("Plot",""))

            # 🎬 Trailer (FIXED)
            st.markdown("### 🎥 Trailer")
            st.markdown(f"[▶ Watch Trailer]({get_trailer(movie)})")

            # ❤️ Watchlist
            if movie not in watchlist:
                if st.button("❤️ Add to Watchlist"):
                    watchlist.append(movie)
                    save_watchlist(watchlist)
                    st.success("Added!")
            else:
                if st.button("❌ Remove from Watchlist"):
                    watchlist.remove(movie)
                    save_watchlist(watchlist)
                    st.warning("Removed")

# ---------------- WATCHLIST ----------------
st.markdown("## ❤️ Watchlist")

if watchlist:
    cols = st.columns(5)

    for i, movie in enumerate(watchlist):
        data = fetch_movie(movie)

        poster = PLACEHOLDER
        if data and data.get("Poster") != "N/A":
            poster = data["Poster"]

        with cols[i % 5]:
            st.image(poster)
            st.write(movie)
else:
    st.info("No movies yet")
