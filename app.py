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

def get_trailer(movie):
    query = movie.replace(" ", "+") + "+trailer"
    return f"https://www.youtube.com/results?search_query={query}"

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Scroll row */
.scroll {
    display:flex;
    overflow-x:auto;
    gap:15px;
    padding:10px;
}

.movie {
    min-width:200px;
    transition:0.3s;
}
.movie:hover { transform:scale(1.1); }

</style>
""", unsafe_allow_html=True)

# ---------------- DATA ----------------
@st.cache_data
def load():
    df = pd.read_csv("movies.csv")
    df["Summary"] = df["Summary"].fillna("").str.lower()
    return df

df = load()

# ---------------- MODEL ----------------
@st.cache_resource
def model(data):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(data["Summary"])

    knn = NearestNeighbors(n_neighbors=6, metric="cosine")
    knn.fit(matrix)
    return matrix, knn

matrix, knn = model(df)

# ---------------- RECOMMEND ----------------
def recommend(movie):
    movie = movie.lower()

    if movie not in df["Film Name"].str.lower().values:
        return None

    idx = df[df["Film Name"].str.lower() == movie].index[0]
    _, indices = knn.kneighbors(matrix[idx])

    return [df.iloc[i]["Film Name"] for i in indices[0][1:]]

# ---------------- SEARCH SUGGEST ----------------
query = st.text_input("🔍 Search movie")

suggestions = []
if query:
    suggestions = df[df["Film Name"].str.lower().str.contains(query.lower())]["Film Name"].head(5)

for s in suggestions:
    if st.button(s):
        query = s

# ---------------- RECOMMEND ----------------
if query:
    recs = recommend(query)

    if recs:
        st.subheader("🎯 Recommendations")

        for movie in recs:
            data = fetch_movie(movie)

            if data:
                col1, col2 = st.columns([1,2])

                with col1:
                    if data["Poster"] != "N/A":
                        st.image(data["Poster"])

                with col2:
                    st.markdown(f"### {movie}")
                    st.write(data["Plot"])

                    # 🎬 Trailer
                    st.markdown(f"[▶ Watch Trailer]({get_trailer(movie)})")

                    # ❤️ Watchlist
                    if movie not in watchlist:
                        if st.button(f"❤️ Add to Watchlist - {movie}"):
                            watchlist.append(movie)
                            save_watchlist(watchlist)
                            st.success("Added!")
                    else:
                        if st.button(f"❌ Remove - {movie}"):
                            watchlist.remove(movie)
                            save_watchlist(watchlist)
                            st.warning("Removed")

# ---------------- WATCHLIST VIEW ----------------
st.markdown("---")
st.subheader("❤️ Your Watchlist")

if watchlist:
    for movie in watchlist:
        st.write("🎬", movie)
else:
    st.write("No movies yet")
