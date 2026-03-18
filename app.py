import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎬 Movie AI", layout="wide")

WATCHLIST_FILE = "watchlist.json"
PLACEHOLDER = "https://via.placeholder.com/300x450?text=No+Poster"

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

def get_trailer(movie):
    query = movie.replace(" ", "+") + "+trailer"
    return f"https://www.youtube.com/results?search_query={query}"

# ---------------- STYLE ----------------
st.markdown("""
<style>
.block-container { padding: 1.5rem 3rem; }

.card {
    background: #111827;
    padding: 15px;
    border-radius: 15px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    text-align: center;
    transition: 0.3s;
}
.card:hover { transform: translateY(-5px); }

.title { font-size: 16px; font-weight: 600; }

.stButton button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 8px;
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

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "❤️ Watchlist", "📊 Analytics"])

# ================= TAB 1 =================
with tab1:
    st.title("🎬 Movie Recommender")

    search = st.text_input("🔍 Search a movie")

    if search:
        recs = recommend(search)

        if recs:
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

                    with st.expander("Details"):
                        if data:
                            st.write(f"⭐ {data.get('imdbRating','N/A')}")
                            st.write(f"🎭 {data.get('Genre','')}")
                            st.write(f"📅 {data.get('Year','')}")
                            st.write(data.get("Plot",""))

                            st.markdown(f"[▶ Watch Trailer]({get_trailer(movie)})")

                    # ❤️ Watchlist
                    if movie not in watchlist:
                        if st.button("❤️ Add", key=f"add_{i}"):
                            watchlist.append(movie)
                            save_watchlist(watchlist)
                            st.success("Added")
                    else:
                        if st.button("❌ Remove", key=f"remove_{i}"):
                            watchlist.remove(movie)
                            save_watchlist(watchlist)
                            st.warning("Removed")

# ================= TAB 2 =================
with tab2:
    st.title("❤️ Your Watchlist")

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
        st.info("No movies added yet")

# ================= TAB 3 =================
with tab3:
    st.title("📊 Movie Insights")

    # Ratings distribution
    df_plot = pd.read_csv("movies.csv")
    df_plot["Ratings"] = pd.to_numeric(df_plot["Ratings"], errors="coerce")

    st.subheader("Ratings Distribution")
    st.bar_chart(df_plot["Ratings"].value_counts().sort_index())

    # Top movies
    st.subheader("Top Rated Movies")
    top = df_plot.sort_values(by="Ratings", ascending=False).head(10)
    st.dataframe(top[["Film Name", "Ratings"]])
