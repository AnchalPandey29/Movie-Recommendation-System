import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎬 Movie AI", layout="wide")

WATCHLIST_FILE = "watchlist.json"
PLACEHOLDER = "https://dummyimage.com/300x450/cccccc/000000&text=No+Poster"

# ---------------- STATE ----------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

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
    try:
        data = requests.get(url).json()
        return data if data.get("Response") == "True" else None
    except:
        return None

def get_trailer_embed(movie):
    query = movie.replace(" ", "+") + "+trailer"
    return f"https://www.youtube.com/embed?listType=search&list={query}"

# ---------------- STYLE ----------------
st.markdown("""
<style>

/* HERO */
.hero {
    padding: 40px;
    border-radius: 20px;
    background: linear-gradient(135deg,#1e293b,#020617);
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}

/* SEARCH */
.stTextInput {
    display: flex;
    justify-content: center;
}
.stTextInput input {
    width: 60%;
    border-radius: 12px;
    padding: 10px;
}

/* CARD */
.card {
    background: rgba(17,24,39,0.85);
    backdrop-filter: blur(10px);
    padding: 12px;
    border-radius: 15px;
    color: white;
    text-align: center;
    transition: 0.3s;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
}
.card:hover {
    transform: scale(1.05);
}

/* DETAILS PANEL */
.details-box {
    background: #f9fafb;
    padding: 25px;
    border-radius: 20px;
    margin-top: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* BUTTON */
.stButton button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 10px;
    border: none;
}

/* ANALYTICS CARD */
.analytics-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
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

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Recommendation System</h1>
    <p>Discover movies intelligently with AI</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SEARCH ----------------
search = st.text_input("🔍 Search your movie")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "❤️ Watchlist", "📊 Analytics"])

# ================= TAB 1 =================
with tab1:

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

                    st.markdown(f"<div class='card'><b>{movie}</b></div>", unsafe_allow_html=True)

                    if st.button("🎬 View", key=f"view_{i}"):
                        st.session_state.selected_movie = movie

    # DETAILS PANEL
    if st.session_state.selected_movie:
        movie = st.session_state.selected_movie
        data = fetch_movie(movie)

        if data:
            st.markdown("<div class='details-box'>", unsafe_allow_html=True)

            col1, col2 = st.columns([1,2])

            poster = PLACEHOLDER
            if data.get("Poster") != "N/A":
                poster = data["Poster"]

            with col1:
                st.image(poster)

            with col2:
                st.title(movie)
                st.markdown(f"⭐ **{data.get('imdbRating','N/A')}**")
                st.markdown(f"🎭 {data.get('Genre','')}")
                st.markdown(f"📅 {data.get('Year','')}")
                st.write(data.get("Plot",""))

                st.markdown("### 🎥 Trailer")
                st.components.v1.iframe(get_trailer_embed(movie), height=300)

                if movie not in watchlist:
                    if st.button("❤️ Add to Watchlist"):
                        watchlist.append(movie)
                        save_watchlist(watchlist)
                        st.success("Added")
                else:
                    if st.button("❌ Remove"):
                        watchlist.remove(movie)
                        save_watchlist(watchlist)
                        st.warning("Removed")

            st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("❤️ Your Watchlist")

    if watchlist:
        cols = st.columns(5)

        for i, movie in enumerate(watchlist):
            data = fetch_movie(movie)

            poster = PLACEHOLDER
            if data and data.get("Poster") != "N/A":
                poster = data["Poster"]

            with cols[i % 5]:
                st.image(poster)
                st.markdown(f"<div class='card'>{movie}</div>", unsafe_allow_html=True)
    else:
        st.info("No movies yet")

# ================= TAB 3 =================
with tab3:
    st.subheader("📊 Analytics Dashboard")

    df_plot = pd.read_csv("movies.csv")
    df_plot["Ratings"] = pd.to_numeric(df_plot["Ratings"], errors="coerce")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.markdown("### Ratings Distribution")
        st.bar_chart(df_plot["Ratings"].value_counts().sort_index())
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='analytics-card'>", unsafe_allow_html=True)
        st.markdown("### Top Rated Movies")
        top = df_plot.sort_values(by="Ratings", ascending=False).head(10)
        st.dataframe(top[["Film Name", "Ratings"]])
        st.markdown("</div>", unsafe_allow_html=True)
