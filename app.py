import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Netflix AI", layout="wide")

# ---------------- API ----------------
def fetch_movie(name):
    url = f"http://www.omdbapi.com/?t={name}&apikey=thewdb"
    data = requests.get(url).json()
    return data if data["Response"] == "True" else None

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Remove padding */
.block-container {
    padding: 1rem 2rem;
}

/* HERO */
.hero {
    position: relative;
    height: 400px;
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 30px;
}

.hero img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(40%);
}

.hero-content {
    position: absolute;
    bottom: 30px;
    left: 30px;
    color: white;
}

.hero h1 {
    font-size: 40px;
}

/* ROW TITLE */
.row-title {
    font-size: 22px;
    margin: 20px 0 10px 10px;
    color: white;
}

/* HORIZONTAL SCROLL */
.scroll-container {
    display: flex;
    overflow-x: auto;
    gap: 15px;
    padding: 10px;
}

.scroll-container::-webkit-scrollbar {
    display: none;
}

/* MOVIE CARD */
.movie {
    min-width: 180px;
    transition: 0.3s;
}

.movie img {
    border-radius: 10px;
}

.movie:hover {
    transform: scale(1.15);
}

/* SEARCH */
.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 10px;
}

.stButton button {
    background: red;
    color: white;
    border-radius: 8px;
}

/* BACKGROUND */
body {
    background-color: #0b0f1a;
    color: white;
}

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

# ---------------- HERO ----------------
featured = df.sample(1)["Film Name"].values[0]
hero_data = fetch_movie(featured)

if hero_data and hero_data["Poster"] != "N/A":
    st.markdown(f"""
    <div class="hero">
        <img src="{hero_data['Poster']}">
        <div class="hero-content">
            <h1>{featured}</h1>
            <p>{hero_data['Plot'][:150]}...</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SEARCH ----------------
col1, col2 = st.columns([4,1])

with col1:
    movie_input = st.text_input("🔍 Search movie")

with col2:
    st.write("")
    search = st.button("Recommend")

# ---------------- RECOMMENDED ROW ----------------
if search and movie_input:
    recs = recommend(movie_input)

    if recs:
        st.markdown('<div class="row-title">Recommended</div>', unsafe_allow_html=True)

        html = '<div class="scroll-container">'

        for name in recs:
            data = fetch_movie(name)
            if data and data["Poster"] != "N/A":
                html += f"""
                <div class="movie">
                    <img src="{data['Poster']}" width="180">
                </div>
                """

        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

# ---------------- POPULAR ROW ----------------
st.markdown('<div class="row-title">🔥 Popular</div>', unsafe_allow_html=True)

html = '<div class="scroll-container">'

for name in df.sample(10)["Film Name"]:
    data = fetch_movie(name)
    if data and data["Poster"] != "N/A":
        html += f"""
        <div class="movie">
            <img src="{data['Poster']}" width="180">
        </div>
        """

html += '</div>'
st.markdown(html, unsafe_allow_html=True)
