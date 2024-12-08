import os
import pickle
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Fetch movie poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path', None)
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return "https://via.placeholder.com/500x750.png?text=No+Image"

# Recommend movies dynamically
def recommend(movie):
    # Find the index of the selected movie
    index = movies[movies['title'] == movie].index[0]
    
    # Calculate cosine similarity dynamically
    movie_features = movies['combined_features']
    count_matrix = CountVectorizer(stop_words='english').fit_transform(movie_features)
    cosine_sim = cosine_similarity(count_matrix)
    
    # Get the top 5 similar movies
    distances = sorted(list(enumerate(cosine_sim[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movie_names, recommended_movie_posters

# Streamlit app
st.header('Movie Recommender System')

# Specify the path to the movie dictionary file
file_path = 'model/movie_dict.pkl'  # Adjust the path if your file is in a different location

# Check if the file exists
if not os.path.exists(file_path):
    st.error("File not found. Please ensure 'movie_dict.pkl' is in the 'model/' folder.")
    st.stop()

# Load movie list
movies_dict = pickle.load(open(file_path, 'rb'))
movies = pd.DataFrame(movies_dict)

# Combine relevant features into a single string for similarity computation
movies['combined_features'] = movies[['genres', 'keywords', 'tagline', 'cast', 'crew']].fillna('').agg(' '.join, axis=1)

# Create dropdown for movie selection
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

# Show recommendations on button click
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])



      
