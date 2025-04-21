import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from hybrid_recommendation import (
    load_hybrid_model,
    retrieve_relevant_movies,
    generate_recommendations_with_rag,
    compute_query_embedding,
    extract_preferences_from_llm
)

# ------------------- MODEL LOADING -------------------

@st.cache_resource
def load_model():
    """Load the sentence transformer model and movie embeddings"""
    try:
        with open('embeddings/movie_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
            # Handle both possible formats of the pickle file
            if isinstance(data, tuple) and len(data) == 2:
                movie_titles, movie_embeddings = data
            else:
                st.error("Unexpected format in movie_embeddings.pkl")
                return None, None, None
                
        # Load the sentence transformer model
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Ensure embeddings are in numpy array format
        if not isinstance(movie_embeddings, np.ndarray):
            movie_embeddings = np.array(movie_embeddings)
            
        return embedder, movie_embeddings, movie_titles
    except FileNotFoundError:
        st.error("Movie embeddings file not found. Please make sure 'embeddings/movie_embeddings.pkl' exists.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def load_movie_data(movies_path="movies.csv", tags_path="tags.csv"):
    """Load and process movie and tag data from CSV files"""
    try:
        # Load movies.csv
        movies = pd.read_csv(movies_path)
        
        # Process movies data
        movies.rename(columns={'title': 'title_year'}, inplace=True)
        movies[['title', 'year']] = movies['title_year'].str.extract(r'^(.*)s\((\d{4})\)$')
        movies.drop(columns=['title_year'], inplace=True)
        movies['year'] = movies['year'].fillna(-1).astype(int)
        movies['genres'] = movies['genres'].str.split('|').apply(lambda x: ' '.join([g.lower().replace(' ', '_') for g in x]))

        # Load tags.csv
        tags = pd.read_csv(tags_path)
        
        # Process tags data
        tags['tag'] = tags['tag'].fillna('').astype(str).str.lower().str.replace(' ', '_')
        tag_grouped = tags.groupby('movieId')['tag'].agg(list).reset_index()

        # Merge movies and tags
        movies = movies.merge(tag_grouped, on='movieId', how='left')
        movies['tag'] = movies['tag'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        movies['soup'] = movies['tag'] + ' ' + movies['genres']

        # Remove rows with empty soup
        movies['soup'] = movies['soup'].str.strip()
        movies = movies[movies['soup'].str.len() > 0]

        return movies
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# ------------------- RECOMMENDATION FUNCTIONS -------------------

def get_movie_details(title, movie_df):
    """Get detailed information about a movie from the dataframe"""
    if movie_df.empty:
        return None
        
    # Find the movie in the dataframe (case-insensitive)
    movie = movie_df[movie_df["title"].str.lower() == title.lower()]
    
    # If exact match not found, try partial match
    if movie.empty:
        movie = movie_df[movie_df["title"].str.lower().str.contains(title.lower())]
        
    # Return the first match (if any)
    if not movie.empty:
        return movie.iloc[0].to_dict()
    return None

def get_recommendations(user_query, top_k=10):
    """Get movie recommendations based on user query"""
    # Load models and data
    movie_data, tfidf_vectorizer, tfidf_matrix, faiss_index, word2vec_model = load_hybrid_model()
    
    # Extract preferences from user query
    preferences = {"genres": user_query, "themes": user_query}
    
    # Retrieve relevant movies
    retrieved_movies = retrieve_relevant_movies(
        preferences, 
        movie_data, 
        faiss_index, 
        tfidf_vectorizer, 
        top_k=top_k
    )
    
    # Generate recommendations with RAG
    recommendations = generate_recommendations_with_rag(user_query, retrieved_movies)
    
    # Return both the retrieved movies and the generated recommendations
    return retrieved_movies, recommendations

def filter_recommendations(movies, min_year=None, max_year=None, genres=None, sort_by='year'):
    """Filter movie recommendations based on criteria"""
    filtered_movies = movies.copy()

    # Filter by year
    if min_year:
        filtered_movies = filtered_movies[filtered_movies['year'] >= min_year]
    if max_year:
        filtered_movies = filtered_movies[filtered_movies['year'] <= max_year]

    # Filter by genres
    if genres:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(
            lambda x: any(genre.lower() in x for genre in genres)
        )]

    # Sort the DataFrame
    if sort_by in filtered_movies.columns:
        filtered_movies.sort_values(sort_by, ascending=False, inplace=True)

    return filtered_movies

def process_user_preferences(preferences_text):
    """Process user preferences text into a structured format"""
    # This is a simple implementation - could be enhanced with NLP
    preferences = {}
    
    # Extract genres
    genres = ["action", "adventure", "animation", "comedy", "crime", "documentary", 
              "drama", "family", "fantasy", "history", "horror", "music", "mystery",
              "romance", "science fiction", "thriller", "war", "western"]
    
    preferences["genres"] = [genre for genre in genres if genre in preferences_text.lower()]
    
    # Extract mood
    moods = ["happy", "sad", "excited", "relaxed", "thoughtful", "tense"]
    for mood in moods:
        if mood in preferences_text.lower():
            preferences["mood"] = mood
            break
    
    # Extract era preferences
    eras = ["classic", "retro", "modern", "contemporary"]
    preferences["eras"] = [era for era in eras if era in preferences_text.lower()]
    
    return preferences