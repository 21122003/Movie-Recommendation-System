import streamlit as st
import sqlite3
import os
import pickle
import hashlib
import secrets
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
import faiss  # Ensure FAISS is installed
from transformers import pipeline, GPT2Tokenizer

# Import required functions from hybrid_recommendation.py
from hybrid_recommendation import (
    load_hybrid_model,
    retrieve_relevant_movies,
    generate_recommendations_with_rag,
    compute_query_embedding,
    create_faiss_index,
    extract_preferences_from_llm  # Ensure this function is imported
)

# ------------------- DATABASE SETUP -------------------

def create_db():
    """Create database and necessary tables if they don't exist"""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect('data/users.db')
    cursor = conn.cursor()

    # Users table with password hash and salt
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # User history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            preferences TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # User ratings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            movie_title TEXT NOT NULL,
            rating INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, movie_title)
        )
    ''')

    conn.commit()
    conn.close()

def get_db_connection():
    """Create and return a database connection"""
    return sqlite3.connect('data/users.db')

def migrate_database():
    """Migrate database from old schema to new schema"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if the old schema exists
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'password' in column_names and 'password_hash' not in column_names:
            # Backup existing users
            cursor.execute("SELECT username, email, password FROM users")
            old_users = cursor.fetchall()
            
            # Create temporary table
            cursor.execute("ALTER TABLE users RENAME TO users_old")
            
            # Create new users table
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Migrate users
            for username, email, password in old_users:
                password_hash, salt = hash_password(password)
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                    (username, email, password_hash, salt)
                )
            
            # Drop old table
            cursor.execute("DROP TABLE users_old")
            
            conn.commit()
            st.success("Database migrated successfully!")
        
        conn.close()
    except Exception as e:
        st.error(f"Migration error: {str(e)}")

# ------------------- SECURITY FUNCTIONS -------------------

def hash_password(password, salt=None):
    """Hash password with a salt using SHA-256"""
    if salt is None:
        salt = secrets.token_hex(16)  # Generate a random salt
    
    # Combine password and salt, then hash
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    
    return password_hash, salt

def validate_password(password):
    """Ensure password meets minimum security requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
        
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
        
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"

def validate_email(email):
    """Validate email format"""
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if re.match(email_pattern, email):
        return True, "Email is valid"
    return False, "Invalid email format"

# ------------------- USER MANAGEMENT -------------------

def get_user_from_db(username, password):
    """Authenticate user with username and password (compatible with both schemas)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Try the new schema first
    try:
        cursor.execute("SELECT id, username, email, password_hash, salt FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        
        if user:
            stored_hash = user[3]
            salt = user[4]
            calculated_hash, _ = hash_password(password, salt)
            
            if calculated_hash == stored_hash:
                return user
    except sqlite3.OperationalError:
        # If new schema fails, try old schema
        try:
            cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            user = cursor.fetchone()
            if user:
                return user
        except Exception:
            pass
    
    conn.close()
    return None

def insert_user_to_db(username, email, password):
    """Register a new user with secure password storage"""
    # Check if we should use the old schema
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'password_hash' in column_names:
            # Use new schema
            password_hash, salt = hash_password(password)
            
            try:
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)", 
                    (username, email, password_hash, salt)
                )
                conn.commit()
                return True, "Registration successful"
            except sqlite3.IntegrityError:
                return False, "Username already exists"
        else:
            # Use old schema
            try:
                cursor.execute(
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                    (username, email, password)
                )
                conn.commit()
                return True, "Registration successful"
            except sqlite3.IntegrityError:
                return False, "Username already exists"
    except Exception as e:
        return False, f"Registration error: {str(e)}"
    finally:
        conn.close()

def save_user_preferences(username, preferences):
    """Save user preferences to history table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO user_history (username, preferences) VALUES (?, ?)",
            (username, preferences)
        )
        conn.commit()
        return True, "Preferences saved successfully"
    except Exception as e:
        return False, f"Error saving preferences: {str(e)}"
    finally:
        conn.close()

def get_user_history(username, limit=5):
    """Get user's preference history, most recent first"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT preferences, timestamp FROM user_history WHERE username = ? ORDER BY timestamp DESC LIMIT ?", 
        (username, limit)
    )
    history = cursor.fetchall()
    conn.close()
    
    return history

def save_movie_rating(username, movie_title, rating):
    """Save or update a user's movie rating"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if the rating already exists
        cursor.execute(
            "SELECT id FROM user_ratings WHERE username = ? AND movie_title = ?",
            (username, movie_title)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing rating
            cursor.execute(
                "UPDATE user_ratings SET rating = ?, timestamp = CURRENT_TIMESTAMP WHERE username = ? AND movie_title = ?",
                (rating, username, movie_title)
            )
        else:
            # Insert new rating
            cursor.execute(
                "INSERT INTO user_ratings (username, movie_title, rating) VALUES (?, ?, ?)",
                (username, movie_title, rating)
            )
            
        conn.commit()
        return True, "Rating saved"
    except Exception as e:
        return False, f"Error saving rating: {str(e)}"
    finally:
        conn.close()

def get_user_ratings(username):
    """Get all ratings by a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT movie_title, rating, timestamp FROM user_ratings WHERE username = ? ORDER BY timestamp DESC",
        (username,)
    )
    ratings = cursor.fetchall()
    conn.close()
    
    return ratings

# ------------------- MODEL AND DATA LOADING -------------------

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
def load_movie_data(
    movies_path="movies.csv",
    tags_path="tags.csv"
):
    """Load and process movie and tag data from CSV files"""
    try:
        # Load movies.csv
        movies = pd.read_csv(movies_path)
        st.info(f"Loaded movies data with {len(movies)} rows.")

        # Process movies data
        movies.rename(columns={'title': 'title_year'}, inplace=True)
        movies[['title', 'year']] = movies['title_year'].str.extract(r'^(.*)\s\((\d{4})\)$')
        movies.drop(columns=['title_year'], inplace=True)
        movies['year'] = movies['year'].fillna(-1).astype(int)
        movies['genres'] = movies['genres'].str.split('|').apply(lambda x: ' '.join([g.lower().replace(' ', '_') for g in x]))

        # Load tags.csv
        tags = pd.read_csv(tags_path)
        st.info(f"Loaded tags data with {len(tags)} rows.")

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

        st.info(f"Processed movie data with {len(movies)} rows remaining.")
        return movies
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_movie_details(title, movie_df):
    """Get detailed information about a movie from the dataframe"""
    if movie_df.empty:
        return None
        
    # Find the movie in the dataframe (case-insensitive)
    movie = movie_df[movie_df["Title"].str.lower() == title.lower()]
    
    # If exact match not found, try partial match
    if movie.empty:
        movie = movie_df[movie_df["Title"].str.lower().str.contains(title.lower())]
        
    # Return the first match (if any)
    if not movie.empty:
        return movie.iloc[0].to_dict()
    return None

# ------------------- UI COMPONENTS -------------------

def movie_card(movie, similarity_score, username, movie_df):
    """Create a UI card for a movie recommendation with details from movie_df"""
    # Get movie details if available
    movie_details = get_movie_details(movie, movie_df)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display poster if available
        if movie_details and "Poster" in movie_details and movie_details["Poster"] != "N/A":
            st.image(movie_details["Poster"], width=200)
        else:
            # Display placeholder
            st.markdown("🎬")
    
    with col2:
        # Title and year
        if movie_details and "Year" in movie_details:
            st.header(f"{movie} ({movie_details['Year']})")
        else:
            st.header(movie)
            
        # Match score
        st.write(f"**Match score:** {similarity_score:.2f}")
        
        # Display movie details
        if movie_details:
            if "Genre" in movie_details:
                st.write(f"**Genre:** {movie_details['Genre']}")
                
            if "Director" in movie_details:
                st.write(f"**Director:** {movie_details['Director']}")
                
            if "Actors" in movie_details:
                st.write(f"**Starring:** {movie_details['Actors']}")
                
            if "Plot" in movie_details and movie_details["Plot"] != "N/A":
                st.write(f"**Plot:** {movie_details['Plot']}")
                
            if "imdbRating" in movie_details and movie_details["imdbRating"] != "N/A":
                st.write(f"**IMDb Rating:** ⭐ {movie_details['imdbRating']}/10")
        
        # Rating widget
        rating = st.select_slider(
            f"Rate '{movie}'",
            options=[1, 2, 3, 4, 5],
            value=3,
            key=f"rating_{movie}"
        )
        
        if st.button("Save Rating", key=f"save_{movie}"):
            success, message = save_movie_rating(username, movie, rating)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    st.divider()

def display_ratings_chart(ratings):
    """Display a chart of user ratings"""
    if not ratings:
        st.info("No ratings yet. Rate some movies to see your chart!")
        return
        
    # Create a DataFrame from ratings
    df = pd.DataFrame(ratings, columns=['Movie', 'Rating', 'Timestamp'])
    
    # Create a bar chart of ratings
    fig = px.bar(
        df, 
        x='Movie', 
        y='Rating',
        title='Your Movie Ratings',
        color='Rating',
        color_continuous_scale='RdYlGn',  # Red to Green scale
        range_color=[1, 5]
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

# ------------------- PAGES -------------------

def login_page():
    """Display the login page"""
    st.title("🎬 Movie Recommender - Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return
                
            user = get_user_from_db(username, password)
            if user:
                st.session_state['user'] = user
                st.session_state['username'] = username  # Use the username directly
                st.success("Login successful! Redirecting...")
                st.session_state['page'] = "recommendations"
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.divider()
    
    st.write("Don't have an account?")
    if st.button("Register Here"):
        st.session_state['page'] = "registration"
        st.rerun()

def registration_page():
    """Display the registration page"""
    st.title("🎬 Movie Recommender - Registration")
    
    with st.form("registration_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')
        submit = st.form_submit_button("Register")
        
        if submit:
            # Validate inputs
            if not username or not email or not password:
                st.error("All fields are required")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match")
                return
                
            # Validate email format
            valid_email, email_msg = validate_email(email)
            if not valid_email:
                st.error(email_msg)
                return
                
            # Validate password strength
            valid_password, password_msg = validate_password(password)
            if not valid_password:
                st.error(password_msg)
                return
                
            # Register the user
            success, message = insert_user_to_db(username, email, password)
            if success:
                st.success(f"{message}! Redirecting to preferences form.")
                st.session_state['username'] = username
                st.session_state['page'] = "preferences"
                st.rerun()
            else:
                st.error(message)
    
    st.divider()
    
    st.write("Already have an account?")
    if st.button("Login Here"):
        st.session_state['page'] = "login"
        st.rerun()

def preferences_form_page():
    """Display the user preferences form"""
    st.title("🎬 Movie Recommender - Your Preferences")
    
    if 'username' not in st.session_state:
        st.error("Please log in first")
        st.session_state['page'] = "login"
        st.rerun()
        
    username = st.session_state['username']
    st.write(f"Hello, **{username}**! Tell us about your movie preferences:")
    
    with st.form("preferences_form"):
        genre = st.multiselect(
            "Favorite genres:",
            options=["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
                     "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
                     "Romance", "Science Fiction", "Thriller", "War", "Western"],
            default=["Comedy", "Action"]
        )
        
        mood = st.radio(
            "What mood are you in today?",
            options=["Happy", "Sad", "Excited", "Relaxed", "Thoughtful", "Tense"]
        )
        
        era = st.select_slider(
            "What era of movies do you prefer?",
            options=["Classic (pre-1970)", "1970s-1980s", "1990s-2000s", "2010s-present", "Any era"]
        )
        
        length = st.radio(
            "Preferred movie length:",
            options=["Short (< 90 min)", "Average (90-120 min)", "Long (> 120 min)", "Any length"]
        )
        
        fav_actor = st.text_input("Any favorite actors or actresses? (comma-separated)")
        
        additional_info = st.text_area("Anything else you want to tell us about your tastes?", 
                                      placeholder="e.g., I like plot twists, foreign films, etc.")
        
        submit = st.form_submit_button("Save Preferences & Get Recommendations")
        
        if submit:
            # Construct preference string
            genre_str = ", ".join(genre) if genre else "any genres"
            user_input = (
                f"I like {genre_str} movies. "
                f"I'm feeling {mood.lower()} today. "
                f"I prefer movies from {era}. "
                f"I like {length.lower()} movies. "
            )
            
            if fav_actor:
                user_input += f"I like movies with {fav_actor}. "
                
            if additional_info:
                user_input += additional_info
                
            # Save to session state and database
            st.session_state['user_preferences'] = user_input
            success, message = save_user_preferences(username, user_input)
            if success:
                st.success(f"{message}! Redirecting to recommendations.")
                st.session_state['page'] = "recommendations"
                st.rerun()
            else:
                st.error(message)

def generate_movie_description(selected_movie):
    """
    Generates a detailed description of the selected movie using a local model.
    """
    generator = pipeline("text-generation", model="gpt2")  # Replace "gpt2" with your preferred local model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Prepare the context
    context = f"""
    Movie Title: {selected_movie['title']}
    Year: {selected_movie['year']}
    Genres: {', '.join(selected_movie['genres'])}
    Tags: {selected_movie['tag']}
    """

    prompt = f"""
    You are a creative and engaging movie assistant who describes the movie selected by the user according to the given context.

    The user has expressed interest in movies with emotional depth, meaningful storytelling, and engaging genres like romantic comedies or heartfelt dramas. Based on the user's preferences, craft a compelling and detailed description of the selected movie that highlights its story, genre, and unique elements. Explain why this movie aligns with the user's taste and what makes it a must-watch:

    {context}
    """

    # Truncate the prompt to fit within the model's token limit
    max_input_tokens = 1024 - 150  # Reserve 150 tokens for the generated output
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    truncated_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Generate the description
    response = generator(truncated_prompt, max_new_tokens=150, num_return_sequences=1)[0]["generated_text"]
    return response

def recommendation_page():
    st.title("🎬 Hybrid Movie Recommendations")

    if 'username' not in st.session_state:
        st.error("Please log in first")
        st.session_state['page'] = "login"
        st.rerun()

    username = st.session_state['username']

    # Load hybrid model components
    model_load = load_hybrid_model()
    if any(component is None for component in model_load):
        st.error("Failed to load hybrid recommendation model.")
        return

    movie_data, tfidf_vectorizer, tfidf_matrix, faiss_index, word2vec_model = model_load

    with st.sidebar:
        st.header("Customize Your Recommendations")
        user_query = st.text_area("Describe the type of movie you want to watch:",
                                  placeholder="e.g., action-packed adventure with a strong hero")
        top_n = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)

        if st.button("Get Recommendations"):
            if not user_query.strip():
                st.error("Please enter a valid query.")
                return

            extracted_preferences, matching_movie_indices = extract_preferences_from_llm(
                user_query=user_query,
                faiss_index=faiss_index,
                movie_embeddings=tfidf_matrix,
                k=top_n
            )

            retrieved_movies = retrieve_relevant_movies(
                extracted_preferences, movie_data, faiss_index, tfidf_vectorizer, top_k=top_n
            )

            if 'title' not in retrieved_movies.columns:
                st.error("The 'title' column is missing in the recommendations. Please check the pipeline.")
                return

            # Save results to session
            st.session_state['retrieved_movies'] = retrieved_movies
            st.session_state['user_query'] = user_query
            st.session_state['selected_movie_index'] = None

    # Check if recommendations are available
    if 'retrieved_movies' in st.session_state:
        retrieved_movies = st.session_state['retrieved_movies']
        user_query = st.session_state.get('user_query', '')

        st.subheader("🎯 Recommended Movies")
        col1, col2 = st.columns([1, 3])  # Left narrow, right wide

        with col2:
            for i, movie in retrieved_movies.iterrows():
                with st.container():
                    st.markdown(f"### {movie['title']} ({movie['year']})")
                    st.markdown(f"**Genres:** {', '.join(movie['genres'])}")
                    st.markdown(f"**Tags:** {movie['tag'][:100]}...")

                    if st.button(f"Select this", key=f"select_{i}"):
                        st.session_state['selected_movie_index'] = i

                    st.markdown("---")

        # Show selected movie description
        if st.session_state.get('selected_movie_index') is not None:
            selected_movie = retrieved_movies.iloc[st.session_state['selected_movie_index']]
            st.subheader(f"📽️ Detailed Description for '{selected_movie['title']}'")
            with st.spinner("Generating description..."):
                description = generate_movie_description(selected_movie)
                st.write(description)

# ------------------- MAIN -------------------

def main():
    """Main application entry point"""
    # Set page configuration
    st.set_page_config(
        page_title="Movie Recommendation App",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
        
    # Apply custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
    }
    .st-emotion-cache-1yycg8b {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Migrate database if needed
    migrate_database()
    
    # Create database if not exists
    create_db()
    
    # Render the appropriate page
    if st.session_state['page'] == 'login':
        login_page()
    elif st.session_state['page'] == 'registration':
        registration_page()
    elif st.session_state['page'] == 'preferences':
        preferences_form_page()
    elif st.session_state['page'] == 'recommendations':
        recommendation_page()

if __name__ == "__main__":
    main()