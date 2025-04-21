import streamlit as st
import pandas as pd
from auth import get_user, register_user, save_preferences, get_user_history, get_user_ratings
from utils import validate_email, validate_password, show_header, show_success_message, show_error_message
from recommendation import get_recommendations, filter_recommendations, process_user_preferences
from ui.components import movie_card, display_ratings_chart, sidebar_navigation

# ------------------- PAGE FUNCTIONS -------------------

def login_page():
    """Display an attractive login page"""
    show_header("Movie Recommender", "Sign in to your account")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("<p style='font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem;'>Login</p>", unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type='password', placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Login", use_container_width=True)
            with col2:
                register_button = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                if not username or not password:
                    show_error_message("Please enter both username and password")
                    return
                    
                user = get_user(username, password)
                if user:
                    st.session_state['user'] = user
                    st.session_state['username'] = username
                    show_success_message("Login successful! Redirecting...")
                    st.session_state['page'] = "recommendations"
                    st.rerun()
                else:
                    show_error_message("Invalid username or password")
            
            if register_button:
                st.session_state['page'] = "registration"
                st.rerun()

def registration_page():
    """Display an attractive registration page"""
    show_header("Movie Recommender", "Create your account")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("registration_form"):
            st.markdown("<p style='font-size: 1.2rem; font-weight: 500; margin-bottom: 1rem;'>Register</p>", unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type='password', placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type='password', placeholder="Confirm your password")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Register", use_container_width=True)
            with col2:
                login_button = st.form_submit_button("Back to Login", use_container_width=True)
            
            if submit:
                # Validate inputs
                if not username or not email or not password:
                    show_error_message("All fields are required")
                    return
                    
                if password != confirm_password:
                    show_error_message("Passwords do not match")
                    return
                    
                # Validate email format
                valid_email, email_msg = validate_email(email)
                if not valid_email:
                    show_error_message(email_msg)
                    return
                    
                # Validate password strength
                valid_password, password_msg = validate_password(password)
                if not valid_password:
                    show_error_message(password_msg)
                    return
                    
                # Register the user
                success, message = register_user(username, email, password)
                if success:
                    show_success_message(f"{message}! Redirecting to preferences form.")
                    st.session_state['username'] = username
                    st.session_state['page'] = "preferences"
                    st.rerun()
                else:
                    show_error_message(message)
            
            if login_button:
                st.session_state['page'] = "login"
                st.rerun()

def preferences_form_page():
    """Display an attractive user preferences form"""
    show_header("Your Movie Preferences", "Tell us what you like to watch")
    
    if 'username' not in st.session_state:
        show_error_message("Please log in first")
        st.session_state['page'] = "login"
        st.rerun()
        
    username = st.session_state['username']
    
    with st.form("preferences_form"):
        st.markdown(f"<p style='font-size: 1.2rem;'>Hello, <b>{username}</b>! Let's find your perfect movie match.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            genre = st.multiselect(
                "Favorite genres:",
                options=["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
                         "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
                         "Romance", "Science Fiction", "Thriller", "War", "Western"],
                default=["Comedy", "Action"]
            )
            
            mood = st.select_slider(
                "Current mood:",
                options=["Very Serious", "Serious", "Neutral", "Light", "Very Light"],
                value="Neutral"
            )
        
        with col2:
            length_preference = st.select_slider(
                "Preferred movie length:",
                options=["Very Short", "Short", "Average", "Long", "Very Long"],
                value="Average"
            )
            
            era_preference = st.multiselect(
                "Preferred movie eras:",
                options=["Classic (Pre-1970)", "Retro (1970-1999)", "Modern (2000-2015)", "Contemporary (2015+)"],
                default=["Modern (2000-2015)", "Contemporary (2015+)"]
            )
        
        additional_preferences = st.text_area(
            "Any specific preferences? (e.g., favorite actors, directors, themes)",
            height=100,
            placeholder="I like movies directed by Christopher Nolan, starring Tom Hanks..."
        )
        
        submit = st.form_submit_button("Find My Movies", use_container_width=True)
        
        if submit:
            preferences = f"I prefer {', '.join(genre)} movies. "
            preferences += f"I'm in a {mood.lower()} mood and prefer {length_preference.lower()} length movies. "
            preferences += f"I enjoy movies from these eras: {', '.join(era_preference)}. "
            if additional_preferences:
                preferences += f"Additional preferences: {additional_preferences}"
            
            success, message = save_preferences(username, preferences)
            if success:
                show_success_message("Preferences saved! Finding your recommendations...")
                st.session_state['preferences'] = preferences
                st.session_state['page'] = "recommendations"
                st.rerun()
            else:
                show_error_message(message)

def recommendations_page():
    """Display movie recommendations based on user preferences"""
    show_header("Your Movie Recommendations", "Personalized just for you")
    
    if 'username' not in st.session_state:
        show_error_message("Please log in first")
        st.session_state['page'] = "login"
        st.rerun()
    
    username = st.session_state['username']
    
    # Get user's most recent preferences
    history = get_user_history(username, limit=1)
    
    if not history:
        st.warning("You haven't set any preferences yet. Let's do that first!")
        st.button("Set My Preferences", on_click=lambda: setattr(st.session_state, 'page', 'preferences'))
        return
    
    preferences_text = history[0][0]
    
    # Show the preferences used for recommendations
    with st.expander("Based on these preferences", expanded=False):
        st.write(preferences_text)
    
    # Get recommendations based on preferences
    with st.spinner("Finding the perfect movies for you..."):
        retrieved_movies, recommendation_text = get_recommendations(preferences_text, top_k=10)
    
    # Display recommendations
    if not retrieved_movies.empty:
        st.subheader("Here are your personalized recommendations:")
        
        # Add filters
        with st.expander("Filter Results", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_year = st.number_input("Minimum Year", min_value=1900, max_value=2023, value=1990, step=5)
            
            with col2:
                max_year = st.number_input("Maximum Year", min_value=1900, max_value=2023, value=2023, step=5)
            
            with col3:
                sort_by = st.selectbox("Sort By", options=["year", "title"], index=0)
            
            filter_genres = st.multiselect(
                "Include Genres",
                options=["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
                         "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
                         "Romance", "Science Fiction", "Thriller", "War", "Western"]
            )
            
            apply_filters = st.button("Apply Filters")
            
            if apply_filters:
                retrieved_movies = filter_recommendations(
                    retrieved_movies,
                    min_year=min_year,
                    max_year=max_year,
                    genres=filter_genres,
                    sort_by=sort_by
                )
        
        # Display movie cards
        for i, row in retrieved_movies.iterrows():
            movie_title = row['title']
            similarity_score = 0.9 - (i * 0.05)  # Simulate decreasing similarity
            movie_card(movie_title, similarity_score, username, retrieved_movies)
    else:
        st.error("Sorry, we couldn't find any recommendations based on your preferences. Please try again with different preferences.")

def ratings_page():
    """Display user's movie ratings"""
    show_header("Your Movie Ratings", "Keep track of what you've watched")
    
    if 'username' not in st.session_state:
        show_error_message("Please log in first")
        st.session_state['page'] = "login"
        st.rerun()
    
    username = st.session_state['username']
    ratings = get_user_ratings(username)
    
    if ratings:
        # Display ratings chart
        display_ratings_chart(ratings)
        
        # Display ratings as a table
        st.subheader("Your Rating History")
        
        df = pd.DataFrame(ratings, columns=['Movie', 'Rating', 'Date'])
        df['Rating'] = df['Rating'].apply(lambda x: "⭐" * x)
        
        st.dataframe(
            df,
            column_config={
                "Movie": st.column_config.TextColumn("Movie Title"),
                "Rating": st.column_config.TextColumn("Your Rating"),
                "Date": st.column_config.DatetimeColumn("Date Rated", format="MMM DD, YYYY")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("You haven't rated any movies yet. Start exploring recommendations to rate movies!")
        
        if st.button("Find Movies to Rate"):
            st.session_state['page'] = "recommendations"
            st.rerun()

def search_page():
    """Advanced search page for finding specific movies"""
    show_header("Advanced Movie Search", "Find exactly what you're looking for")
    
    if 'username' not in st.session_state:
        show_error_message("Please log in first")
        st.session_state['page'] = "login"
        st.rerun()
    
    username = st.session_state['username']
    
    # Search form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search for movies", placeholder="Enter keywords, actors, directors...")
    
    with col2:
        search_button = st.button("Search", use_container_width=True)
    
    # Advanced filters
    with st.expander("Advanced Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_year = st.number_input("From Year", min_value=1900, max_value=2023, value=1990)
        
        with col2:
            max_year = st.number_input("To Year", min_value=1900, max_value=2023, value=2023)
        
        with col3:
            min_rating = st.slider("Minimum Rating", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        
        genres = st.multiselect(
            "Genres",
            options=["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", 
                     "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
                     "Romance", "Science Fiction", "Thriller", "War", "Western"]
        )
    
    # Process search
    if search_button and search_query:
        with st.spinner("Searching for movies..."):
            # This would be replaced with actual search functionality
            retrieved_movies, _ = get_recommendations(search_query, top_k=5)
            
            if not retrieved_movies.empty:
                st.subheader(f"Search Results for '{search_query}'")
                
                for i, row in retrieved_movies.iterrows():
                    movie_title = row['title']
                    similarity_score = 0.9 - (i * 0.05)  # Simulate decreasing similarity
                    movie_card(movie_title, similarity_score, username, retrieved_movies)
            else:
                st.info("No movies found matching your search criteria. Try different keywords or filters.")