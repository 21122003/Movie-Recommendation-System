import re
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------- UI UTILITIES -------------------

def set_page_config():
    """Set up the Streamlit page configuration with proper styling"""
    st.set_page_config(
        page_title="LifeStyle Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .main {padding-top: 1rem;}
    .stApp {background-color: var(--background-color);}
    .stButton>button {background-color: #4e67eb; color: white; border-radius: 5px; padding: 0.5rem 1rem;}
    .stButton>button:hover {background-color: #3a50c9;}
    div.stTitle {font-size: 2.5rem; font-weight: bold; color: var(--text-color);}
    div.stHeader {font-size: 2rem; font-weight: bold; color: var(--text-color);}
    .movie-card {background-color: var(--card-bg); border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: var(--card-text);}
    .rating-slider {margin-top: 1rem;}
    .sidebar .sidebar-content {background-color: var(--sidebar-color);}
    
    /* CSS variables for theme compatibility */
    :root {
        --background-color: #f5f7f9;
        --text-color: #1e3a8a;
        --card-bg: white;
        --card-text: #1f2937;
        --sidebar-color: #e0e7ff;
        --placeholder-bg: #e2e8f0;
        --title-color: #1e40af;
        --genre-bg: #e0e7ff;
        --genre-text: #3730a3;
    }
    
    /* Dark mode compatibility */
    [data-theme="dark"] {
        --background-color: #0e1117;
        --text-color: #ffffff;
        --card-bg: #262730;
        --card-text: #fafafa;
        --sidebar-color: #262730;
        --placeholder-bg: #374151;
        --title-color: #93c5fd;
        --genre-bg: #1e3a8a;
        --genre-text: #e0e7ff;
    }
    
    /* Additional dark mode overrides */
    [data-theme="dark"] .stApp {background-color: var(--background-color);}
    [data-theme="dark"] .movie-card {background-color: var(--card-bg);}
    [data-theme="dark"] p {color: var(--card-text);}
    </style>
    """, unsafe_allow_html=True)

def show_header(title, subtitle=None):
    """Display a styled header with optional subtitle"""
    st.markdown(f"<h1 class='stTitle'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='font-size: 1.2rem; color: #4b5563;'>{subtitle}</p>", unsafe_allow_html=True)
    st.divider()

def show_success_message(message, duration=3):
    """Show a success message that automatically disappears"""
    message_placeholder = st.empty()
    message_placeholder.success(message)
    # Auto-clear after duration (if using this in a loop, be careful with st.rerun())

def show_error_message(message):
    """Show an error message with consistent styling"""
    st.error(message)

# ------------------- VALIDATION UTILITIES -------------------

def validate_email(email):
    """Validate email format"""
    email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if re.match(email_pattern, email):
        return True, "Email is valid"
    return False, "Invalid email format"

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

# ------------------- DATA VISUALIZATION UTILITIES -------------------

def create_ratings_chart(ratings):
    """Create a Plotly chart for user ratings"""
    if not ratings:
        return None
        
    df = pd.DataFrame(ratings, columns=['Movie', 'Rating', 'Timestamp'])
    
    fig = px.bar(
        df, 
        x='Movie', 
        y='Rating',
        title='Your Movie Ratings',
        color='Rating',
        color_continuous_scale='RdYlGn',  # Red to Green scale
        range_color=[1, 5]
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=14),
        title_font=dict(family="Arial", size=18)
    )
    
    return fig