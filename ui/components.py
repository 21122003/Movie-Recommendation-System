import streamlit as st
import pandas as pd
import plotly.express as px
from auth import save_rating

# ------------------- UI COMPONENTS -------------------

def movie_card(movie, similarity_score, username, movie_df):
    """Create an attractive UI card for a movie recommendation"""
    from recommendation import get_movie_details
    
    # Get movie details if available
    movie_details = get_movie_details(movie, movie_df)
    
    # Create a card-like container with styling
    with st.container():
        st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display poster if available
            if movie_details and "Poster" in movie_details and movie_details["Poster"] != "N/A":
                st.image(movie_details["Poster"], width=200)
            else:
                # Display placeholder with styling
                st.markdown(
                    "<div style='background-color:var(--placeholder-bg, #e2e8f0); height:250px; border-radius:10px; "
                    "display:flex; align-items:center; justify-content:center;'>"
                    "<span style='font-size:48px;'>🎬</span></div>",
                    unsafe_allow_html=True
                )
        
        with col2:
            # Title and year with better styling
            if movie_details and "Year" in movie_details:
                st.markdown(f"<h2 style='margin-bottom:0.5rem; color:var(--title-color, #1e40af);'>{movie} ({movie_details['Year']})</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='margin-bottom:0.5rem; color:var(--title-color, #1e40af);'>{movie}</h2>", unsafe_allow_html=True)
                
            # Match score with visual indicator
            score_color = "#10b981" if similarity_score > 0.7 else "#f59e0b" if similarity_score > 0.4 else "#ef4444"
            st.markdown(
                f"<div style='display:flex; align-items:center; margin-bottom:1rem;'>"
                f"<div style='background-color:{score_color}; width:{int(similarity_score*100)}px; "
                f"height:8px; border-radius:4px; margin-right:10px;'></div>"
                f"<span><b>Match:</b> {similarity_score:.2f}</span></div>",
                unsafe_allow_html=True
            )
            
            # Display movie details with better formatting
            if movie_details:
                details_html = ""
                
                if "Genre" in movie_details:
                    genres = movie_details["Genre"].split(", ")
                    genre_pills = ""
                    for genre in genres:
                        genre_pills += f"<span style='background-color:var(--genre-bg, #e0e7ff); color:var(--genre-text, #3730a3); "
                        genre_pills += f"padding:0.2rem 0.5rem; border-radius:4px; margin-right:0.3rem; "
                        genre_pills += f"font-size:0.8rem;'>{genre}</span>"
                    
                    details_html += f"<div style='margin-bottom:0.5rem;'>{genre_pills}</div>"
                
                if "Director" in movie_details:
                    details_html += f"<p style='margin-bottom:0.3rem;'><b>Director:</b> {movie_details['Director']}</p>"
                    
                if "Actors" in movie_details:
                    details_html += f"<p style='margin-bottom:0.3rem;'><b>Starring:</b> {movie_details['Actors']}</p>"
                    
                if "Plot" in movie_details and movie_details["Plot"] != "N/A":
                    details_html += f"<p style='margin-bottom:0.8rem; font-style:italic;'>\"{movie_details['Plot']}\"</p>"
                    
                if "imdbRating" in movie_details and movie_details["imdbRating"] != "N/A":
                    rating = float(movie_details["imdbRating"])
                    stars = "⭐" * int(rating/2) + ("½" if rating % 2 >= 0.5 else "")
                    details_html += f"<p><b>IMDb:</b> {stars} ({rating}/10)</p>"
                
                st.markdown(details_html, unsafe_allow_html=True)
            
            # Rating widget with better styling
            st.markdown("<div class='rating-slider'>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-weight:500; margin-bottom:0.3rem;'>Rate this movie:</p>", unsafe_allow_html=True)
            rating = st.select_slider(
                "",
                options=[1, 2, 3, 4, 5],
                value=3,
                key=f"rating_{movie}",
                format_func=lambda x: "★" * x + "☆" * (5-x)
            )
            
            if st.button("Save Rating", key=f"save_{movie}", use_container_width=True):
                success, message = save_rating(username, movie, rating)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

def display_ratings_chart(ratings):
    """Display an attractive chart of user ratings"""
    if not ratings:
        st.info("No ratings yet. Rate some movies to see your chart!")
        return
        
    # Create a DataFrame from ratings
    df = pd.DataFrame(ratings, columns=['Movie', 'Rating', 'Timestamp'])
    
    # Create a bar chart of ratings with improved styling
    fig = px.bar(
        df, 
        x='Movie', 
        y='Rating',
        title='Your Movie Ratings',
        color='Rating',
        color_continuous_scale='RdYlGn',  # Red to Green scale
        range_color=[1, 5],
        labels={'Movie': 'Movie Title', 'Rating': 'Your Rating'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=50, b=100),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=14),
        title_font=dict(family="Arial", size=18, color='#1e3a8a'),
        title_x=0.5,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )
    
    st.plotly_chart(fig, use_container_width=True)

def sidebar_navigation():
    """Create a styled sidebar navigation"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/cinema-.png", width=80)
        st.markdown("<h1 style='text-align: center; color: #1e3a8a; margin-bottom: 2rem;'>Movie Recommender</h1>", unsafe_allow_html=True)
        
        # Navigation links
        if 'username' in st.session_state:
            st.markdown(f"<p style='text-align: center; margin-bottom: 2rem;'>Welcome, <b>{st.session_state['username']}</b>!</p>", unsafe_allow_html=True)
            
            if st.button("🏠 Home", use_container_width=True):
                st.session_state['page'] = "recommendations"
                st.rerun()
                
            if st.button("🎯 My Preferences", use_container_width=True):
                st.session_state['page'] = "preferences"
                st.rerun()
                
            if st.button("⭐ My Ratings", use_container_width=True):
                st.session_state['page'] = "ratings"
                st.rerun()
                
            if st.button("🔍 Advanced Search", use_container_width=True):
                st.session_state['page'] = "search"
                st.rerun()
                
            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state['page'] = "login"
                st.rerun()
        else:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state['page'] = "login"
                st.rerun()
                
            if st.button("📝 Register", use_container_width=True):
                st.session_state['page'] = "registration"
                st.rerun()
        
        # Footer
        st.markdown("<div style='position: fixed; bottom: 20px; left: 20px; right: 20px; text-align: center;'>"
                    "<p style='color: #6b7280; font-size: 0.8rem;'>© 2023 Movie Recommender</p>"
                    "</div>", unsafe_allow_html=True)