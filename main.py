import streamlit as st
import os
from utils import set_page_config
from ui.components import sidebar_navigation
from ui.pages import (
    login_page,
    registration_page,
    preferences_form_page,
    recommendations_page,
    ratings_page,
    search_page
)
from database import create_db

# Ensure database exists
create_db()

# Set up page configuration and styling
set_page_config()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'

# Display sidebar navigation
sidebar_navigation()

# Route to the appropriate page based on session state
if st.session_state['page'] == 'login':
    login_page()
elif st.session_state['page'] == 'registration':
    registration_page()
elif st.session_state['page'] == 'preferences':
    preferences_form_page()
elif st.session_state['page'] == 'recommendations':
    recommendations_page()
elif st.session_state['page'] == 'ratings':
    ratings_page()
elif st.session_state['page'] == 'search':
    search_page()
else:
    # Default to login page
    st.session_state['page'] = 'login'
    st.rerun()