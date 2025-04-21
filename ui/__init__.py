# UI Package
from ui.components import movie_card, display_ratings_chart, sidebar_navigation
from ui.pages import (
    login_page,
    registration_page,
    preferences_form_page,
    recommendations_page,
    ratings_page,
    search_page
)

# Export all components and pages
__all__ = [
    'movie_card',
    'display_ratings_chart',
    'sidebar_navigation',
    'login_page',
    'registration_page',
    'preferences_form_page',
    'recommendations_page',
    'ratings_page',
    'search_page'
]