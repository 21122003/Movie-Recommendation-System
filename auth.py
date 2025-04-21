import sqlite3
import hashlib
import secrets
import re
import streamlit as st
from utils import validate_email, validate_password

# ------------------- DATABASE FUNCTIONS -------------------

def get_db_connection():
    """Create and return a database connection"""
    return sqlite3.connect('data/users.db')

def hash_password(password, salt=None):
    """Hash password with a salt using SHA-256"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt

# ------------------- USER AUTHENTICATION -------------------

def get_user(username, password):
    """Authenticate user with username and password"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, username, email, password_hash, salt FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        
        if user:
            stored_hash = user[3]
            salt = user[4]
            calculated_hash, _ = hash_password(password, salt)
            
            if calculated_hash == stored_hash:
                return user
    finally:
        conn.close()
    return None

def register_user(username, email, password):
    """Register a new user with secure password storage"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash, salt = hash_password(password)
        
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)", 
            (username, email, password_hash, salt)
        )
        conn.commit()
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, f"Registration error: {str(e)}"
    finally:
        conn.close()

# ------------------- USER PREFERENCES -------------------

def save_preferences(username, preferences):
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

def save_rating(username, movie_title, rating):
    """Save or update a user's movie rating"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT id FROM user_ratings WHERE username = ? AND movie_title = ?",
            (username, movie_title)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute(
                "UPDATE user_ratings SET rating = ?, timestamp = CURRENT_TIMESTAMP WHERE username = ? AND movie_title = ?",
                (rating, username, movie_title)
            )
        else:
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