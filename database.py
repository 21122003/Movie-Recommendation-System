import sqlite3
import os
import hashlib
import secrets
import streamlit as st

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

def hash_password(password, salt=None):
    """Hash password with a salt using SHA-256"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt

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
            return True, "Database migrated successfully!"
        
        conn.close()
        return False, "No migration needed"
    except Exception as e:
        return False, f"Migration error: {str(e)}"

def get_user_from_db(username, password):
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

def insert_user_to_db(username, email, password):
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