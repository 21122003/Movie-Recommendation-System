#!/usr/bin/env python
# coding: utf-8

# In[9]:


try:
    import numpy as np
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(f"Failed to import required libraries: {e}. Please ensure numpy, transformers, and sentence-transformers are installed.")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import faiss
from sklearn.metrics.pairwise import linear_kernel


# In[10]:


'''import requests
import zipfile

import os

# Download the file
url = "http://files.grouplens.org/datasets/movielens/ml-32m.zip"
filename = "ml-32m.zip"

response = requests.get(url, stream=True)
with open(filename, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)

# Extract the ZIP file
with zipfile.ZipFile(filename, "r") as zip_ref:
    zip_ref.extractall("ml-32m")

print("Download and extraction complete!")'''


# In[11]:


df=pd.read_csv('movies.csv')
df.head(5)


# In[12]:


#separtes title and year from dataset
movies = df.copy()

# rename columns
movies.rename(columns={'title': 'title_year'}, inplace=True)

# Split 'title_year' into 'title' and 'year'
movies[['title', 'year']] = movies['title_year'].str.extract(r'^(.*)\s\((\d{4})\)$')
movies.drop(columns=['title_year'], inplace=True)
# Convert 'year' to integers
movies['year'] = movies['year'].fillna(-1).astype(int)

# split genres string into a list
movies['genres'] = movies['genres'].str.split('|')
movies['genres'] = movies['genres'].apply(lambda x: [genre.lower().replace(' ', '_') for genre in x])

movies.head()


# In[13]:


'''def clean_text(text):
    """
    Cleans text by removing special characters, converting to lowercase, etc.
    """
    return text.lower()

def preprocess_movie_data(df):
    """
    Prepares movie data by cleaning missing values, formatting genres, 
    and creating a combined text column for content-based filtering.
    """
    df = df.dropna(subset=['title', 'genres'])  # Ensure no missing values in key columns
    df['genres'] = df['genres'].str.replace('|', ' ')  # Convert '|' to spaces for text processing

    # Combine only title and genres for content-based similarity
    df['combined_text'] = df['title'] + " " + df['genres']
    return df
movies=preprocess_movie_data(movies)
movies.head(5)'''


# In[14]:


def filter_movies(movies, min_year=2000, max_year=2024,genres=None, sort_by='year'):
    """
    Filters movies based on given criteria.

    Args:
        movies: DataFrame of movies.
        min_year: Minimum year of release (inclusive).
        max_year: Maximum year of release (inclusive).
        min_score: Minimum score.
        min_rating_average: Minimum average rating.
        min_rating_count: Minimum number of ratings required.
        genres: List of genres to include.
        sort_by: Column to sort by.
        top_k: Number of top movies to return.

    Returns:
        Filtered DataFrame of movies.
    """
    filtered_movies = movies.copy()

    # Filter by year
    if min_year:
        filtered_movies = filtered_movies[filtered_movies['year'] >= min_year]
    if max_year:
        filtered_movies = filtered_movies[filtered_movies['year'] <= max_year]

    # Filter by genres
    if genres:
        filtered_movies = filtered_movies[filtered_movies['genres'].apply(lambda x: any(genre.lower() in x for genre in genres))]

    # Sort the DataFrame
    if sort_by:
        
        filtered_movies.sort_values(sort_by, ascending=False, inplace=True)

    # Return the top K movies
    return filtered_movies


# In[15]:


filtered_movies=filter_movies(movies, min_year=2000, max_year=2024,genres=['Action', 'Adventure'], sort_by='year')
filtered_movies.head(5)


# In[16]:


filtered_movies.shape


# In[17]:



filtered_movie_ids = filtered_movies['movieId'].tolist()
len(filtered_movie_ids)


# In[18]:


import pandas as pd
tag=pd.read_csv('tags.csv')
tag.head(5)


# In[19]:


tag.shape


# In[20]:


# Load tag dataset

tag['tag'] = tag['tag'].str.lower().str.replace(' ', '_').astype(str)
tag.head()


# In[21]:


# Group tags by movieId and aggregate them into lists
tag_filtered = tag[tag['movieId'].isin(filtered_movie_ids)]
tag_grouped = tag_filtered.groupby('movieId')['tag'].agg(list).reset_index()

# Display the result
tag_grouped.head()


# In[22]:


tag_grouped['tag'].shape


# In[23]:


# Ensure 'title' is preserved during merging
movie_tag = filtered_movies[['movieId', 'title', 'year', 'genres']].copy().reset_index(drop=True)
movie_tag = movie_tag.merge(tag_grouped, on='movieId', how='left')

# Ensure 'tag' is converted to a string
movie_tag['tag'] = movie_tag['tag'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Ensure 'genres' is converted to a string
movie_tag['genres'] = movie_tag['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Combine 'tag' and 'genres' into the 'soup' column
movie_tag['soup'] = movie_tag['tag'] + ' ' + movie_tag['genres']

# Ensure 'title' is preserved in the final dataset
if 'title' not in movie_tag.columns:
    raise KeyError("'title' column is missing from the merged dataset. Ensure it is included during merging.")

# Display the result
movie_tag.head(5)


# In[24]:


# Create and fit the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movie_tag['soup'])
tfidf_matrix


# In[25]:


# Function to compute TF-IDF similarity
def compute_tfidf_similarity(movie_data, stop_words='english'):
    """
    Converts movie descriptions into TF-IDF vectors and computes cosine similarity.

    Parameters:
    - movie_data (DataFrame): DataFrame containing a 'combined_text' column.
    - stop_words (str or list): Stop words to be used by TfidfVectorizer. Default is 'english'.

    Returns:
    - similarity_matrix (ndarray): Cosine similarity matrix.
    - tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    if 'soup' not in movie_data.columns:
        raise ValueError("Input DataFrame must contain a 'soup' column.")

    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_data['soup'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix, tfidf_vectorizer
similarity_matrix, tfidf_vectorizer = compute_tfidf_similarity(movie_tag, stop_words='english')

print(similarity_matrix)


# In[26]:


similarity_matrix.shape


# In[28]:


import os
HF_TOKEN=os.environ.get('HF_TOKEN')
huggingface_repo_id='mistralai/Mistral-7B-Instruct-v0.3'


# In[30]:


import faiss
import numpy as np

def create_faiss_index(movie_data, embedding_model, embedding_dim=128):
    """
    Creates a FAISS index for the given movie data.

    Parameters:
    - movie_data (list of str): List of movie descriptions or tags (e.g., 'soup' column).
    - embedding_model: Pre-trained Word2Vec or GloVe model.
    - embedding_dim (int): Dimensionality of the embeddings.

    Returns:
    - faiss.IndexFlatL2: The FAISS index containing the movie embeddings.
    - ndarray: The movie embeddings used to build the index.
    """
    # Generate embeddings for each movie
    movie_embeddings = np.array([
        np.mean([embedding_model.wv[word] for word in movie.split() if word in embedding_model.wv], axis=0)
        if any(word in embedding_model.wv for word in movie.split()) else np.zeros(embedding_dim)
        for movie in movie_data
    ])

    # Create a FAISS index
    faiss_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
    faiss_index.add(movie_embeddings)  # Add the embeddings to the index

    return faiss_index, movie_embeddings


# In[31]:


# Example movie data
from gensim.models import Word2Vec

# Example movie data
movie_data = movie_tag['soup'].tolist()  # Convert the 'soup' column to a list of strings

# Train a Word2Vec model
word2vec_model = Word2Vec(sentences=[movie.split() for movie in movie_data], vector_size=128, min_count=1, workers=4)
faiss_index, movie_embeddings = create_faiss_index(movie_data, word2vec_model)


# In[32]:


def compute_query_embedding(query, word2vec_model):
    import numpy as np

    words = query.split()
    word_embeddings = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]

    print("Words in query:", words)
    print("Words found in vocabulary:", [word for word in words if word in word2vec_model.wv])
    print("Word embeddings shape:", [vec.shape for vec in word_embeddings])

    if not word_embeddings:
        print("No words found in vocabulary! Returning zero vector.")
        return np.zeros(word2vec_model.wv.vector_size)

    return np.mean(word_embeddings, axis=0)


# In[33]:


import numpy as np

def search_similar_movies_faiss(query_embedding, faiss_index, k=10):
    """
    Searches for similar movies using FAISS.

    Parameters:
    - query_embedding (ndarray): Embedding vector for the user query.
    - faiss_index (FAISS Index): Pre-built FAISS index for movie embeddings.
    - k (int): Number of nearest neighbors to retrieve.

    Returns:
    - ndarray: Similarity scores for the top-k movies.
    """
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, k)
    similarity_scores = 1 - distances.flatten()  # Convert distances to similarity scores
    return similarity_scores


# In[34]:


def extract_preferences_from_llm(user_query, faiss_index, movie_embeddings, k=10):
    """
    Extracts user preferences (genres, themes, keywords) using a local model
    and retrieves matching movie embeddings using FAISS.
    """
    # Load a local model for text generation
    generator = pipeline("text-generation", model="gpt2")  # Replace "gpt2" with your preferred model

    # Generate preferences
    prompt = (
        "You are a movie recommendation assistant. Extract genres, themes, and keywords "
        f"from this query: {user_query}"
    )
    preferences = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]

    # Parse the response into a dictionary (adjust based on the response format)
    extracted_preferences = {"genres": preferences, "themes": preferences}

    # Generate query embedding for FAISS search
    query_embedding = compute_query_embedding(user_query, word2vec_model)
    _, matching_movie_indices = faiss_index.search(query_embedding.reshape(1, -1), k)

    return extracted_preferences, matching_movie_indices.flatten()


# In[35]:


def retrieve_relevant_movies(preferences, movie_data, faiss_index, tfidf_model, top_k=10):
    query = preferences.get("genres", "") + " " + preferences.get("themes", "")
    query_embedding = compute_query_embedding(query, word2vec_model)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding.reshape(1, -1), k=top_k)

    user_tfidf_vector = tfidf_model.transform([query])
    tfidf_scores = linear_kernel(user_tfidf_vector, tfidf_matrix).flatten()

    faiss_scores = (faiss_scores.max() - faiss_scores)
    faiss_scores = faiss_scores / faiss_scores.max()

    full_faiss_scores = np.zeros_like(tfidf_scores)
    for i, idx in enumerate(faiss_indices[0]):
        full_faiss_scores[idx] = faiss_scores[0][i]

    combined_scores = tfidf_scores * 0.5 + full_faiss_scores * 0.5
    top_indices = combined_scores.argsort()[-top_k:][::-1]
    result = movie_data.iloc[top_indices].reset_index(drop=True)

    # Ensure title exists
    if 'title' not in result.columns:
        raise KeyError("'title' column is missing from the result.")

    return result



# In[36]:


from transformers import pipeline, GPT2Tokenizer

def generate_recommendations_with_rag(user_query, retrieved_movies):
    """
    Generates personalized recommendations using a local model.
    """
    # Load a local text generation model and tokenizer
    generator = pipeline("text-generation", model="gpt2")  # Replace "gpt2" with your preferred local model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Prepare the context
    context = "\n".join(retrieved_movies['soup'].tolist())
    prompt = (
        f"Given the user wants: {user_query}, and these are the top similar movies:\n{context}\n"
        "Generate a list of personalized movie suggestions with reasons."
    )

    # Truncate the prompt to fit within the model's token limit
    max_input_tokens = 1024 - 150  # Reserve 150 tokens for the generated output
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    truncated_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Generate recommendations
    response = generator(truncated_prompt, max_new_tokens=150, num_return_sequences=1)[0]["generated_text"]
    return response


# In[52]:


import subprocess
try:
    import huggingface_hub
except ImportError:
    subprocess.check_call(["pip", "install", "huggingface_hub"])


# In[54]:


user_query = "I want something like a romantic comedy with deep emotions, not just jokes."
extracted_preferences, matching_movie_indices = extract_preferences_from_llm(
    user_query=user_query,
    faiss_index=faiss_index,
    movie_embeddings=movie_embeddings,
    k=10  # Optional, defaults to 10
)

# Ensure 'title' is available in movie_data before calling retrieve_relevant_movies
movie_data = movie_tag[['movieId', 'title', 'soup', 'year', 'genres', 'tag']].copy()

retrieved_movies = retrieve_relevant_movies(extracted_preferences, movie_data, faiss_index, tfidf_vectorizer, top_k=10)
recommendations = generate_recommendations_with_rag(user_query, retrieved_movies)
print(recommendations)


# In[57]:


selected_movie = retrieved_movies.iloc[0]  # or any movie user selects
context = f"""
Movie Title: {selected_movie['movieId']}
Year: {selected_movie['year']}
Genres: {', '.join(selected_movie['genres'])}
Tags: {selected_movie['tag']}
"""


# In[58]:


prompt = f"""
You are a creative and engaging movie assistant who describes movie selected by user according context given.

The user has expressed interest in movies with emotional depth, meaningful storytelling, and engaging genres like romantic comedies or heartfelt dramas. Based on the user's preferences, craft a compelling and detailed description of the selected movie that highlights its story, genre, and unique elements. Explain why this movie aligns with the user's taste and what makes it a must-watch:

{context}
"""


# In[59]:


from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")  # Replace "gpt2" with your preferred local model

movie_description = generator(prompt, max_new_tokens=150, num_return_sequences=1)[0]["generated_text"]
print("🎬 Description:", movie_description)


# In[60]:


def load_hybrid_model():
    """
    Load the hybrid recommendation model components.
    """
    # Load movie data
    movie_data = movie_tag  # Assuming `movie_tag` is already prepared in this file
    if movie_data.empty:
        raise ValueError("Movie data is missing or could not be loaded.")

    # Load TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movie_data['soup'])

    # Train Word2Vec model
    word2vec_model = Word2Vec(
        sentences=[movie.split() for movie in movie_data['soup'].tolist()],
        vector_size=128,
        min_count=1,
        workers=4
    )

    # Create FAISS index
    faiss_index, _ = create_faiss_index(
        movie_data['soup'].tolist(), word2vec_model
    )

    return movie_data, tfidf_vectorizer, tfidf_matrix, faiss_index, word2vec_model
